"""Evaluate the self-collected dataset and the CelebA 100-identity dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_database import IMAGE_SUFFIXES, build_database, create_engine, read_image
from config import SETTINGS
from face_database import FaceDatabase
from face_engine import FaceEngine, l2_normalize
from recognize import FaceRecognizer


SELF_ANNOTATIONS = SETTINGS.dataset_dir / "test" / "annotations.jsonl"
CELEBA_DIR = SETTINGS.dataset_dir / "celeba_100_identities_3reg_3test"
CELEBA_REGISTER_DIR = CELEBA_DIR / "register"
CELEBA_TEST_DIR = CELEBA_DIR / "test"
CELEBA_DATABASE_PATH = SETTINGS.output_dir / "celeba_database.npz"
DEFAULT_REPORT_PATH = SETTINGS.output_dir / "evaluation_report.json"


@dataclass
class SelfCollectedMetrics:
    images: int
    annotated_faces: int
    matched_faces: int
    extra_detections: int
    correct_faces: int
    overall_accuracy: float
    detection_recall: float
    known_faces: int
    known_correct: int
    known_accuracy: float
    unknown_faces: int
    unknown_correct: int
    unknown_recall: float
    strict_image_accuracy: float
    failures: list[dict[str, Any]]


@dataclass
class CelebAMetrics:
    identities: int
    registration_images: int
    usable_registration_images: int | None
    test_images: int
    detected_test_images: int
    correct: int
    top1_accuracy: float
    detection_recall: float
    failures: list[dict[str, Any]]


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_directory_database(
    register_dir: Path,
    output_path: Path,
    engine: FaceEngine,
) -> tuple[FaceDatabase, int]:
    """Build a database whose identity labels are the child directory names."""
    identity_dirs = sorted(path for path in register_dir.iterdir() if path.is_dir())
    if not identity_dirs:
        raise ValueError(f"No identity directories found in: {register_dir}")

    identity_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    valid_registration_images = 0

    for identity_dir in identity_dirs:
        identity_embeddings: list[np.ndarray] = []
        for path in image_paths(identity_dir):
            try:
                image = read_image(path)
                face = engine.select_primary_face(engine.detect(image))
                identity_embeddings.append(engine.extract_embedding(image, face))
                valid_registration_images += 1
            except (ValueError, cv2.error) as error:
                print(f"[WARN] CelebA registration {path}: {error}")

        if not identity_embeddings:
            raise RuntimeError(
                f"No valid registration face found for {identity_dir.name}."
            )

        identity_ids.append(identity_dir.name)
        embeddings.append(
            l2_normalize(np.mean(identity_embeddings, axis=0))
        )

    database = FaceDatabase(identity_ids, identity_ids, np.stack(embeddings))
    database.save(output_path)
    print(
        f"[DONE] CelebA database: {len(identity_ids)} identities, "
        f"{valid_registration_images} images"
    )
    return database, valid_registration_images


def bbox_iou(first: list[int], second: list[int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = [
        (bbox_iou(expected["bbox"], predicted["bbox"]), expected_index, pred_index)
        for expected_index, expected in enumerate(ground_truth)
        for pred_index, predicted in enumerate(predictions)
    ]
    used_expected: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for overlap, expected_index, pred_index in sorted(candidates, reverse=True):
        if overlap < iou_threshold:
            break
        if expected_index in used_expected or pred_index in used_predictions:
            continue
        used_expected.add(expected_index)
        used_predictions.add(pred_index)
        matches.append((expected_index, pred_index, overlap))
    return matches


def load_annotations(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_self_collected(
    database_path: Path,
    iou_threshold: float,
) -> SelfCollectedMetrics:
    recognizer = FaceRecognizer(database_path=database_path)
    annotations = load_annotations(SELF_ANNOTATIONS)

    annotated_faces = matched_faces = extra_detections = correct_faces = 0
    known_faces = known_correct = unknown_faces = unknown_correct = 0
    correct_images = 0
    failures: list[dict[str, Any]] = []

    for annotation in annotations:
        image_path = SELF_ANNOTATIONS.parent / annotation["image"]
        expected_faces = annotation["faces"]
        predictions = recognizer.recognize_file(image_path)
        matches = match_boxes(expected_faces, predictions, iou_threshold)
        prediction_for_expected = {
            expected_index: (pred_index, overlap)
            for expected_index, pred_index, overlap in matches
        }

        annotated_faces += len(expected_faces)
        matched_faces += len(matches)
        extra_detections += len(predictions) - len(matches)
        image_correct = (
            len(matches) == len(expected_faces)
            and len(matches) == len(predictions)
        )

        for expected_index, expected in enumerate(expected_faces):
            expected_id = expected["identity_id"]
            if expected_id == "unknown":
                unknown_faces += 1
            else:
                known_faces += 1

            matched = prediction_for_expected.get(expected_index)
            if matched is None:
                image_correct = False
                failures.append(
                    {
                        "image": annotation["image"],
                        "expected": expected_id,
                        "predicted": None,
                        "reason": "face_not_matched",
                    }
                )
                continue

            pred_index, overlap = matched
            prediction = predictions[pred_index]
            is_correct = prediction["identity_id"] == expected_id
            correct_faces += int(is_correct)
            image_correct = image_correct and is_correct

            if expected_id == "unknown":
                unknown_correct += int(is_correct)
            else:
                known_correct += int(is_correct)

            if not is_correct:
                failures.append(
                    {
                        "image": annotation["image"],
                        "expected": expected_id,
                        "predicted": prediction["identity_id"],
                        "similarity": prediction["similarity"],
                        "iou": round(overlap, 4),
                    }
                )

        correct_images += int(image_correct)

    return SelfCollectedMetrics(
        images=len(annotations),
        annotated_faces=annotated_faces,
        matched_faces=matched_faces,
        extra_detections=extra_detections,
        correct_faces=correct_faces,
        overall_accuracy=correct_faces / annotated_faces,
        detection_recall=matched_faces / annotated_faces,
        known_faces=known_faces,
        known_correct=known_correct,
        known_accuracy=known_correct / known_faces,
        unknown_faces=unknown_faces,
        unknown_correct=unknown_correct,
        unknown_recall=unknown_correct / unknown_faces,
        strict_image_accuracy=correct_images / len(annotations),
        failures=failures,
    )


def evaluate_celeba(
    database: FaceDatabase,
    engine: FaceEngine,
    registration_images: int,
    usable_registration_images: int | None,
) -> CelebAMetrics:
    test_identity_dirs = sorted(path for path in CELEBA_TEST_DIR.iterdir() if path.is_dir())
    register_ids = set(database.identity_ids)
    test_ids = {path.name for path in test_identity_dirs}
    if register_ids != test_ids:
        missing_test = sorted(register_ids - test_ids)
        missing_register = sorted(test_ids - register_ids)
        raise ValueError(
            "CelebA register/test identities differ. "
            f"Missing in test: {missing_test}; missing in register: {missing_register}"
        )

    test_images = detected_images = correct = 0
    failures: list[dict[str, Any]] = []

    for identity_dir in test_identity_dirs:
        expected_id = identity_dir.name
        for path in image_paths(identity_dir):
            test_images += 1
            try:
                image = read_image(path)
                face = engine.select_primary_face(engine.detect(image))
                query = engine.extract_embedding(image, face)
                similarities = database.embeddings @ query
                best_index = int(np.argmax(similarities))
                predicted_id = database.identity_ids[best_index]
                similarity = float(similarities[best_index])
                detected_images += 1
                is_correct = predicted_id == expected_id
                correct += int(is_correct)
                if not is_correct:
                    failures.append(
                        {
                            "image": str(path.relative_to(CELEBA_DIR)),
                            "expected": expected_id,
                            "predicted": predicted_id,
                            "similarity": round(similarity, 4),
                        }
                    )
            except (ValueError, cv2.error) as error:
                failures.append(
                    {
                        "image": str(path.relative_to(CELEBA_DIR)),
                        "expected": expected_id,
                        "predicted": None,
                        "reason": str(error),
                    }
                )

    return CelebAMetrics(
        identities=len(test_identity_dirs),
        registration_images=registration_images,
        usable_registration_images=usable_registration_images,
        test_images=test_images,
        detected_test_images=detected_images,
        correct=correct,
        top1_accuracy=correct / test_images,
        detection_recall=detected_images / test_images,
        failures=failures,
    )


def print_self_metrics(metrics: SelfCollectedMetrics) -> None:
    print("\n=== Self-collected 20-identity evaluation ===")
    print(f"Images:              {metrics.images}")
    print(f"Annotated faces:     {metrics.annotated_faces}")
    print(
        f"Overall accuracy:    {metrics.overall_accuracy:.2%} "
        f"({metrics.correct_faces}/{metrics.annotated_faces})"
    )
    print(f"Detection recall:    {metrics.detection_recall:.2%}")
    print(
        f"Known-face accuracy: {metrics.known_accuracy:.2%} "
        f"({metrics.known_correct}/{metrics.known_faces})"
    )
    print(
        f"Unknown recall:      {metrics.unknown_recall:.2%} "
        f"({metrics.unknown_correct}/{metrics.unknown_faces})"
    )
    print(f"Strict image acc.:   {metrics.strict_image_accuracy:.2%}")
    print(f"Extra detections:    {metrics.extra_detections}")
    print(f"Failure records:     {len(metrics.failures)}")


def print_celeba_metrics(metrics: CelebAMetrics) -> None:
    print("\n=== CelebA 100-identity closed-set evaluation ===")
    print(f"Identities:          {metrics.identities}")
    print(f"Registration images: {metrics.registration_images}")
    if metrics.usable_registration_images is not None:
        print(f"Usable reg. images:  {metrics.usable_registration_images}")
    print(f"Test images:         {metrics.test_images}")
    print(f"Detection recall:    {metrics.detection_recall:.2%}")
    print(
        f"Top-1 accuracy:      {metrics.top1_accuracy:.2%} "
        f"({metrics.correct}/{metrics.test_images})"
    )
    print(f"Failure records:     {len(metrics.failures)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("self", "celeba", "all"),
        default="all",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse existing database files instead of rebuilding them.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.30,
        help="Minimum box IoU for the self-collected dataset.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be between 0 and 1.")

    report: dict[str, Any] = {
        "settings": {
            "dataset": args.dataset,
            "recognition_threshold": SETTINGS.recognition_threshold,
            "iou_threshold": args.iou_threshold,
        }
    }

    if args.dataset in {"self", "all"}:
        if not args.skip_build:
            build_database()
        self_metrics = evaluate_self_collected(
            SETTINGS.database_path,
            args.iou_threshold,
        )
        print_self_metrics(self_metrics)
        report["self_collected"] = asdict(self_metrics)

    if args.dataset in {"celeba", "all"}:
        engine = create_engine()
        registration_images = sum(
            len(image_paths(path))
            for path in CELEBA_REGISTER_DIR.iterdir()
            if path.is_dir()
        )
        if args.skip_build:
            celeba_database = FaceDatabase.load(CELEBA_DATABASE_PATH)
            usable_registration_images = None
        else:
            celeba_database, usable_registration_images = build_directory_database(
                CELEBA_REGISTER_DIR,
                CELEBA_DATABASE_PATH,
                engine,
            )
        celeba_metrics = evaluate_celeba(
            celeba_database,
            engine,
            registration_images,
            usable_registration_images,
        )
        print_celeba_metrics(celeba_metrics)
        report["celeba_100"] = asdict(celeba_metrics)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport saved to: {args.report.resolve()}")


if __name__ == "__main__":
    main()
