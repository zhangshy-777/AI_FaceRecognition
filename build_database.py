"""Build a face embedding database from dataset/registered."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from config import SETTINGS
from face_database import FaceDatabase
from face_engine import FaceEngine, l2_normalize


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(path: str | Path) -> np.ndarray:
    """Read images safely when a Windows path contains Chinese characters."""
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def load_identity_names(path: str | Path) -> dict[str, str]:
    path = Path(path)
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                return {
                    row["identity_id"].strip(): row["name"].strip()
                    for row in csv.DictReader(file)
                }
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Cannot decode identity CSV: {path}") from last_error


def create_engine() -> FaceEngine:
    return FaceEngine(
        detector_model=SETTINGS.detector_model,
        recognizer_model=SETTINGS.recognizer_model,
        score_threshold=SETTINGS.detection_score_threshold,
        nms_threshold=SETTINGS.detection_nms_threshold,
        top_k=SETTINGS.detection_top_k,
        minimum_face_size=SETTINGS.minimum_face_size,
        max_input_size=SETTINGS.detector_max_input_size,
    )


def build_database(
    registered_dir: str | Path = SETTINGS.registered_dir,
    identities_csv: str | Path = SETTINGS.identities_csv,
    output_path: str | Path = SETTINGS.database_path,
) -> FaceDatabase:
    registered_dir = Path(registered_dir)
    identity_names = load_identity_names(identities_csv)
    engine = create_engine()

    identity_ids: list[str] = []
    names: list[str] = []
    identity_embeddings: list[np.ndarray] = []

    for identity_dir in sorted(path for path in registered_dir.iterdir() if path.is_dir()):
        identity_id = identity_dir.name
        if identity_id not in identity_names:
            print(f"[SKIP] {identity_id}: not found in identities.csv")
            continue

        embeddings: list[np.ndarray] = []
        image_paths = sorted(
            path for path in identity_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in image_paths:
            try:
                image = read_image(image_path)
                primary_face = engine.select_primary_face(engine.detect(image))
                embeddings.append(engine.extract_embedding(image, primary_face))
                print(f"[OK]   {identity_id}: {image_path.name}")
            except (ValueError, cv2.error) as error:
                print(f"[WARN] {identity_id}: {image_path.name}: {error}")

        if not embeddings:
            raise RuntimeError(f"No valid registration face found for {identity_id}.")

        identity_ids.append(identity_id)
        names.append(identity_names[identity_id])
        identity_embeddings.append(l2_normalize(np.mean(embeddings, axis=0)))
        print(f"[DONE] {identity_id}: merged {len(embeddings)} registration image(s)")

    database = FaceDatabase(identity_ids, names, np.stack(identity_embeddings))
    database.save(output_path)
    print(f"\nSaved {len(identity_ids)} identities to: {Path(output_path).resolve()}")
    return database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-dir", type=Path, default=SETTINGS.registered_dir)
    parser.add_argument("--identities-csv", type=Path, default=SETTINGS.identities_csv)
    parser.add_argument("--output", type=Path, default=SETTINGS.database_path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_database(args.registered_dir, args.identities_csv, args.output)