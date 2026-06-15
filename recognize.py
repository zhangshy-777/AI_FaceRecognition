"""Public face-recognition API and command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from build_database import create_engine, read_image
from config import SETTINGS
from face_database import FaceDatabase
from face_engine import FaceEngine


class FaceRecognizer:
    """The integration point used by evaluation code and a future frontend."""

    def __init__(
        self,
        database_path: str | Path = SETTINGS.database_path,
        threshold: float = SETTINGS.recognition_threshold,
        engine: FaceEngine | None = None,
    ) -> None:
        self.engine = engine or create_engine()
        self.database = FaceDatabase.load(database_path)
        self.threshold = threshold

    def recognize_image(self, image: np.ndarray) -> list[dict]:
        results: list[dict] = []
        for face, embedding in self.engine.detect_and_embed(image):
            match = self.database.match(embedding, self.threshold)
            results.append(
                {
                    "identity_id": match.identity_id,
                    "name": match.name,
                    "bbox": list(face.bbox),
                    "similarity": round(match.similarity, 4),
                    "detection_score": round(face.score, 4),
                }
            )
        return results

    def recognize_file(self, image_path: str | Path) -> list[dict]:
        return self.recognize_image(read_image(image_path))

    @staticmethod
    def draw_results(image: np.ndarray, results: list[dict]) -> np.ndarray:
        """Optional rendering helper for the teammate implementing the frontend."""
        rendered = image.copy()
        for result in results:
            x, y, width, height = result["bbox"]
            known = result["identity_id"] != "unknown"
            color = (40, 180, 40) if known else (30, 30, 220)
            cv2.rectangle(rendered, (x, y), (x + width, y + height), color, 2)

            label = (
                f'{result["identity_id"]} '
                f'{result["name"]} '
                f'{result["similarity"]:.3f}'
            )
            # OpenCV's default font may not render Chinese. The identity ID and
            # score remain visible; a frontend can render the name with Pillow.
            cv2.putText(
                rendered,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--database", type=Path, default=SETTINGS.database_path)
    parser.add_argument("--threshold", type=float, default=SETTINGS.recognition_threshold)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recognizer = FaceRecognizer(args.database, args.threshold)
    predictions = recognizer.recognize_file(args.image)
    print(json.dumps(predictions, ensure_ascii=False, indent=2))
