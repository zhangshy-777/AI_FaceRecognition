"""Face detection, alignment and embedding extraction."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import cv2
import numpy as np


@dataclass
class DetectedFace:
    """A detected face and the information required by SFace alignment."""

    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray
    score: float
    raw_detection: np.ndarray

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "landmarks": self.landmarks.round(2).tolist(),
            "detection_score": round(self.score, 4),
        }


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length face embedding.")
    return vector / norm


class FaceEngine:
    """Local YuNet detector and SFace recognizer wrapper."""

    def __init__(
        self,
        detector_model: str | Path,
        recognizer_model: str | Path,
        score_threshold: float = 0.85,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
        minimum_face_size: int = 40,
        max_input_size: int | None = 960,
    ) -> None:
        self.detector_model = Path(detector_model)
        self.recognizer_model = Path(recognizer_model)
        self.minimum_face_size = minimum_face_size
        self.max_input_size = max_input_size
        self._validate_model_files()
        detector_model_path = self._opencv_model_path(self.detector_model)
        recognizer_model_path = self._opencv_model_path(self.recognizer_model)

        self.detector = cv2.FaceDetectorYN.create(
            detector_model_path,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            recognizer_model_path,
            "",
        )

    def _validate_model_files(self) -> None:
        missing = [
            path
            for path in (self.detector_model, self.recognizer_model)
            if not path.is_file()
        ]
        if missing:
            paths = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "Required local model files are missing:\n"
                f"{paths}\n"
                "See model/README.md for download instructions."
            )

    @staticmethod
    def _opencv_model_path(path: Path) -> str:
        """Return a model path OpenCV can read on Windows with Unicode folders."""
        path = path.resolve()
        try:
            relative_path = path.relative_to(Path.cwd().resolve())
            relative_text = str(relative_path)
            if relative_text.isascii():
                return relative_text
        except ValueError:
            pass

        if str(path).isascii():
            return str(path)

        cache_dir = Path(tempfile.gettempdir()) / "ai_face_recognition_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = cache_dir / path.name
        if (
            not cached_path.exists()
            or cached_path.stat().st_size != path.stat().st_size
            or cached_path.stat().st_mtime_ns < path.stat().st_mtime_ns
        ):
            shutil.copy2(path, cached_path)
        return os.fspath(cached_path)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        self._validate_image(image)
        height, width = image.shape[:2]
        scale = 1.0
        detection_image = image
        if self.max_input_size and max(width, height) > self.max_input_size:
            scale = self.max_input_size / max(width, height)
            detection_image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        detection_height, detection_width = detection_image.shape[:2]
        self.detector.setInputSize((detection_width, detection_height))
        _, detections = self.detector.detect(detection_image)

        if detections is None:
            return []

        faces: list[DetectedFace] = []
        for row in detections:
            row = np.asarray(row, dtype=np.float32).copy()
            if scale != 1.0:
                row[:14] /= scale

            x, y, w, h = row[:4]
            if min(w, h) < self.minimum_face_size:
                continue

            bbox = self._clip_bbox(x, y, w, h, width, height)
            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            faces.append(
                DetectedFace(
                    bbox=bbox,
                    landmarks=landmarks,
                    score=float(row[14]),
                    raw_detection=row,
                )
            )
        return faces

    def extract_embedding(
        self,
        image: np.ndarray,
        face: DetectedFace,
    ) -> np.ndarray:
        self._validate_image(image)
        aligned_face = self.recognizer.alignCrop(image, face.raw_detection)
        embedding = self.recognizer.feature(aligned_face)
        return l2_normalize(embedding)

    def detect_and_embed(
        self,
        image: np.ndarray,
    ) -> list[tuple[DetectedFace, np.ndarray]]:
        return [(face, self.extract_embedding(image, face)) for face in self.detect(image)]

    @staticmethod
    def select_primary_face(faces: Iterable[DetectedFace]) -> DetectedFace:
        faces = list(faces)
        if not faces:
            raise ValueError("No usable face was detected in the image.")
        return max(
            faces,
            key=lambda face: (face.score, face.bbox[2] * face.bbox[3]),
        )

    @staticmethod
    def _clip_bbox(
        x: float,
        y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        left = max(0, int(round(x)))
        top = max(0, int(round(y)))
        right = min(image_width, int(round(x + width)))
        bottom = min(image_height, int(round(y + height)))
        return left, top, max(0, right - left), max(0, bottom - top)

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image is empty or invalid.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a BGR image with three channels.")