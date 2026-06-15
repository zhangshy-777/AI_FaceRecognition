"""Project-wide paths and algorithm parameters."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    dataset_dir: Path = PROJECT_ROOT / "dataset"
    registered_dir: Path = dataset_dir / "registered"
    identities_csv: Path = dataset_dir / "identities.csv"

    model_dir: Path = PROJECT_ROOT / "model"
    detector_model: Path = model_dir / "face_detection_yunet_2023mar.onnx"
    recognizer_model: Path = model_dir / "face_recognition_sface_2021dec.onnx"

    output_dir: Path = PROJECT_ROOT / "output"
    database_path: Path = output_dir / "face_database.npz"

    detection_score_threshold: float = 0.50
    detection_nms_threshold: float = 0.30
    detection_top_k: int = 5000
    detector_max_input_size: int = 960

    # This is an initial open-set threshold. It should be finalized on a
    # validation set, not on the submitted test set.
    recognition_threshold: float = 0.45
    minimum_face_size: int = 40


SETTINGS = Settings()