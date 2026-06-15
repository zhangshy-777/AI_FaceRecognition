"""Persistence and nearest-neighbor matching for registered identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from face_engine import l2_normalize


@dataclass(frozen=True)
class MatchResult:
    identity_id: str
    name: str
    similarity: float
    accepted: bool


class FaceDatabase:
    def __init__(
        self,
        identity_ids: list[str],
        names: list[str],
        embeddings: np.ndarray,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("Database embeddings must be a two-dimensional array.")
        if len(identity_ids) != len(names) or len(identity_ids) != len(embeddings):
            raise ValueError("Identity IDs, names and embeddings have different lengths.")
        if not identity_ids:
            raise ValueError("Face database cannot be empty.")

        self.identity_ids = list(identity_ids)
        self.names = list(names)
        self.embeddings = np.stack([l2_normalize(item) for item in embeddings])

    def match(self, query_embedding: np.ndarray, threshold: float) -> MatchResult:
        query = l2_normalize(query_embedding)
        similarities = self.embeddings @ query
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])
        accepted = best_score >= threshold

        if not accepted:
            return MatchResult(
                identity_id="unknown",
                name="未知人物",
                similarity=best_score,
                accepted=False,
            )

        return MatchResult(
            identity_id=self.identity_ids[best_index],
            name=self.names[best_index],
            similarity=best_score,
            accepted=True,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            identity_ids=np.asarray(self.identity_ids, dtype=np.str_),
            names=np.asarray(self.names, dtype=np.str_),
            embeddings=self.embeddings.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FaceDatabase":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Face database does not exist: {path}\n"
                "Run build_database.py before recognition."
            )
        with np.load(path, allow_pickle=False) as data:
            return cls(
                identity_ids=data["identity_ids"].astype(str).tolist(),
                names=data["names"].astype(str).tolist(),
                embeddings=data["embeddings"],
            )

