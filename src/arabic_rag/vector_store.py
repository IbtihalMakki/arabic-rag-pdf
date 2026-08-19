import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []
        self.metadatas = []

    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[str],
        metadatas: list[dict] | None = None,
    ):
        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

        if metadatas is None:
            metadatas = [{} for _ in chunks]

        self.metadatas.extend(metadatas)

    def search(self, query_embedding: np.ndarray, k: int = 3):
        query_embedding = np.array([query_embedding]).astype("float32")

        if len(self.chunks) == 0:
            return []

        k = max(1, min(k, len(self.chunks)))

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append(
                {
                    "index": int(idx),
                    "chunk_id": int(idx) + 1,
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                }
            )

        return results