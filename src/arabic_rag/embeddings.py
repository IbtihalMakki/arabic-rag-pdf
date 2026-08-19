import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Embedding device: {self.device}")

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=self.device,
        )

    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )