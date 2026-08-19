from src.arabic_rag.loader import PDFLoader
from src.arabic_rag.text_processor import ArabicTextProcessor
from src.arabic_rag.chunker import TextChunker
from src.arabic_rag.embeddings import EmbeddingModel
from src.arabic_rag.vector_store import VectorStore
from src.arabic_rag.retriever import Retriever
from src.arabic_rag.generator import AnswerGenerator


def main():
    # 1. Load PDF
    loader = PDFLoader("data/pdfs/sample.pdf")
    text = loader.load()

    # 2. Process Arabic text
    processor = ArabicTextProcessor()
    text = processor.process(text)

    # 3. Split text into chunks
    chunker = TextChunker()
    chunk_records = chunker.split_with_metadata(
        text=text,
        source_document="data/pdfs/sample.pdf",
    )
    chunks = [item["text"] for item in chunk_records]
    metadatas = [
        {
            "chunk_id": item["chunk_id"],
            "section": item["section"],
            "page": item["page"],
            "source_document": item["source_document"],
        }
        for item in chunk_records
    ]

    print(f"Total chunks: {len(chunks)}")
    print("=" * 60)

    # 4. Create embeddings
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode(chunks)

    print(f"Embedding shape: {embeddings.shape}")
    print("=" * 60)

    # 5. Create vector store
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add(embeddings, chunks, metadatas=metadatas)

    # 6. Create retriever
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    # 7. User question
    query = "من هو غازي القصيبي؟"

    print(f"Question: {query}")
    print("=" * 60)

    # 8. Retrieve relevant chunks
    dynamic_k = retriever.suggest_k(query)
    results = retriever.retrieve(query, k=dynamic_k)

    # 9. Display retrieved context
    print("RETRIEVED CONTEXT")
    print("=" * 60)

    for i, result in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Fused Score: {result.get('fused_score', result['score']):.4f}")
        print(f"Dense Score: {result.get('dense_score', 0.0):.4f}")
        print(f"Sparse Score: {result.get('sparse_score', 0.0):.4f}")
        if result.get("chunk_id") is not None:
            print(f"Chunk ID: {result['chunk_id']}")
        metadata = result.get("metadata", {})
        if metadata.get("section"):
            print(f"Section: {metadata['section']}")
        print("-" * 40)
        print(result["chunk"])
        print("=" * 60)

    # 10. Build context
    context = "\n\n".join(
        result["chunk"]
        for result in results
    )

    # 11. Generate answer
    generator = AnswerGenerator()

    answer = generator.generate(
        question=query,
        context=context,
        retrieved_chunks=results,
    )

    print("\nAnswer:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()