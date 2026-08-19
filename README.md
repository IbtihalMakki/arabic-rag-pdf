# Arabic Retrieval-Augmented Generation (RAG) System

A complete Python implementation of a retrieval-augmented generation system for question answering over Arabic PDF documents. The system extracts text from PDFs, processes and chunks content, retrieves relevant passages using hybrid dense-sparse retrieval, and generates grounded answers using a fine-tuned language model.

## Problem

Answering factual questions about content in Arabic documents requires:
- Robust PDF text extraction handling OCR artifacts and encoding issues
- Proper normalization of Arabic text (handling diacritics, various alef forms, character variants)
- Effective chunking that preserves document structure and references
- Intelligent retrieval combining dense semantic search with sparse lexical matching
- Answer grounding that ensures responses are faithful to source material
- Safe refusal when the model cannot generate trustworthy answers

This system provides end-to-end solutions for each challenge.

## Architecture

The pipeline follows a standard RAG architecture with Arabic-specific optimizations:

```
PDF → Extract → Normalize → Chunk → Index → Retrieve → Ground → Generate
```

### Key Components

#### 1. **PDF Extraction** (`loader.py`)
- Dual-extractor approach: PyMuPDF (preferred) with PyPDF2 fallback
- Quality selection: compares extraction quality and selects best result
- Handles corrupted PDFs and encoding issues gracefully

#### 2. **Text Normalization** (`text_processor.py`)
- Removes diacritics (tashkeel)
- Unifies various alef forms (ا, أ, إ, آ)
- Removes tatweel characters (ـــ)
- Conservative line-break repair: joins obvious 1-2 letter fragments
- Preserves source integrity for grounding

#### 3. **Chunking** (`chunker.py`)
- Semantic chunking with metadata preservation
- Tracks section, page, and document source for each chunk
- Filters out reference sections (المصادر) and URLs
- Maintains chunk references for accurate grounding

#### 4. **Embedding** (`embeddings.py`)
- Dense embeddings via transformer-based model
- GPU-accelerated encoding
- Supports batch processing

#### 5. **Retrieval** (`retriever.py`, `vector_store.py`)
- **Dense Retrieval**: FAISS vector similarity search
- **Sparse Retrieval**: BM25-style lexical matching with Arabic tokenization
- **Hybrid Fusion**: RRF (Reciprocal Rank Fusion) combines both signals
- **Intent-Aware Expansion**: Detects query type (temporal, location, entity, list, etc.) and expands with related terms
- **Dynamic Top-K**: Adjusts retrieval count based on query complexity

#### 6. **Query Analysis** (`query_analysis.py`)
- Extracts intents: temporal references, locations, entities, list requests, education/work history
- Enables context-aware retrieval adjustments

#### 7. **Grounding Validation** (`generator.py`)
- **Source Quotation Validation**: Ensures generated quotes appear verbatim in source material
- **Numeric Claim Validation**: Checks that factual numbers in answers exist in evidence
- **Supported Flag Check**: Verifies answer claim is marked as supported by evidence
- **Hallucination Detection**: Catches model-generated claims not in source and rejects them
- **Safe Refusal**: Returns "I cannot answer this question based on the provided documents" when confidence is insufficient

#### 8. **Answer Generation** (`generator.py`)
- Fine-tuned language model inference (Qwen 2.5 1.5B Instruct)
- Template-based prompt construction with retrieved context
- JSON-formatted structured output with source attribution

## Installation

### Requirements
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration, optional but recommended)
- 4GB+ RAM
- ~2GB disk space for model downloads

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/arabic-rag-pdf.git
cd arabic-rag-pdf

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model files (automatic on first use)
python app.py
```

### GPU Support
For CUDA GPU acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

## Usage

### Quick Example: Interactive Question Answering

```bash
python app.py
```

This runs the complete pipeline on the sample PDF and answers a sample question.

### Trace Pipeline Execution

See detailed execution traces showing every step:

```bash
python rag_trace.py
```

Output includes:
- Raw and normalized text
- Repair operations applied
- Chunks extracted
- Retrieved results with scores
- Model input and output
- Grounding validation decisions
- Final answer

### Quality Audit

Run comprehensive quality assessment:

```bash
python rag_quality_audit.py
```

Tests the system on a question dataset and reports:
- Overall answerable question coverage
- Extraction quality metrics
- Grounding validation success rates
- Per-question detailed results

## Testing

Run the full test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Individual test modules:
- `test_pdf_extraction.py` - PDF loading and dual-extractor quality selection
- `test_text_repair.py` - Text normalization and line-break repair
- `test_rag_grounding.py` - Grounding validation logic
- `test_rag_quality.py` - End-to-end pipeline quality
- `test_generator_unsupported_questions.py` - Safe refusal behavior
- `test_vocab_bridge.py` - Vocabulary expansion module

## Project Structure

```
.
├── app.py                           # Main entry point: full pipeline demo
├── rag_trace.py                     # Execution trace with diagnostics
├── rag_quality_audit.py             # Quality assessment tool
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── src/arabic_rag/                  # Core RAG library
│   ├── loader.py                    # PDF extraction
│   ├── text_processor.py            # Arabic text normalization
│   ├── chunker.py                   # Text chunking
│   ├── embeddings.py                # Dense embeddings
│   ├── vector_store.py              # FAISS vector storage
│   ├── retriever.py                 # Hybrid retrieval (dense + sparse)
│   ├── query_analysis.py            # Query intent extraction
│   ├── generator.py                 # LLM inference + grounding
│   ├── vocab_bridge.py              # Vocabulary expansion (optional)
│   └── __init__.py
│
├── tests/                           # Test suite (44 tests)
│   ├── test_pdf_extraction.py
│   ├── test_text_repair.py
│   ├── test_rag_grounding.py
│   ├── test_rag_quality.py
│   ├── test_generator_unsupported_questions.py
│   └── test_vocab_bridge.py
│
├── scripts/                         # Utility and diagnostic scripts
│   ├── rag_trace.py                 # Backend trace implementation
│   ├── rag_quality_audit.py         # Backend audit implementation
│   └── _diag_*.py                   # Internal diagnostics
│
└── data/
    └── pdfs/
        └── sample.pdf               # Sample Arabic PDF for testing
```

## Key Features

### PDF Extraction Quality
- Automatic fallback between multiple extraction methods
- Quality comparison ensures best text extraction
- Handles corrupted and problematic PDFs

### Arabic Text Processing
- Proper handling of diacritics and character variants
- Conservative repair of line-break artifacts
- Preserves original text for accurate grounding

### Hybrid Retrieval
- Combines dense semantic similarity with lexical matching
- Intent-aware query expansion for better recall
- Dynamic top-k selection based on query complexity
- Reciprocal Rank Fusion (RRF) for balanced scoring

### Grounding & Safety
- Source-faithful answer generation
- Automatic rejection of hallucinated claims
- Safe refusal when confident answer cannot be generated
- Numeric validation ensures factual claims match evidence

### Extensibility
- Modular component design
- Support for different embedding models
- Pluggable vector stores (currently FAISS)
- Customizable language models (via transformer interface)

## Technologies

- **Python 3.10**: Core language
- **PyTorch**: Deep learning framework
- **Transformers**: LLM and embedding models
- **FAISS**: Vector similarity search
- **PyMuPDF**: Advanced PDF extraction
- **PyPDF2 (pypdf)**: Fallback PDF extraction

## Performance Characteristics

- Embedding generation: GPU-accelerated (CUDA) or CPU fallback
- FAISS retrieval: Sub-millisecond similarity search
- Model inference: Optimized for 1.5B parameter models
- Memory footprint: ~2GB for embeddings + model weights

## Contributing

Contributions welcome. Please ensure:
1. All tests pass: `python -m unittest discover -s tests`
2. Code follows project style conventions
3. New features include test coverage
4. Documentation is updated

## License

MIT License - see LICENSE file for details

## Citation

If you use this system in research, please cite:

```
@software{arabic_rag_2024,
  title = {Arabic Retrieval-Augmented Generation System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/arabic-rag-pdf}
}
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.
