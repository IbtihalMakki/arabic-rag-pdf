# Arabic PDF RAG — Hybrid Retrieval, Grounded QA & Safe Refusal

End-to-end Arabic PDF question-answering system with hybrid dense-sparse retrieval, multi-layer grounding validation, and safe refusal. Designed around the specific challenges of Arabic text in PDFs: encoding corruption, RTL rendering artifacts, Unicode glyph variation, fragmented line extraction, and diacritic inconsistency between query and corpus vocabulary.

## Why This Project Matters

Arabic RAG systems can fail before generation begins — malformed PDF text, RTL artifacts, vocabulary mismatch, and weak evidence retrieval can all propagate into hallucinated answers. This project treats extraction, retrieval, generation, and grounding as separate engineering layers and validates each layer independently.

## Problem

Answering factual questions about content in Arabic documents requires:
- Robust PDF text extraction handling encoding issues, Unicode mapping defects, RTL artifacts, and fragmented text layers.
- Proper normalization of Arabic text (handling diacritics, various alef forms, character variants)
- Effective chunking that preserves document structure and references
- Intelligent retrieval combining dense semantic search with sparse lexical matching
- Answer grounding that ensures responses are faithful to source material
- Safe refusal when the model cannot generate trustworthy answers

This system provides end-to-end solutions for each challenge.

## Architecture

The pipeline is designed around Arabic-specific extraction and retrieval challenges:

```
PDF
  → Quality-gated extraction   (PyMuPDF + pypdf, score-selected)
  → Arabic normalization        (diacritics, alef unification, tatweel)
  → Conservative text repair    (line-break fragment joining)
  → Metadata-aware chunking     (section / page / source tracking)
  → Dense indexing              (FAISS + sentence-transformers)
  → Sparse indexing             (BM25-style Arabic tokenization)
  → Query analysis              (intent, entity, key-term extraction)
  → Hybrid retrieval            (RRF fusion of dense + sparse)
  → Instruction-tuned LLM       (Qwen2.5-1.5B-Instruct)
  → Grounding validation        (source quote / numeric / supported-flag)
  → Grounded answer or safe refusal
```
## Engineering Highlights

- Dual PDF extraction with quality-based source selection
- Arabic Unicode normalization and conservative text repair
- Hybrid dense + sparse retrieval with Reciprocal Rank Fusion
- Intent-aware query expansion and dynamic top-k retrieval
- Evidence-first answer validation with safe refusal
- Regression suite with 44 automated tests
- Modular architecture allowing independent replacement of retrieval,
  embedding, and generation components
  
### Key Components

#### 1. **PDF Extraction** (`loader.py`)
- Runs both **PyMuPDF** and **pypdf** extractors on every document
- Selects the higher-quality result using a Unicode suspicious-glyph ratio score — avoiding hard-coded extractor priority
- Explicitly handles Arabic presentation forms (U+FB50–U+FDFF, U+FE70–U+FEFF), RTL bidi control marks, and common PDF text-layer corruption patterns
- Content-stream order (sort=False) used for PyMuPDF to avoid RTL line reshuffling artifacts

#### 2. **Text Normalization** (`text_processor.py`)
- Removes diacritics (tashkeel)
- Unifies various alef forms (ا, أ, إ, آ)
- Removes tatweel characters (ـــ)
- Conservative line-break repair: joins obvious 1-2 letter fragments
- Preserves source integrity for grounding

#### 3. **Chunking** (`chunker.py`)
- Structure-aware chunking with metadata preservation
- Tracks section, page, and document source for each chunk
- Filters out reference sections (المصادر) and URLs
- Maintains chunk references for accurate grounding

#### 4. **Embedding** (`embeddings.py`)
- Multilingual dense embeddings using `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
- Chosen for multilingual Arabic-English coverage at efficient inference size
- GPU-accelerated via CUDA with automatic CPU fallback
- Batch encoding with progress reporting

#### 5. **Retrieval** (`retriever.py`, `vector_store.py`)
- **Dense Retrieval**: FAISS vector similarity search
- **Sparse Retrieval**: BM25-style lexical matching with Arabic tokenization
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) combines dense and sparse rank lists — more robust than score-level fusion because Arabic PDF text fragmentation means dense and sparse signals are complementary rather than redundant
- **Intent-Aware Expansion**: Detects query type (temporal, location, entity, list, etc.) and expands with related terms
- **Dynamic Top-K**: Adjusts retrieval count based on query complexity

#### 6. **Query Analysis** (`query_analysis.py`)
- Extracts intents: temporal references, locations, entities, list requests, education/work history
- Enables context-aware retrieval adjustments

#### 7. **Grounding Validation** (`generator.py`)
- **Source Quotation Validation**: Ensures generated quotes appear verbatim in source material
- **Numeric Claim Validation**: Checks that factual numbers in answers exist in evidence
- **Supported Flag Check**: Verifies answer claim is marked as supported by evidence
- **Hallucination Detection**: Claim-level grounding checks reject answers whose cited evidence or factual claims cannot be validated against retrieved source text.
- **Safe Refusal**: Returns "I cannot answer this question based on the provided documents" when confidence is insufficient

#### 8. **Answer Generation** (`generator.py`)
- Instruction-tuned language model inference using **Qwen2.5-1.5B-Instruct** (no fine-tuning performed in this repository)
- Structured prompt construction with retrieved context windows
- JSON-formatted output with explicit `supported`, `answer`, `sources`, and `evidence` fields
- All outputs pass through grounding validation before being returned

## Installation

### Requirements
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration, optional but recommended)
- 4GB+ RAM
- ~2GB disk space for model downloads

### Setup

```bash
# Clone repository
git clone https://github.com/IbtihalMakki/arabic-rag-pdf.git
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
- `test_pdf_extraction.py` — PDF loading and quality-gated extractor selection
- `test_text_repair.py` — Arabic normalization and conservative line-break repair
- `test_rag_grounding.py` — Grounding validation: source quotation, numeric, supported-flag checks
- `test_rag_quality.py` — End-to-end pipeline quality on the sample document
- `test_generator_unsupported_questions.py` — Safe refusal: verifies the pipeline refuses rather than hallucinating when evidence is absent
- `test_vocab_bridge.py` — Vocabulary expansion module and grounding safety guards

Validated safety properties (evaluated on the included sample dataset):
- **Refusal accuracy: 100%** — no unsafe answers generated when evidence is absent
- **Unsafe answer rate: 0%** — all hallucinated or unsupported claims blocked by grounding validation

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
- **Transformers (Hugging Face)**: Qwen2.5-1.5B-Instruct inference
- **sentence-transformers**: `paraphrase-multilingual-MiniLM-L12-v2` dense embeddings
- **FAISS**: Approximate nearest-neighbour vector search
- **PyMuPDF**: Primary PDF text extraction
- **pypdf**: Secondary PDF extractor (quality-score selection)

## Runtime Characteristics

- GPU acceleration supported through PyTorch/CUDA.
- CPU fallback available.
- FAISS provides efficient vector similarity search.
- Embeddings and LLM weights are loaded on demand.

## Contributing

Contributions welcome. Please ensure:
1. All tests pass: `python -m unittest discover -s tests`
2. Code follows project style conventions
3. New features include test coverage
4. Documentation is updated

## License

A LICENSE file has not yet been added to this repository. Check the repository for current license status before use.

## Citation

If you use this system in research, please cite:

```bibtex
@software{arabic_rag_pdf_2026,
  title  = {Arabic PDF RAG --- Hybrid Retrieval, Grounded QA and Safe Refusal},
  author = {Makki, Ibtihal},
  year   = {2026},
  url    = {https://github.com/IbtihalMakki/arabic-rag-pdf}
}
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.
