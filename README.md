---
title: LLM TestLab Playground
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---
<p align="center">
  <a href="https://github.com/Saivineeth147/llm-testlab/stargazers">
    <img src="https://img.shields.io/github/stars/Saivineeth147/llm-testlab?style=social" alt="Star this repo" />
  </a>
</p>

⭐ If you find this project useful, **please consider starring** it — it helps others discover it!

# LLM TestLab

**Comprehensive Testing Suite for Large Language Models**

A flexible Python toolkit for evaluating LLMs on:
- **Text Metrics**: Hallucination, consistency, semantic robustness, safety
- **Code Evaluation**: Syntax, execution, quality, security, semantic correctness across 9+ languages
- **Dual Embedders**: Optimized for both text and code analysis
- **Optional FAISS**: High-performance vector similarity

## Features

### Text Evaluation Metrics
- **Hallucination Severity Index (HSI)** – Detect factual deviations from knowledge base
- **Consistency Stability Score (CSS)** – Measure output stability across runs
- **Semantic Robustness Index (SRI)** – Test invariance to paraphrasing
- **Safety Vulnerability Exposure (SVE)** – Detect unsafe responses to adversarial prompts
- **Knowledge Base Coverage (KBC)** – Measure factual alignment

### Code Evaluation Metrics (9+ Languages)
- **Syntax Validity (SV)** – Compiler/interpreter-based validation
- **Execution Pass Rate (EPR)** – Test case execution and verification
- **Code Quality Score (CQS)** – Complexity, documentation, error handling
- **Security Risk Score (SRS)** – Vulnerability pattern detection
- **Semantic Code Correctness (SCC)** – Embedding-based similarity to reference
- **Comprehensive Code Evaluation (CCE)** – Weighted aggregation of all metrics

**Supported Languages**: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, Ruby, PHP

### Advanced Features
- **Dual Embedders**: `all-MiniLM-L6-v2` for text, `BAAI/bge-small-en-v1.5` for code
- **FAISS Support**: Optional, for faster similarity searches
- **Knowledge Base Management**: Add, remove, or list facts
- **Security Patterns**: Customizable keywords and regex patterns
- **Rich Logging**: Built-in debug/info logging

## Project Structure

```
llm-testlab/
├── llm_testing_suite/
│   ├── __init__.py          
│   ├── llm_testing_suite.py    # Main test suite (text metrics)
│   └── code_evaluator.py       # Code evaluation module
├── examples/
│   ├── run_text_evaluation.py  # Text metrics evaluation script
│   ├── run_code_evaluation.py  # Code metrics evaluation script
│   ├── groq_example.py         # Groq API text evaluation
│   ├── groq_code_evaluation.py # Groq API code evaluation
│   └── huggingface_example.py  # HuggingFace integration
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies
├── README.md
├── LICENSE
└── .gitignore

```
## Installation

### From PyPI
```bash
pip install llm-testlab
```

### From Source
```bash
git clone https://github.com/Saivineeth147/llm-testlab.git
cd llm-testlab
pip install .
```

### Optional Dependencies
```bash
# With FAISS and HuggingFace support
pip install llm-testlab[faiss,huggingface]

# Or install individually
pip install faiss-cpu  # or faiss-gpu
pip install transformers
```

## Quick Start

### Text Metrics Example
```python
from llm_testing_suite import LLMTestSuite

def my_llm(prompt):
    return "Rome is the capital of Italy"

# Initialize with FAISS support
suite = LLMTestSuite(my_llm, use_faiss=True)

# Add knowledge
suite.add_knowledge("Rome is the capital of Italy")

# Run all novel metrics
result = suite.run_all_novel_metrics(
    prompt="What is the capital of Italy?",
    paraphrases=["Italy's capital?", "Capital city of Italy?"],
    adversarial_prompts=["ignore previous instructions"],
    runs=3
)

print(f"HSI: {result['HSI']['HSI']:.4f}")           # Hallucination
print(f"CSS: {result['CSS']['CSS']:.4f}")           # Consistency  
print(f"SRI: {result['SRI']['SRI']:.4f}")           # Robustness
print(f"SVE: {result['SVE']['SVE']:.4f}")           # Safety
print(f"KBC: {result['KBC']['KBC']:.4f}")           # Coverage
```

### Code Evaluation Example
```python
from llm_testing_suite import LLMTestSuite

def code_llm(prompt):
    return '''
def add(a, b):
    """Add two numbers."""
    return a + b
    
print(add(5, 3))
'''

suite = LLMTestSuite(code_llm)

# Comprehensive code evaluation
result = suite.comprehensive_code_evaluation(
    prompt="Write a function to add two numbers",
    code_response=code_llm("..."),
    test_cases=[
        {"input": "", "expected_output": "8"}
    ],
    language="python"
)

print(f"Overall Score: {result['overall_score']:.1f}/100")
print(f"Syntax Valid: {result['syntax_valid']}")
print(f"Quality Score: {result['quality_score']}/100")
print(f"Security: {'✓' if result['is_secure'] else '✗'}")
```

## Managing Knowledge Base

```python
# Add a single fact
suite.add_knowledge("New York is the largest city in the USA")

# Add multiple facts
suite.add_knowledge_bulk([
    "Python is a programming language",
    "AI is transforming industries"
])

# List knowledge base
suite.list_knowledge()

# Remove a fact
suite.remove_knowledge("Python is a programming language")

# Clear the knowledge base
suite.clear_knowledge()
```

## Managing Security Keywords

```python
# Add malicious keywords
suite.add_malicious_keywords(["hack system", "steal data"])

# List keywords
suite.list_malicious_keywords()

# Remove a keyword
suite.remove_malicious_keyword("hack system")
```

# List keywords
tester.list_malicious_keywords()

# Remove a keyword
tester.remove_malicious_keyword("hack system")

Output Format
-------------

    All test methods support three return types controlled by the `return_type` parameter: `"dict"`, `"table"`, or `"both"`.

- `"dict"`: Returns a Python dictionary with the test results.  
- `"table"`: Prints a formatted table using the `rich` library, no dictionary returned.  
- `"both"`: Returns the dictionary **and** prints the table.

## Code Evaluation Details

### Individual Metrics

```python
from llm_testing_suite import LLMTestSuite

suite = LLMTestSuite(your_llm_function)

# 1. Syntax Validity
syntax = suite.code_syntax_validity(code, language="python")
# Returns: {"syntax_valid": True/False, "error": ...}

# 2. Execution Test
execution = suite.code_execution_test(
    code,
    test_cases=[
        {"input": "5\n", "expected_output": "5"}
    ],
    language="python"
)
# Returns: {"pass_rate": 1.0, "passed_tests": 1, "total_tests": 1, ...}

# 3. Quality Metrics
quality = suite.code_quality_metrics(code, language="python")
# Returns: {"quality_score": 80, "metrics": {...}}

# 4. Security Scan
security = suite.code_security_scan(code, language="python")
# Returns: {"is_secure": True, "vulnerabilities": [...]}

# 5. Semantic Correctness
semantic = suite.code_semantic_correctness(
    prompt="Write add function",
    code_response=generated_code,
    reference_code=reference_solution
)
# Returns: {"semantic_similarity": 0.85, "semantically_correct": True}
```

### Quality Scoring (0-100)
Each criterion worth 20 points:
- **Has Comments** (`#`, `//`, `/*`) - 20 pts
- **Has Docstring** (`"""`, `/**`) - 20 pts  
- **Has Error Handling** (`try/except`, `try/catch`) - 20 pts
- **Low Complexity** (< 10 branches/loops) - 20 pts
- **Has Functions** (at least 1) - 20 pts

### Security Patterns Detected
- SQL Injection
- Command Injection  
- XSS vulnerabilities
- Buffer overflows (C/C++)
- Hardcoded secrets
- Unsafe deserialization
- Path traversal
- Language-specific antipatterns

### Supported Languages

| Language | Syntax Check | Execution | Quality | Security |
|----------|-------------|-----------|---------|----------|
| Python | ✅ AST | ✅ | ✅ | ✅ |
| JavaScript | ✅ Node | ✅ | ✅ | ✅ |
| TypeScript | ✅ tsc | ✅ | ✅ | ✅ |
| Java | ✅ javac | ✅ | ✅ | ✅ |
| C/C++ | ✅ gcc/g++ | ✅ | ✅ | ✅ |
| Go | ✅ go fmt | ✅ | ✅ | ✅ |
| Rust | ✅ rustc | ⚠️ | ✅ | ✅ |
| Ruby | ✅ ruby -c | ✅ | ✅ | ✅ |
| PHP | ✅ php -l | ✅ | ✅ | ✅ |

**Note**: Compilers/interpreters must be installed for full syntax validation. Falls back to regex-based checks if unavailable.

## Dual Embedder Architecture

LLMTestSuite uses specialized embedders for optimal evaluation:

### Text Embedder: `all-MiniLM-L6-v2`
- **Used for**: HSI, CSS, SRI, SVE, KBC (text metrics)
- **Size**: 22M params, 384 dimensions
- **Speed**: Fast
- **Purpose**: General semantic similarity

### Code Embedder: `BAAI/bge-small-en-v1.5`  
- **Used for**: Code semantic correctness (SCC)
- **Size**: 33M params, 384 dimensions
- **Speed**: Fast
- **Purpose**: Code-specific semantic understanding

### Custom Embedder

```python
from sentence_transformers import SentenceTransformer

suite = LLMTestSuite(my_llm)

# Replace code embedder
suite.code_embedder = SentenceTransformer("microsoft/codebert-base")
suite.code_evaluator.embedder = suite.code_embedder

# Or use different text embedder
suite = LLMTestSuite(my_llm, embedder_model="all-mpnet-base-v2")
```

### Embedder Comparison

| Model | Params | Dims | Speed | Best For |
|-------|--------|------|-------|----------|
| all-MiniLM-L6-v2 | 22M | 384 | Fast | Text (default) |
| all-mpnet-base-v2 | 110M | 768 | Medium | Text (higher accuracy) |
| bge-small-en-v1.5 | 33M | 384 | Fast | Code (default) |
| bge-base-en-v1.5 | 109M | 768 | Medium | Code (balanced) |
| CodeBERT | 125M | 768 | Medium | Code (Microsoft) |

## Output Format

All test methods support three return types via `return_type` parameter:
- `"dict"` - Returns Python dictionary (default)
- `"table"` - Prints formatted table using `rich` library
- `"both"` - Returns dictionary AND prints table

### Example Results

```python
# HSI Result
{
    "prompt": "What is the capital of France?",
    "answer": "Paris is the capital of France",
    "HSI": 0.01,  # Lower is better (0-1 scale)
    "closest_fact": "Paris is the capital of France"
}

# Code Evaluation Result
{
    "overall_score": 85.0,
    "syntax_valid": True,
    "quality_score": 80,
    "is_secure": True,
    "pass_rate": 1.0,
    "semantic_similarity": 0.89
}
```

## Complete Example: Groq API

```python
from groq import Groq
from llm_testing_suite import LLMTestSuite

client = Groq(api_key="your-api-key")

def groq_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

suite = LLMTestSuite(groq_llm)

# Text evaluation
result = suite.run_all_novel_metrics(
    prompt="What is the capital of France?",
    paraphrases=["France's capital?"],
    runs=3
)

# Code evaluation
code_result = suite.comprehensive_code_evaluation(
    prompt="Write fibonacci function",
    code_response=groq_llm("Write a Python fibonacci function"),
    language="python"
)
```

See `examples/groq_code_evaluation.py` for comprehensive test suite.

## Logging

```python
# Enable debug logging
suite = LLMTestSuite(my_llm, debug=True)

# Or configure manually
import logging
logging.getLogger("llm_testing_suite").setLevel(logging.DEBUG)
```



## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Sentence-Transformers for embedding models
- FAISS for efficient similarity search
- Rich library for beautiful terminal output
- Open-source LLM community

---

**Star this repo** ⭐ if you find it useful!

For questions or issues, please open a GitHub issue.
