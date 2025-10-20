"""
Comprehensive Code Evaluation Test Suite using Groq API
Tests all code evaluation metrics across multiple programming languages
"""

from groq import Groq
from llm_testing_suite import LLMTestSuite
import json

# 🔐 Insert your Groq API key here
GROQ_API_KEY = "your-groq-api-key-here"  # Replace with your actual API key

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def groq_llm(prompt: str, max_tokens=512, temperature=0.3) -> str:
    """
    Use Groq API for code generation (lower temperature for deterministic code).
    """
    messages = [
        {"role": "system", "content": "You are an expert programmer. Generate clean, well-documented, production-quality code."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
    )

    return response.choices[0].message.content.strip()


def extract_code_block(response: str, language: str = "python") -> str:
    """Extract code from markdown code blocks if present."""
    # Try to find code block
    import re
    pattern = rf"```{language}\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try generic code block
    pattern = r"```\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Return as-is if no code block found
    return response.strip()


print("=" * 80)
print("COMPREHENSIVE CODE EVALUATION TEST SUITE - GROQ API")
print("=" * 80)

# Initialize test suite
suite = LLMTestSuite(groq_llm)
print(f"✓ Text embedder: all-MiniLM-L6-v2")
print(f"✓ Code embedder: BAAI/bge-small-en-v1.5")
print(f"✓ Using FAISS: {suite.use_faiss}")

# ============================================================================
# TEST 1: Python - Sum Two Numbers
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Python - Sum Two Numbers Function")
print("=" * 80)

prompt = """Write a Python function called 'add_numbers' that:
1. Takes two integers as input
2. Returns their sum
3. Includes a docstring
4. Includes error handling for non-numeric inputs
5. Has a main block that reads from stdin and prints the result

The program should read two space-separated integers from stdin and output their sum."""

print(f"Prompt: {prompt[:100]}...")
print("Generating code with Groq...")

code_response = groq_llm(prompt, max_tokens=400)
generated_code = extract_code_block(code_response, "python")

print(f"\n--- Generated Code ---")
print(generated_code)

# Define test cases
test_cases = [
    {"input": "5 3\n", "expected_output": "8"},
    {"input": "10 -5\n", "expected_output": "5"},
    {"input": "0 0\n", "expected_output": "0"},
]

# Reference solution
reference_code = """
import sys

def add_numbers(a, b):
    '''Add two numbers and return the result.'''
    return a + b

if __name__ == '__main__':
    nums = sys.stdin.read().split()
    result = add_numbers(int(nums[0]), int(nums[1]))
    print(result)
"""

# Run comprehensive evaluation
print("\n--- Running Comprehensive Evaluation ---")
result = suite.comprehensive_code_evaluation(
    prompt=prompt,
    code_response=generated_code,
    reference_code=reference_code,
    test_cases=test_cases,
    language="python",
    save_json=False,
    return_type="dict"
)

print(f"\n📊 RESULTS:")
print(f"  Overall Score: {result['overall_score']:.2f}/100")
print(f"  Syntax Valid: {result['syntax_valid']}")
print(f"  Pass Rate: {result.get('pass_rate', 'N/A')}")
print(f"  Quality Score: {result['quality_score']}/100")
print(f"  Is Secure: {result['is_secure']}")
print(f"  Semantic Similarity: {result.get('semantic_similarity', 'N/A'):.4f}")

if not result['is_secure']:
    print(f"  ⚠️ Vulnerabilities: {len(result['vulnerabilities'])}")

# ============================================================================
# TEST 2: JavaScript - Factorial Function
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: JavaScript - Factorial Function")
print("=" * 80)

prompt_js = """Write a JavaScript function called 'factorial' that:
1. Calculates the factorial of a number
2. Includes JSDoc documentation
3. Includes error handling for negative numbers
4. Reads from stdin and prints the result

The program should read a number from stdin and output its factorial."""

print(f"Prompt: {prompt_js[:100]}...")
print("Generating code with Groq...")

code_response_js = groq_llm(prompt_js, max_tokens=400)
generated_code_js = extract_code_block(code_response_js, "javascript")

print(f"\n--- Generated Code ---")
print(generated_code_js[:300] + "..." if len(generated_code_js) > 300 else generated_code_js)

# Test JavaScript code
test_cases_js = [
    {"input": "5\n", "expected_output": "120"},
    {"input": "0\n", "expected_output": "1"},
    {"input": "3\n", "expected_output": "6"},
]

result_js = suite.comprehensive_code_evaluation(
    prompt=prompt_js,
    code_response=generated_code_js,
    test_cases=test_cases_js,
    language="javascript",
    save_json=False,
    return_type="dict"
)

print(f"\n📊 RESULTS:")
print(f"  Overall Score: {result_js['overall_score']:.2f}/100")
print(f"  Syntax Valid: {result_js['syntax_valid']}")
print(f"  Pass Rate: {result_js.get('pass_rate', 'N/A')}")
print(f"  Quality Score: {result_js['quality_score']}/100")
print(f"  Is Secure: {result_js['is_secure']}")

# ============================================================================
# TEST 3: Python - FizzBuzz with Quality Focus
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Python - FizzBuzz (Quality & Security Focus)")
print("=" * 80)

prompt_fb = """Write a Python function called 'fizzbuzz' that:
1. Takes an integer n as input
2. Returns 'Fizz' if divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if both, else the number
3. Includes comprehensive docstring with examples
4. Includes type hints
5. Includes input validation with try-except
6. Has unit-test-friendly design

Also write a main function that reads a number from stdin and prints the result."""

print(f"Prompt: {prompt_fb[:100]}...")
print("Generating code with Groq...")

code_response_fb = groq_llm(prompt_fb, max_tokens=500)
generated_code_fb = extract_code_block(code_response_fb, "python")

print(f"\n--- Generated Code ---")
print(generated_code_fb)

# Individual metric tests
print("\n--- Individual Metric Analysis ---")

# 1. Syntax Validity
syntax_result = suite.code_syntax_validity(generated_code_fb, language="python")
print(f"1. Syntax Valid: {syntax_result['syntax_valid']}")
if syntax_result.get('error'):
    print(f"   Error: {syntax_result['error']}")

# 2. Code Quality
quality_result = suite.code_quality_metrics(generated_code_fb, language="python")
print(f"2. Quality Score: {quality_result['quality_score']}/100")
print(f"   - Has Comments: {quality_result['metrics'].get('has_comments', False)}")
print(f"   - Has Docstring: {quality_result['metrics'].get('has_docstring', False)}")
print(f"   - Has Error Handling: {quality_result['metrics'].get('has_error_handling', False)}")
print(f"   - Cyclomatic Complexity: {quality_result['metrics'].get('cyclomatic_complexity', 0)}")
print(f"   - Number of Functions: {quality_result['metrics'].get('num_functions', 0)}")

# 3. Security Scan
security_result = suite.code_security_scan(generated_code_fb, language="python")
print(f"3. Security: {'✓ SECURE' if security_result['is_secure'] else '✗ VULNERABLE'}")
if not security_result['is_secure']:
    print(f"   Vulnerabilities found: {security_result['vulnerability_count']}")
    for vuln in security_result['vulnerabilities'][:3]:
        print(f"   - {vuln['type']} ({vuln['severity']})")

# 4. Execution Test
test_cases_fb = [
    {"input": "3\n", "expected_output": "Fizz"},
    {"input": "5\n", "expected_output": "Buzz"},
    {"input": "15\n", "expected_output": "FizzBuzz"},
    {"input": "7\n", "expected_output": "7"},
]

exec_result = suite.code_execution_test(
    generated_code_fb, 
    test_cases_fb, 
    language="python"
)
print(f"4. Execution Pass Rate: {exec_result['pass_rate']:.1%}")
print(f"   Passed: {exec_result['passed_tests']}/{exec_result['total_tests']}")

# ============================================================================
# TEST 4: Multi-Language Comparison
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Multi-Language Comparison (Hello World)")
print("=" * 80)

languages_to_test = [
    ("python", "Write a Python program that prints 'Hello, World!' to stdout."),
    ("javascript", "Write a JavaScript program that prints 'Hello, World!' to stdout using console.log."),
    ("java", "Write a Java program with a main method that prints 'Hello, World!' to stdout."),
]

comparison_results = []

for lang, prompt_lang in languages_to_test:
    print(f"\n--- {lang.upper()} ---")
    code = groq_llm(prompt_lang, max_tokens=200)
    code = extract_code_block(code, lang)
    
    syntax = suite.code_syntax_validity(code, language=lang)
    quality = suite.code_quality_metrics(code, language=lang)
    security = suite.code_security_scan(code, language=lang)
    
    comparison_results.append({
        "language": lang,
        "syntax_valid": syntax['syntax_valid'],
        "quality_score": quality['quality_score'],
        "is_secure": security['is_secure']
    })
    
    print(f"  Syntax: {syntax['syntax_valid']} | Quality: {quality['quality_score']}/100 | Secure: {security['is_secure']}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)

print("\nTest 1 (Python Sum):")
print(f"  Overall: {result['overall_score']:.1f}/100")
print(f"  Syntax: {'✓' if result['syntax_valid'] else '✗'} | "
      f"Quality: {result['quality_score']}/100 | "
      f"Security: {'✓' if result['is_secure'] else '✗'} | "
      f"Tests: {result.get('pass_rate', 0):.0%}")

print("\nTest 2 (JavaScript Factorial):")
print(f"  Overall: {result_js['overall_score']:.1f}/100")
print(f"  Syntax: {'✓' if result_js['syntax_valid'] else '✗'} | "
      f"Quality: {result_js['quality_score']}/100 | "
      f"Security: {'✓' if result_js['is_secure'] else '✗'} | "
      f"Tests: {result_js.get('pass_rate', 0):.0%}")

print("\nTest 3 (Python FizzBuzz):")
print(f"  Syntax: {'✓' if syntax_result['syntax_valid'] else '✗'} | "
      f"Quality: {quality_result['quality_score']}/100 | "
      f"Security: {'✓' if security_result['is_secure'] else '✗'} | "
      f"Tests: {exec_result['pass_rate']:.0%}")

print("\nTest 4 (Multi-Language):")
for res in comparison_results:
    print(f"  {res['language'].upper()}: "
          f"Syntax {'✓' if res['syntax_valid'] else '✗'} | "
          f"Quality {res['quality_score']}/100 | "
          f"Secure {'✓' if res['is_secure'] else '✗'}")

# Save full results to JSON
print("\n" + "=" * 80)
print("Saving results to 'groq_code_evaluation_results.json'...")

full_results = {
    "test_1_python_sum": result,
    "test_2_javascript_factorial": result_js,
    "test_3_python_fizzbuzz": {
        "syntax": syntax_result,
        "quality": quality_result,
        "security": security_result,
        "execution": exec_result
    },
    "test_4_multilang_comparison": comparison_results
}

with open("groq_code_evaluation_results.json", "w") as f:
    json.dump(full_results, f, indent=2, default=str)

print("✓ Results saved!")
print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
