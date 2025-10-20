"""
Code evaluation for all three Groq models
Tests Python and JavaScript code generation
Uses LLM Testing Suite with CCE formula from paper
"""

from groq import Groq
import json
import re
import sys
import os
from datetime import datetime

# Add parent directory to path to import llm_testing_suite
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_testing_suite import LLMTestSuite

# API Key
GROQ_API_KEY = "your-groq-api-key-here"

# Models to test
MODELS = {
    "GPT-OSS-20B": "openai/gpt-oss-20b",
    "Llama-3.3-70B": "llama-3.3-70b-versatile",
    "Qwen3-32B": "qwen/qwen3-32b"
}

client = Groq(api_key=GROQ_API_KEY)

def create_llm_function(model_name):
    """Create LLM function for specific model with Qwen-safe parameters"""
    def llm_func(prompt: str, max_tokens=600, temperature=0.3) -> str:
        # For Qwen models, use non-thinking mode parameters
        is_qwen = "qwen" in model_name.lower()

        # Enhanced system message to prevent <think> tags
        system_content = """You are an expert programmer. Generate ONLY clean, 
executable code without any explanations, reasoning, or commentary. 
Do not use <think> tags or any surrounding text. Return pure code only."""

        # Enhanced prompt to emphasize code-only output
        enhanced_prompt = f"{prompt}\n\nIMPORTANT: Return ONLY the code, no explanations or reasoning."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": enhanced_prompt}
        ]

        try:
            # Parameters based on Qwen best practices
            api_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            if is_qwen:
                # Non-thinking mode for Qwen (general dialogue settings)
                # Note: Groq API doesn't support top_k or min_p
                api_params.update({
                    "temperature": 0.7,
                    "top_p": 0.8,
                })
            else:
                # Standard parameters for other models
                api_params.update({
                    "temperature": temperature,
                    "top_p": 0.8,
                })

            response = client.chat.completions.create(**api_params)
            raw_response = response.choices[0].message.content.strip()

            # Debug: Print raw response for debugging
            if is_qwen:
                print(f"   [DEBUG] Qwen raw response ({len(raw_response)} chars):")
                print(f"   {raw_response[:500]}")

            return raw_response
        except Exception as e:
            print(f"   Error: {e}")
            return ""
    return llm_func

def extract_code_block(response: str, language: str = "python") -> str:
    """Extract code from markdown code blocks and remove reasoning text"""
    original_response = response

    # FIRST: Aggressively remove <think> tags and their content
    # Use multiple patterns to catch variations
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
    response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)

    # Remove any remaining think tags (unclosed or malformed)
    response = re.sub(r'</?think.*?>', '', response, flags=re.IGNORECASE)
    response = re.sub(r'</?thinking.*?>', '', response, flags=re.IGNORECASE)

    # Try to extract from markdown code blocks with language
    pattern = rf"```{language}\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to extract from generic markdown code blocks
    pattern = r"```\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try just ``` with any language
    pattern = r"```\w*\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Special handling: Extract only valid Python code lines
    lines = response.split('\n')
    code_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines at the start
        if not code_lines and not stripped:
            continue

        # Skip obvious reasoning/explanation lines
        lower_line = line.lower()
        if any(marker in lower_line for marker in [
            'okay,', 'let me', 'i need to', 'first,', 'to solve', 'the user',
            'then,', 'but need', 'make sure', 'wait,', 'hmm,', 'so,',
            'starting with', 'in the main', 'oh right', 'like if'
        ]):
            continue

        # Check if line looks like valid Python code
        is_code_like = (
            stripped.startswith(('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ',
                               'try:', 'except', 'return ', 'print(', '@')) or
            stripped.startswith('if __name__') or
            (stripped and not stripped[0].isupper() and '=' in stripped) or  # assignment
            (in_code_block and (stripped.startswith((' ', '\t')) or not stripped))  # indented or blank
        )

        # Start collecting code when we find a def, class, import, or if __name__
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'if __name__')):
            in_code_block = True
            code_lines.append(line)
        elif in_code_block:
            # Continue collecting if it looks like code
            if is_code_like or not stripped:
                code_lines.append(line)
            elif stripped and (stripped[0].isupper() or ':' not in line):
                # Stop if we hit explanatory text (starts with capital, no colons)
                # But allow docstrings
                if '"""' not in line and "'''" not in line:
                    break
                else:
                    code_lines.append(line)
            else:
                code_lines.append(line)

    if code_lines:
        # Clean up trailing empty lines and explanatory text
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()

        # Remove any trailing lines that look like explanations
        while code_lines:
            last_line = code_lines[-1].strip().lower()
            if any(marker in last_line for marker in ['but need', 'make sure', 'wait', 'note:']):
                code_lines.pop()
            else:
                break

        return '\n'.join(code_lines).strip()

    # Last resort: return cleaned response
    response = re.sub(r'^(Here\'s|Here is|This is).*?:\s*', '', response, flags=re.IGNORECASE)
    return response.strip()

print("="*80)
print("CODE EVALUATION FOR THREE GROQ MODELS")
print("Using LLM Testing Suite with CCE Formula from Paper")
print("="*80)
print(f"Models: {', '.join(MODELS.keys())}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Code generation prompt
PYTHON_PROMPT = """Write a Python program that:
1. Defines a function called 'add_numbers' that takes two integers and returns their sum
2. Includes a docstring for the function
3. Has a main block that reads two space-separated integers from stdin and prints the result
4. The output should be just the number, nothing else

Example: if input is "2 3", output should be "5" """

PYTHON_TEST_CASES = [
    {"input": "2 3\n", "expected_output": "5"},
    {"input": "-1 4\n", "expected_output": "3"},
    {"input": "0 0\n", "expected_output": "0"},
    {"input": "10 -5\n", "expected_output": "5"}
]

# Reference implementation for semantic comparison
REFERENCE_CODE = '''def add_numbers(a, b):
    """Add two numbers and return the result."""
    return a + b

if __name__ == "__main__":
    numbers = input().split()
    a, b = int(numbers[0]), int(numbers[1])
    print(add_numbers(a, b))
'''

all_results = {}

for model_display_name, model_id in MODELS.items():
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_display_name} ({model_id})")
    print(f"{'='*80}")

    # Create LLM function for this model
    llm_func = create_llm_function(model_id)

    # Initialize testing suite
    suite = LLMTestSuite(llm_func=llm_func)

    print("\nGenerating Python code...")
    code_response = llm_func(PYTHON_PROMPT, max_tokens=500, temperature=0.3)
    generated_code = extract_code_block(code_response, "python")

    print(f"Generated {len(generated_code)} characters of code")
    print("\nCode preview:")
    print("-" * 40)
    print(generated_code[:300] + ("..." if len(generated_code) > 300 else ""))
    print("-" * 40)

    # Run comprehensive evaluation using the suite
    print("\nRunning comprehensive code evaluation...")
    result = suite.comprehensive_code_evaluation(
        prompt=PYTHON_PROMPT,
        code_response=generated_code,
        reference_code=REFERENCE_CODE,
        test_cases=PYTHON_TEST_CASES,
        language="python",
        return_type="dict"
    )

    # Display results
    print(f"\n--- EVALUATION RESULTS ---")
    print(f"Syntax Valid (SV): {result['syntax_valid']} ({1.0 if result['syntax_valid'] else 0.0})")
    print(f"Execution Pass Rate (EPR): {result.get('pass_rate', 1.0):.3f}")
    print(f"Code Quality Score (CQS): {result['quality_score']}/100 ({result['quality_score']/100:.3f})")
    print(f"Security Risk Score (SRS): {result.get('srs', 0.0):.3f}")
    print(f"Security Score (1-SRS): {1 - result.get('srs', 0.0):.3f}")
    print(f"Semantic Correctness (SCC): {result.get('semantic_similarity', 1.0):.3f}")
    print(f"\n--- CCE BREAKDOWN ---")
    breakdown = result.get('metrics_breakdown', {})
    print(f"α*SV  (0.15 × {breakdown.get('sv', 0):.3f}) = {0.15 * breakdown.get('sv', 0):.4f}")
    print(f"β*EPR (0.35 × {breakdown.get('epr', 0):.3f}) = {0.35 * breakdown.get('epr', 0):.4f}")
    print(f"γ*CQS (0.20 × {breakdown.get('cqs', 0):.3f}) = {0.20 * breakdown.get('cqs', 0):.4f}")
    print(f"δ*(1-SRS) (0.15 × {breakdown.get('security_score', 0):.3f}) = {0.15 * breakdown.get('security_score', 0):.4f}")
    print(f"ϵ*SCC (0.15 × {breakdown.get('scc', 0):.3f}) = {0.15 * breakdown.get('scc', 0):.4f}")
    print(f"\nCCE Score: {result['cce_score']:.4f}")
    print(f"Overall Score: {result['overall_score']:.2f}/100")

    all_results[model_display_name] = {
        "model_name": model_display_name,
        "model_id": model_id,
        "python": {
            "syntax_valid": result['syntax_valid'],
            "pass_rate": result.get('pass_rate', 1.0),
            "quality_score": result['quality_score'],
            "srs": result.get('srs', 0.0),
            "security_score": 1 - result.get('srs', 0.0),
            "semantic_similarity": result.get('semantic_similarity', 1.0),
            "cce_score": result['cce_score'],
            "overall_score": result['overall_score'],
            "metrics_breakdown": result.get('metrics_breakdown', {}),
            "generated_code": generated_code
        }
    }

# Save results
output_file = "code_evaluation_results.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*80}")
print(f"CODE EVALUATION COMPLETE")
print(f"Results saved to {output_file}")
print(f"{'='*80}")

# Summary table
print("\n\nSUMMARY TABLE - CCE SCORES:")
print("-" * 120)
print(f"{'Model':<20} {'SV':<8} {'EPR':<8} {'CQS':<8} {'1-SRS':<8} {'SCC':<8} {'CCE':<10} {'Overall':<12}")
print("-" * 120)

for model_name, results in all_results.items():
    py = results["python"]
    breakdown = py["metrics_breakdown"]
    print(f"{model_name:<20} "
          f"{breakdown.get('sv', 0):.3f}    "
          f"{breakdown.get('epr', 0):.3f}    "
          f"{breakdown.get('cqs', 0):.3f}    "
          f"{breakdown.get('security_score', 0):.3f}    "
          f"{breakdown.get('scc', 0):.3f}    "
          f"{py['cce_score']:.4f}     "
          f"{py['overall_score']:.2f}/100")

print("-" * 120)

# Formula verification
print("\n\nFORMULA VERIFICATION:")
print("CCE = α*SV + β*EPR + γ*CQS + δ*(1-SRS) + ϵ*SCC")
print("Weights: α=0.15, β=0.35, γ=0.20, δ=0.15, ϵ=0.15")
print("All metrics normalized to [0, 1]")
print("="*80)
