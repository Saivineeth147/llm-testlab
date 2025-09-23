#Sample Example Using Hugging Face
from huggingface_hub import InferenceClient
from llm_testing_suite import LLMTestSuite
HF_TOKEN = "" # replace with your token

# Initialize the client (token only, model is passed in method)
client = InferenceClient(
    token=HF_TOKEN,
)

def hf_llm(prompt: str) -> str:
    """
    Use Hugging Face Inference API to get full text completion (non-streaming).
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        messages=messages,
        max_tokens=150,
        temperature=0.7,
        top_p=0.95,
    )

    # Extract text from response
    text = response.choices[0].message["content"]
    return text.strip()


# Example with your test suite
suite = LLMTestSuite(hf_llm)
print("Using FAISS:", suite.use_faiss)
suite.add_knowledge("Rome is the capital of Italy")
suite.list_knowledge()
result = suite.run_tests(
    prompt="Rome is the capital of Italy?",
    runs=3,
    return_type="both",
    save_json=True
)

print(result)
