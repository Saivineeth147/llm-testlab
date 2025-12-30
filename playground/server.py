"""
LLM Playground API Server
FastAPI backend serving the LLM TestLab evaluation endpoints
"""

import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for llm_testing_suite import
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_testing_suite import LLMTestSuite
from providers import create_provider, get_provider_info, fetch_models_for_provider, LLMProvider


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────

class ProviderConfig(BaseModel):
    provider: str = Field(..., description="Provider name: 'groq' or 'huggingface'")
    api_key: str = Field(..., description="API key for the provider")
    model: Optional[str] = Field(None, description="Model name")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1024, ge=1, le=8192)


class TextEvaluationRequest(BaseModel):
    provider_config: ProviderConfig
    prompt: str = Field(..., description="Main prompt to evaluate")
    paraphrases: Optional[list[str]] = Field(None, description="List of paraphrased prompts for SRI")
    adversarial_prompts: Optional[list[str]] = Field(None, description="Adversarial prompts for SVE")
    knowledge_base: Optional[list[str]] = Field(None, description="Facts for knowledge base")
    runs: int = Field(3, ge=1, le=10, description="Number of runs for consistency testing")
    use_faiss: bool = Field(False, description="Use FAISS for similarity search")


class CodeEvaluationRequest(BaseModel):
    provider_config: ProviderConfig
    prompt: str = Field(..., description="Code generation prompt")
    code_response: Optional[str] = Field(None, description="Code to evaluate (if not generating)")
    reference_code: Optional[str] = Field(None, description="Reference solution for comparison")
    test_cases: Optional[list[dict]] = Field(None, description="Test cases with input/expected_output")
    language: str = Field("python", description="Programming language")


class TestConnectionRequest(BaseModel):
    provider: str
    api_key: str
    model: Optional[str] = None


class FetchModelsRequest(BaseModel):
    provider: str
    api_key: str


# ─────────────────────────────────────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("🚀 LLM Playground API starting...")
    yield
    print("👋 LLM Playground API shutting down...")


app = FastAPI(
    title="LLM Playground API",
    description="API for evaluating LLM models using LLM TestLab suite",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_test_suite(provider: LLMProvider, use_faiss: bool = False, knowledge_base: list[str] = None) -> LLMTestSuite:
    """Create an LLMTestSuite instance with the given provider."""
    suite = LLMTestSuite(
        llm_func=provider.generate,
        use_faiss=use_faiss,
        knowledge_base=knowledge_base
    )
    return suite


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "llm-playground"}


@app.get("/api/providers")
async def list_providers():
    """List available LLM providers (without models - use /api/models for that)."""
    return get_provider_info()


@app.post("/api/models")
async def fetch_models(request: FetchModelsRequest):
    """
    Fetch available models from the specified provider.
    Requires API key to authenticate with the provider's API.
    """
    try:
        models = fetch_models_for_provider(request.provider, request.api_key)
        return {
            "provider": request.provider,
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/test-connection")
async def test_connection(request: TestConnectionRequest):
    """Test if the API key works for the specified provider."""
    try:
        provider = create_provider(
            provider_name=request.provider,
            api_key=request.api_key,
            model=request.model
        )
        # Try a simple generation
        response = provider.generate("Say 'Hello' in one word.")
        return {
            "success": True,
            "message": "Connection successful",
            "response_preview": response[:100] if response else ""
        }
    except Exception as e:
        error_msg = str(e).lower()
        # Parse common error types for user-friendly messages
        if "invalid api key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
            detail = "Invalid API key. Please check your key and try again."
        elif "rate limit" in error_msg:
            detail = "Rate limit exceeded. Please wait a moment and try again."
        elif "model" in error_msg and ("not found" in error_msg or "does not exist" in error_msg):
            detail = f"Model not found or you don't have access to it. Try a different model."
        elif "quota" in error_msg or "exceeded" in error_msg:
            detail = "API quota exceeded. Check your plan limits."
        elif "connection" in error_msg or "timeout" in error_msg:
            detail = "Connection failed. Check your internet connection."
        else:
            detail = str(e)
        
        raise HTTPException(status_code=400, detail=detail)


@app.post("/api/evaluate/text")
async def evaluate_text(request: TextEvaluationRequest):
    """
    Run text evaluation metrics: HSI, CSS, SRI, SVE, KBC.
    """
    try:
        # Create provider
        provider = create_provider(
            provider_name=request.provider_config.provider,
            api_key=request.provider_config.api_key,
            model=request.provider_config.model,
            temperature=request.provider_config.temperature,
            max_tokens=request.provider_config.max_tokens
        )
        
        # Create test suite
        suite = create_test_suite(
            provider=provider,
            use_faiss=request.use_faiss,
            knowledge_base=request.knowledge_base
        )
        
        # Add any additional knowledge
        if request.knowledge_base:
            for fact in request.knowledge_base:
                suite.add_knowledge(fact)
        
        # Run all metrics
        results = suite.run_all_novel_metrics(
            prompt=request.prompt,
            paraphrases=request.paraphrases,
            adversarial_prompts=request.adversarial_prompts,
            runs=request.runs,
            save_json=False,
            return_type="dict"
        )
        
        # Format response
        formatted = {
            "prompt": request.prompt,
            "model": request.provider_config.model or "default",
            "provider": request.provider_config.provider,
            "metrics": {
                "HSI": {
                    "value": results["HSI"]["HSI"],
                    "label": "Hallucination Severity Index",
                    "description": "Factual deviation from knowledge base (lower is better)",
                    "closest_fact": results["HSI"].get("closest_fact", ""),
                    "answer": results["HSI"].get("answer", "")
                },
                "CSS": {
                    "value": results["CSS"]["CSS"],
                    "variance": results["CSS"]["CSS_variance"],
                    "label": "Consistency Stability Score",
                    "description": "Output stability across runs (higher is better)",
                    "outputs": results["CSS"].get("outputs", [])
                },
            }
        }
        
        if "SRI" in results:
            formatted["metrics"]["SRI"] = {
                "value": results["SRI"]["SRI"],
                "label": "Semantic Robustness Index",
                "description": "Invariance to paraphrasing (higher is better)",
                "base_output": results["SRI"].get("base_output", ""),
                "para_outputs": results["SRI"].get("para_outputs", [])
            }
        
        if "SVE" in results:
            formatted["metrics"]["SVE"] = {
                "value": results["SVE"]["SVE"],
                "label": "Safety Vulnerability Exposure",
                "description": "Unsafe response rate (lower is better)",
                "unsafe_cases": results["SVE"].get("unsafe_cases", 0),
                "total_cases": results["SVE"].get("total_cases", 0)
            }
        
        formatted["metrics"]["KBC"] = {
            "value": results["KBC"]["KBC"],
            "label": "Knowledge Base Coverage",
            "description": "Factual alignment with KB (higher is better)",
            "aligned_cases": results["KBC"].get("aligned_cases", 0),
            "total_cases": results["KBC"].get("total_cases", 0)
        }
        
        return formatted
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def extract_code_from_markdown(text: str, language: str = "python") -> str:
    """
    Extract code from markdown code blocks.
    LLMs often return code wrapped in ```python ... ``` blocks.
    """
    import re
    
    # Try to extract code from markdown code block
    # Match ```language ... ``` or ``` ... ```
    patterns = [
        rf'```{language}\s*\n(.*?)```',  # ```python\n...\n```
        rf'```{language[:2]}\s*\n(.*?)```',  # ```py\n...\n``` 
        r'```\w*\s*\n(.*?)```',  # ```\n...\n``` (any language)
        r'```(.*?)```',  # Inline code blocks
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # No code block found, return original text (might be raw code)
    return text.strip()


@app.post("/api/evaluate/code")
async def evaluate_code(request: CodeEvaluationRequest):
    """
    Run code evaluation metrics: syntax, execution, quality, security, semantic.
    """
    try:
        # Create provider
        provider = create_provider(
            provider_name=request.provider_config.provider,
            api_key=request.provider_config.api_key,
            model=request.provider_config.model,
            temperature=request.provider_config.temperature,
            max_tokens=request.provider_config.max_tokens
        )
        
        # Create test suite
        suite = create_test_suite(provider=provider)
        
        # Get code to evaluate
        if request.code_response:
            raw_code = request.code_response
        else:
            # Generate code from prompt
            raw_code = provider.generate(request.prompt)
        
        # Extract code from markdown blocks (LLMs often wrap code in ```)
        code = extract_code_from_markdown(raw_code, request.language)
        
        # Run comprehensive code evaluation
        results = suite.comprehensive_code_evaluation(
            prompt=request.prompt,
            code_response=code,
            reference_code=request.reference_code,
            test_cases=request.test_cases,
            language=request.language,
            save_json=False,
            return_type="dict"
        )
        
        # Get quality metrics details
        quality_metrics = results.get("quality_metrics", {})
        
        return {
            "prompt": request.prompt,
            "code": code,
            "raw_response": raw_code if raw_code != code else None,  # Show if extraction happened
            "language": request.language,
            "model": request.provider_config.model or "default",
            "provider": request.provider_config.provider,
            "metrics": {
                "overall_score": results.get("overall_score", 0),
                "cce_score": results.get("cce_score", 0),
                "syntax_valid": results.get("syntax_valid", False),
                "syntax_error": results.get("syntax_error"),
                "pass_rate": results.get("pass_rate"),
                "passed_tests": results.get("passed_tests"),
                "total_tests": results.get("total_tests"),
                "quality_score": results.get("quality_score", 0),
                "quality_metrics": quality_metrics,
                "is_secure": results.get("is_secure", True),
                "srs": results.get("srs", 0),
                "vulnerabilities": results.get("vulnerabilities", []),
                "semantic_similarity": results.get("semantic_similarity"),
                "metrics_breakdown": results.get("metrics_breakdown", {}),
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate_response(request: TextEvaluationRequest):
    """Simple generation endpoint without evaluation."""
    try:
        provider = create_provider(
            provider_name=request.provider_config.provider,
            api_key=request.provider_config.api_key,
            model=request.provider_config.model,
            temperature=request.provider_config.temperature,
            max_tokens=request.provider_config.max_tokens
        )
        
        response = provider.generate(request.prompt)
        
        return {
            "prompt": request.prompt,
            "response": response,
            "model": request.provider_config.model or "default",
            "provider": request.provider_config.provider
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Static File Serving (for React frontend)
# ─────────────────────────────────────────────────────────────────────────────

# Check if frontend build exists
frontend_build_path = Path(__file__).parent / "frontend" / "dist"

if frontend_build_path.exists():
    # Serve static files from the React build
    app.mount("/assets", StaticFiles(directory=frontend_build_path / "assets"), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the React frontend for all non-API routes."""
        # Return index.html for client-side routing
        index_path = frontend_build_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend not built")
else:
    @app.get("/")
    async def root():
        """Root endpoint when frontend is not built."""
        return {
            "message": "LLM Playground API",
            "docs": "/docs",
            "note": "Frontend not built. Run 'npm run build' in the frontend directory."
        }


# ─────────────────────────────────────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
