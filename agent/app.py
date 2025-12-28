from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
from main import process_query, simple_openai_call, get_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="API for RAG (Retrieval Augmented Generation) agent with document search capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    use_retrieval: Optional[bool] = None
    model: Optional[str] = "gpt-5-mini"
    temperature: Optional[float] = 0.7


class QueryResponse(BaseModel):
    answer: str
    query: str
    used_retrieval: Optional[bool]


class SimpleQueryRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt-5-mini"
    temperature: Optional[float] = 0.7


class SimpleQueryResponse(BaseModel):
    response: str
    prompt: str


class HealthResponse(BaseModel):
    status: str
    message: str


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="RAG Agent API is running"
    )


# Main RAG query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a query using the RAG agent.
    The agent will automatically decide if retrieval is needed, or you can force it.
    
    Args:
        request: QueryRequest containing:
            - query: The user's question/query
            - use_retrieval: Optional flag to force retrieval (default: None, agent decides)
            - model: Optional model name (default: "gpt-5-mini")
            - temperature: Optional temperature (default: 0.7)
    
    Returns:
        QueryResponse with the answer and metadata
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Process the query
        answer = process_query(
            query=request.query,
            use_retrieval=request.use_retrieval
        )
        
        return QueryResponse(
            answer=answer,
            query=request.query,
            used_retrieval=request.use_retrieval
        )
        
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Simple OpenAI call endpoint (without RAG)
@app.post("/simple-query", response_model=SimpleQueryResponse)
async def simple_query_endpoint(request: SimpleQueryRequest):
    """
    Simple OpenAI call without RAG/tool usage.
    
    Args:
        request: SimpleQueryRequest containing:
            - prompt: The user's prompt
            - model: Optional model name (default: "gpt-5-mini")
            - temperature: Optional temperature (default: 0.7)
    
    Returns:
        SimpleQueryResponse with the response
    """
    try:
        logger.info(f"Processing simple query: {request.prompt[:100]}...")
        
        response = simple_openai_call(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature
        )
        
        return SimpleQueryResponse(
            response=response,
            prompt=request.prompt
        )
        
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing simple query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "simple_query": "/simple-query (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

