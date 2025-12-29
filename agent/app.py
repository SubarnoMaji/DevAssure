from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
from main import process_query, get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Agent API",
    description="API for RAG (Retrieval Augmented Generation) agent with document search capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    use_retrieval: Optional[bool] = None
    model: Optional[str] = "gpt-4o-mini"
    temperature: Optional[float] = 0.7


class QueryResponse(BaseModel):
    answer: str
    query: str
    used_retrieval: Optional[bool]


class SimpleQueryRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt-4o-mini"
    temperature: Optional[float] = 0.7


class SimpleQueryResponse(BaseModel):
    response: str
    prompt: str


class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="RAG Agent API is running"
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query[:100]}...")

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


    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing simple query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
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
