from fastapi import APIRouter, HTTPException, status
from app.services.sql_service import sql_service
from pydantic import BaseModel
from typing import List, Dict, Any
# Authentication removed
# from app.core.security import oauth2_scheme

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]

@router.post("/nl", response_model=QueryResponse)
async def execute_nl_query(
    request: QueryRequest
) -> QueryResponse:
    """
    Execute a natural language query and return the results
    """
    results = await sql_service.execute_nl_query(request.query)

    # Check if there was an error
    if results and len(results) == 1 and "error" in results[0]:
        error_message = results[0]["error"]
        if "SQL service not available" in error_message:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SQL service is not available"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message
            )

    return QueryResponse(results=results)

@router.post("/sql", response_model=QueryResponse)
async def execute_sql_query(
    request: QueryRequest
) -> QueryResponse:
    """
    Execute a raw SQL query and return the results
    """
    results = await sql_service.execute_sql_query(request.query)

    # Check if there was an error
    if results and len(results) == 1 and "error" in results[0]:
        error_message = results[0]["error"]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message
        )

    return QueryResponse(results=results)