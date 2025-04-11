from fastapi import APIRouter, Depends, HTTPException, status
from app.services.sql_service import sql_service
from pydantic import BaseModel
from typing import List, Dict, Any
from app.core.security import oauth2_scheme

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]

@router.post("/nl", response_model=QueryResponse)
async def execute_nl_query(
    request: QueryRequest,
    token: str = Depends(oauth2_scheme)
) -> QueryResponse:
    """
    Execute a natural language query and return the results
    """
    try:
        results = await sql_service.execute_nl_query(request.query)
        return QueryResponse(results=results)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/sql", response_model=QueryResponse)
async def execute_sql_query(
    request: QueryRequest,
    token: str = Depends(oauth2_scheme)
) -> QueryResponse:
    """
    Execute a raw SQL query and return the results
    """
    try:
        results = await sql_service.execute_sql_query(request.query)
        return QueryResponse(results=results)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) 