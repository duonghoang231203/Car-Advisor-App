"""
API endpoints for monitoring.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from app.core.monitoring import get_metrics, reset_metrics

router = APIRouter(
    tags=["monitoring"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

@router.get("/metrics",
         summary="Get API metrics",
         description="Get metrics about API usage, performance, and cache efficiency",
         response_model=dict)
async def metrics():
    """
    Get metrics about API usage, performance, and cache efficiency.

    Returns:
        dict: A dictionary containing metrics about API usage, performance, and cache efficiency
    """
    return get_metrics()

@router.post("/reset",
          summary="Reset metrics",
          description="Reset all metrics to zero",
          status_code=status.HTTP_204_NO_CONTENT)
async def reset():
    """
    Reset all metrics to zero.

    Returns:
        None
    """
    reset_metrics()
    return None
