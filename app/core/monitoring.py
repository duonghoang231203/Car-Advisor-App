"""
Monitoring module for tracking API performance and usage.
"""
import time
import functools
import statistics
from typing import Dict, List, Callable, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger

# Store metrics in memory
_metrics = {
    "requests": {
        "total": 0,
        "success": 0,
        "error": 0,
        "by_endpoint": {}
    },
    "response_times": {
        "all": [],
        "by_endpoint": {}
    },
    "cache": {
        "hits": 0,
        "misses": 0
    }
}

def get_metrics() -> Dict:
    """Get current metrics"""
    # Calculate average response times
    avg_response_time = 0
    if _metrics["response_times"]["all"]:
        avg_response_time = statistics.mean(_metrics["response_times"]["all"])
    
    # Calculate average response times by endpoint
    avg_by_endpoint = {}
    for endpoint, times in _metrics["response_times"]["by_endpoint"].items():
        if times:
            avg_by_endpoint[endpoint] = statistics.mean(times)
    
    # Calculate cache hit rate
    cache_hit_rate = 0
    total_cache_requests = _metrics["cache"]["hits"] + _metrics["cache"]["misses"]
    if total_cache_requests > 0:
        cache_hit_rate = _metrics["cache"]["hits"] / total_cache_requests
    
    return {
        "requests": {
            "total": _metrics["requests"]["total"],
            "success": _metrics["requests"]["success"],
            "error": _metrics["requests"]["error"],
            "by_endpoint": _metrics["requests"]["by_endpoint"]
        },
        "response_times": {
            "average": avg_response_time,
            "by_endpoint": avg_by_endpoint
        },
        "cache": {
            "hits": _metrics["cache"]["hits"],
            "misses": _metrics["cache"]["misses"],
            "hit_rate": cache_hit_rate
        }
    }

def reset_metrics() -> None:
    """Reset all metrics"""
    global _metrics
    _metrics = {
        "requests": {
            "total": 0,
            "success": 0,
            "error": 0,
            "by_endpoint": {}
        },
        "response_times": {
            "all": [],
            "by_endpoint": {}
        },
        "cache": {
            "hits": 0,
            "misses": 0
        }
    }

def record_cache_hit() -> None:
    """Record a cache hit"""
    _metrics["cache"]["hits"] += 1

def record_cache_miss() -> None:
    """Record a cache miss"""
    _metrics["cache"]["misses"] += 1

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring API requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Record request
        _metrics["requests"]["total"] += 1
        
        # Get endpoint
        endpoint = request.url.path
        if endpoint not in _metrics["requests"]["by_endpoint"]:
            _metrics["requests"]["by_endpoint"][endpoint] = 0
        _metrics["requests"]["by_endpoint"][endpoint] += 1
        
        # Initialize response time tracking for endpoint
        if endpoint not in _metrics["response_times"]["by_endpoint"]:
            _metrics["response_times"]["by_endpoint"][endpoint] = []
        
        # Measure response time
        start_time = time.time()
        try:
            response = await call_next(request)
            # Record successful request
            _metrics["requests"]["success"] += 1
        except Exception as e:
            # Record error request
            _metrics["requests"]["error"] += 1
            logger.error(f"Error processing request: {str(e)}")
            raise
        finally:
            # Record response time
            response_time = time.time() - start_time
            _metrics["response_times"]["all"].append(response_time)
            _metrics["response_times"]["by_endpoint"][endpoint].append(response_time)
            
            # Limit stored response times to prevent memory issues
            if len(_metrics["response_times"]["all"]) > 1000:
                _metrics["response_times"]["all"] = _metrics["response_times"]["all"][-1000:]
            if len(_metrics["response_times"]["by_endpoint"][endpoint]) > 1000:
                _metrics["response_times"]["by_endpoint"][endpoint] = _metrics["response_times"]["by_endpoint"][endpoint][-1000:]
        
        return response

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            func_name = func.__name__
            logger.debug(f"Function {func_name} executed in {execution_time:.4f} seconds")
    return wrapper
