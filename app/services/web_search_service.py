from typing import List, Dict, Any
import duckduckgo_search
from app.core.logging import logger

class WebSearchService:
    def __init__(self):
        """Initialize the web search service"""
        self.ddgs = duckduckgo_search.DDGS()

    async def search_car_info(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for car information using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, link, and snippet
        """
        try:
            # Enhance the query to focus on car information
            enhanced_query = f"latest {query} car information news reviews"
            
            # Perform the search
            results = []
            for r in self.ddgs.text(enhanced_query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("link", ""),
                    "snippet": r.get("body", "")
                })
            
            logger.info(f"Found {len(results)} web search results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return []

    async def search_car_reviews(self, brand: str, model: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search for specific car reviews
        
        Args:
            brand: Car brand
            model: Car model
            max_results: Maximum number of results to return
            
        Returns:
            List of review results
        """
        try:
            query = f"{brand} {model} car review latest"
            results = []
            
            for r in self.ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("link", ""),
                    "snippet": r.get("body", "")
                })
                
            logger.info(f"Found {len(results)} review results for {brand} {model}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching car reviews: {e}")
            return []

    async def search_car_comparison(self, car1: str, car2: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search for car comparison information
        
        Args:
            car1: First car (brand model)
            car2: Second car (brand model)
            max_results: Maximum number of results to return
            
        Returns:
            List of comparison results
        """
        try:
            query = f"{car1} vs {car2} comparison review"
            results = []
            
            for r in self.ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("link", ""),
                    "snippet": r.get("body", "")
                })
                
            logger.info(f"Found {len(results)} comparison results for {car1} vs {car2}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching car comparison: {e}")
            return []

# Create a singleton instance
web_search_service = WebSearchService() 