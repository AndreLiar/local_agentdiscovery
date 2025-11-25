"""
Tools for the Local Discovery Agent
"""

import os
import requests
import logging
from typing import List, Dict, Any, Optional
from langchain.tools import tool

from app.config import settings

logger = logging.getLogger(__name__)

@tool
def search_places(query: str) -> str:
    """
    Search for places using SerpAPI Google Local search.
    
    Args:
        query: Complete search term like "coffee shops in Paris" or "sushi restaurants in Tokyo"
        
    Returns:
        JSON string with place results including name, rating, address, coordinates, etc.
    """
    try:
        logger.info(f"Searching places with query: {query}")
        
        if not settings.serp_api_key:
            return "Error: SERP_API_KEY not configured. Please set it in environment variables."
        
        # SerpAPI Google Local Search
        search_url = "https://serpapi.com/search"
        params = {
            "api_key": settings.serp_api_key,
            "engine": "google_maps",
            "q": query,
            "type": "search",
            "data": "!4m2!2m1!6e5"  # Local results filter
        }
        
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        local_results = data.get("local_results", [])
        
        if not local_results:
            return f"No places found for query: {query}. Try a different search term or location."
        
        # Extract and format place data
        places = []
        for result in local_results[:10]:  # Limit to top 10 results
            place = {
                "name": result.get("title", "Unknown"),
                "rating": result.get("rating"),
                "address": result.get("address", ""),
                "coordinates": None,
                "distance_km": None,
                "description": result.get("description") or result.get("snippet"),
                "price": result.get("price"),
                "reviews": result.get("reviews"),
                "hours": result.get("hours"),
                "type": result.get("type") or result.get("gps_coordinates", {}).get("google_place_type"),
                "phone": result.get("phone"),
                "website": result.get("website")
            }
            
            # Extract coordinates if available
            if "gps_coordinates" in result:
                coords = result["gps_coordinates"]
                place["coordinates"] = [coords.get("longitude"), coords.get("latitude")]
            
            # Extract distance if available
            if "distance" in result:
                distance_str = result["distance"]
                try:
                    # Parse distance like "0.5 mi" or "1.2 km"
                    if "km" in distance_str:
                        place["distance_km"] = float(distance_str.split()[0])
                    elif "mi" in distance_str:
                        # Convert miles to km
                        miles = float(distance_str.split()[0])
                        place["distance_km"] = miles * 1.60934
                except (ValueError, IndexError):
                    pass
            
            places.append(place)
        
        import json
        result_json = json.dumps({"places": places}, indent=2)
        logger.info(f"Found {len(places)} places for query: {query}")
        
        return result_json
        
    except requests.RequestException as e:
        logger.error(f"SerpAPI request failed: {e}")
        return f"Error searching places: Network request failed. Please check your connection and API key."
    except Exception as e:
        logger.error(f"Unexpected error in search_places: {e}")
        return f"Error searching places: {str(e)}"

@tool  
def get_coordinates(location: str) -> str:
    """
    Get latitude and longitude coordinates for a location using Mapbox Geocoding.
    
    Args:
        location: Location name like "Paris", "New York", "London", "Tokyo"
        
    Returns:
        JSON string with coordinates and location details
    """
    try:
        logger.info(f"Getting coordinates for location: {location}")
        
        if not settings.mapbox_access_token:
            # Fallback to a simple geocoding service or return approximate coords
            logger.warning("MAPBOX_ACCESS_TOKEN not configured, using fallback")
            
            # Simple fallback coordinates for major cities
            city_coords = {
                "paris": [2.3522, 48.8566],
                "london": [-0.1276, 51.5074], 
                "new york": [-74.0060, 40.7128],
                "tokyo": [139.6503, 35.6762],
                "rome": [12.4964, 41.9028],
                "barcelona": [2.1734, 41.3851],
                "madrid": [-3.7038, 40.4168],
                "berlin": [13.4050, 52.5200],
                "amsterdam": [4.9041, 52.3676]
            }
            
            location_lower = location.lower()
            for city, coords in city_coords.items():
                if city in location_lower:
                    result = {
                        "location": location,
                        "coordinates": coords,
                        "longitude": coords[0],
                        "latitude": coords[1],
                        "source": "fallback"
                    }
                    return f"Coordinates for {location}: {coords[1]:.4f}, {coords[0]:.4f} (approximate)"
            
            return f"Coordinates not available for {location}. Please use a major city name."
        
        # Mapbox Geocoding API
        geocoding_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{location}.json"
        params = {
            "access_token": settings.mapbox_access_token,
            "limit": 1
        }
        
        response = requests.get(geocoding_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        features = data.get("features", [])
        
        if not features:
            return f"Location not found: {location}. Try a different location name."
        
        feature = features[0]
        coordinates = feature["geometry"]["coordinates"]  # [longitude, latitude]
        place_name = feature["place_name"]
        
        result = {
            "location": location,
            "place_name": place_name,
            "coordinates": coordinates,
            "longitude": coordinates[0],
            "latitude": coordinates[1],
            "source": "mapbox"
        }
        
        logger.info(f"Found coordinates for {location}: {coordinates[1]:.4f}, {coordinates[0]:.4f}")
        
        import json
        return json.dumps(result, indent=2)
        
    except requests.RequestException as e:
        logger.error(f"Mapbox geocoding request failed: {e}")
        return f"Error getting coordinates: Network request failed. Please check your connection."
    except Exception as e:
        logger.error(f"Unexpected error in get_coordinates: {e}")
        return f"Error getting coordinates: {str(e)}"

# List of available tools
AGENT_TOOLS = [search_places, get_coordinates]