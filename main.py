import requests
import os
import random
import logging
from typing import List, Dict
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("PEXELS_API_KEY")

BASE_URL = "https://api.pexels.com/videos/"
HEADERS = {
    "Authorization": API_KEY
}

def validate_api_response(response: Dict) -> None:
    """
    Validate the structure of the API response.
    """
    if "videos" not in response:
        raise ValueError("Invalid API response: 'videos' key not found")
    if not isinstance(response["videos"], list):
        raise ValueError("Invalid API response: 'videos' is not a list")
    for video in response["videos"]:
        required_keys = ["id", "duration", "width", "height"]
        if not all(key in video for key in required_keys):
            raise ValueError(f"Invalid video object: missing required keys {required_keys}")

def fetch_videos(query: str, per_page: int = 15, page: int = 1) -> Dict:
    """
    Fetch videos from Pexels API based on the given query.
    """
    url = f"{BASE_URL}search"
    params = {
        "query": query,
        "per_page": per_page,
        "page": page
    }

    try:
        logger.info(f"Fetching videos for query: '{query}', page: {page}, per_page: {per_page}")
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        validate_api_response(data)

        # Extract additional metadata for each video
        for video in data['videos']:
            video['title'] = video.get('user', {}).get('name', '')
            video['description'] = video.get('url', '')

        logger.info(f"Successfully fetched {len(data['videos'])} videos")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching videos: {e}", exc_info=True)
        return {"videos": []}
    except ValueError as e:
        logger.error(f"Error in API response: {e}", exc_info=True)
        return {"videos": []}

def mock_fetch_videos(query: str, per_page: int = 15, page: int = 1) -> Dict:
    """
    Generate a mock API response for testing purposes.
    """
    logger.info(f"Generating mock videos for query: '{query}', page: {page}, per_page: {per_page}")
    mock_videos = [
        {
            "id": i,
            "duration": random.randint(5, 60),
            "width": random.choice([1280, 1920, 3840]),
            "height": random.choice([720, 1080, 2160]),
            "title": f"Mock Video {i} about {query}",
            "description": f"This is a mock video description for {query} content. Video number {i}."
        } for i in range(per_page)
    ]
    return {"videos": mock_videos}

def sort_videos(videos: List[Dict], sort_by: str = "relevance", query: str = "") -> List[Dict]:
    """
    Sort videos based on the specified criteria.
    """
    logger.info(f"Sorting videos by: {sort_by}")
    if sort_by == "duration":
        return sorted(videos, key=lambda x: x["duration"])
    elif sort_by == "width":
        return sorted(videos, key=lambda x: x["width"], reverse=True)
    elif sort_by == "height":
        return sorted(videos, key=lambda x: x["height"], reverse=True)
    elif sort_by == "relevance":
        return sorted(videos, key=lambda x: calculate_relevance(x, query), reverse=True)
    else:
        logger.warning(f"Unknown sort criteria: {sort_by}. Defaulting to relevance.")
        return sorted(videos, key=lambda x: calculate_relevance(x, query), reverse=True)

def calculate_relevance(video: Dict, query: str) -> float:
    """
    Calculate the relevance score of a video based on its title and description.
    """
    title_weight = 0.6
    description_weight = 0.4

    title_relevance = sum(query.lower() in word.lower() for word in video.get("title", "").split())
    description_relevance = sum(query.lower() in word.lower() for word in video.get("description", "").split())

    return (title_relevance * title_weight) + (description_relevance * description_weight)

def fetch_and_sort_videos(query: str, sort_by: str = "relevance", max_results: int = 50) -> List[Dict]:
    """
    Fetch and sort videos based on the given query and sorting criteria.
    """
    logger.info(f"Fetching and sorting videos for query: '{query}', sort_by: {sort_by}, max_results: {max_results}")
    all_videos = []
    page = 1

    fetch_func = mock_fetch_videos if not API_KEY or API_KEY.strip() == "" else fetch_videos
    logger.info(f"Using {'mock' if fetch_func == mock_fetch_videos else 'real'} fetch function")

    while len(all_videos) < max_results:
        response = fetch_func(query, per_page=min(max_results - len(all_videos), 15), page=page)
        videos = response.get("videos", [])

        if not videos:
            logger.info(f"No more videos found after fetching {len(all_videos)} videos")
            break

        all_videos.extend(videos)
        page += 1

    sorted_videos = sort_videos(all_videos, sort_by)
    logger.info(f"Fetched and sorted {len(sorted_videos)} videos")
    return sorted_videos

if __name__ == "__main__":
    example_query = "nature"
    sorted_videos = fetch_and_sort_videos(example_query, sort_by="relevance")

    logger.info(f"Fetched and sorted videos for query: '{example_query}'")
    for video in sorted_videos:
        logger.info(f"ID: {video['id']}, Title: {video['title']}, Duration: {video['duration']}s, "
                    f"Width: {video['width']}px, Height: {video['height']}px")
        logger.debug(f"Description: {video['description']}")