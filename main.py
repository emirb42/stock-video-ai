import requests
import os
import random
import logging
from typing import List, Dict
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

nltk.download('averaged_perceptron_tagger')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
required_resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']

for resource in required_resources:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        logger.error(f"Error downloading NLTK resource '{resource}': {str(e)}")
        logger.info(f"Attempting to download '{resource}' without quiet mode...")
        try:
            nltk.download(resource, quiet=False)
        except Exception as e:
            logger.error(f"Failed to download '{resource}'. Please download it manually using nltk.download('{resource}').")
            raise

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("PEXELS_API_KEY")

BASE_URL = "https://api.pexels.com/videos/"
HEADERS = {
    "Authorization": API_KEY
}

def extract_keywords(text: str) -> List[str]:
    """
    Extract relevant keywords from the given text using NLTK.
    Process each sentence separately and combine the results.
    """
    sentences = sent_tokenize(text)
    all_keywords = []

    for sentence in sentences:
        # Tokenize the sentence
        tokens = word_tokenize(sentence.lower())

        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

        # Perform part-of-speech tagging
        pos_tags = pos_tag(tokens)

        # Extract nouns and verbs as keywords
        keywords = [word for word, pos in pos_tags if pos.startswith('NN') or pos.startswith('VB')]

        all_keywords.extend(keywords)

    # Remove duplicates and return
    return list(set(all_keywords))

def classify_text(text: str) -> List[str]:
    """
    Classify the text and return relevant categories.
    """
    # For simplicity, we'll use the extracted keywords as categories
    # In a more advanced implementation, you might use a pre-trained classifier
    return extract_keywords(text)[:5]  # Limit to top 5 categories

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

def fetch_and_sort_videos(script: str, sort_by: str = "relevance", max_results: int = 50) -> List[Dict]:
    """
    Extract keywords from the script, fetch and sort videos based on the extracted keywords and sorting criteria.
    """
    logger.info(f"Processing script and fetching videos. Sort by: {sort_by}, max_results: {max_results}")

    # Split the script into sentences
    sentences = sent_tokenize(script)
    logger.info(f"Split script into {len(sentences)} sentences")

    all_keywords = []
    all_videos = []

    fetch_func = mock_fetch_videos if not API_KEY or API_KEY.strip() == "" else fetch_videos
    logger.info(f"Using {'mock' if fetch_func == mock_fetch_videos else 'real'} fetch function")

    for sentence in sentences:
        keywords = extract_keywords(sentence)
        all_keywords.extend(keywords)
        logger.info(f"Extracted keywords from sentence: {keywords}")

        for keyword in keywords:
            page = 1
            while len(all_videos) < max_results:
                response = fetch_func(keyword, per_page=min(max_results - len(all_videos), 15), page=page)
                videos = response.get("videos", [])

                if not videos:
                    logger.info(f"No more videos found for keyword '{keyword}' after fetching {len(all_videos)} videos")
                    break

                all_videos.extend(videos)
                page += 1

            if len(all_videos) >= max_results:
                break

        if len(all_videos) >= max_results:
            break

    # Remove duplicate keywords
    all_keywords = list(set(all_keywords))

    sorted_videos = sort_videos(all_videos, sort_by, " ".join(all_keywords))
    logger.info(f"Fetched and sorted {len(sorted_videos)} videos")
    return sorted_videos

if __name__ == "__main__":
    example_script = "Dance is more than just movement; it is a form of social interaction. From communal dance gatherings to professional performances, dance brings people together, forming bonds and building communities."

    logger.info("Processing example script:")
    logger.info(example_script)

    sorted_videos = fetch_and_sort_videos(example_script, sort_by="relevance", max_results=15)
    logger.info(f"Fetched and sorted {len(sorted_videos)} videos for the entire script")

    for i, video in enumerate(sorted_videos[:5], 1):  # Display top 5 videos
        logger.info(f"Top {i}:")
        logger.info(f"ID: {video['id']}, Title: {video['title']}, Duration: {video['duration']}s, "
                    f"Width: {video['width']}px, Height: {video['height']}px")
        logger.debug(f"Description: {video['description']}")

    logger.info("---")

    # Demonstrate processing individual sentences
    sentences = sent_tokenize(example_script)
    for i, sentence in enumerate(sentences, 1):
        logger.info(f"\nProcessing sentence {i}:")
        logger.info(sentence)
        keywords = extract_keywords(sentence)
        logger.info(f"Extracted keywords: {keywords}")

        sorted_videos = fetch_and_sort_videos(sentence, sort_by="relevance", max_results=3)
        logger.info(f"Fetched and sorted {len(sorted_videos)} videos for this sentence")

        for j, video in enumerate(sorted_videos, 1):
            logger.info(f"Video {j}:")
            logger.info(f"ID: {video['id']}, Title: {video['title']}, Duration: {video['duration']}s, "
                        f"Width: {video['width']}px, Height: {video['height']}px")
            logger.debug(f"Description: {video['description']}")
        logger.info("---")