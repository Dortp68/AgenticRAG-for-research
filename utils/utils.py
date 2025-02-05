from duckduckgo_search import DDGS
from typing import Optional, List


def web_search_text(query: str, max_results: Optional[int] = 5) -> List:
    """
    Search for text on duckduckgo.com.

    Args:
        query (str): The text to search for.
        max_results (Optional[int]): The maximum number of search results to retrieve (default 10).

    Returns:
        List of search results as strings.
    """

    with DDGS() as ddgs:
        results = results = [r['href'] for r in ddgs.text(query, max_results=max_results)]
    return results

