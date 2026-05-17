import httpx
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

SEARCHAPI_KEY = os.getenv("SEARCHAPI_KEY")
SEARCHAPI_URL = "https://www.searchapi.io/api/v1/search"


def search_shein_products(query: str, limit: int = 5) -> list[dict]:
    params = {
        "engine": "google_shopping",
        "q": f"{query} shein",
        "api_key": SEARCHAPI_KEY,
        "num": limit,
    }

    try:
        response = httpx.get(SEARCHAPI_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"SearchAPI error for '{query}': {e}")
        return []

    results = []
    for item in data.get("shopping_results", [])[:limit]:
        seller = item.get("seller", "")
        if "shein" not in seller.lower() and "shein" not in item.get("title", "").lower():
            continue

        shein_search_url = f"https://www.shein.com/search?q={quote_plus(query)}"

        results.append({
            "name": item.get("title", ""),
            "price": item.get("price", ""),
            "original_price": item.get("original_price"),
            "tag": item.get("tag"),
            "image_url": item.get("thumbnail", ""),
            "product_url": shein_search_url,
        })

    return results
