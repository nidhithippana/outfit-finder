"""
Scrapes Shein product pages to collect training data.
Saves product images to data/images/ and metadata to data/products.csv.

Usage:
    python scrape_shein.py
    python scrape_shein.py --per-category 200   # more data
"""

import asyncio
import csv
import argparse
import hashlib
import httpx
from pathlib import Path
from playwright.async_api import async_playwright

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "products.csv"

# Shein category pages to scrape
CATEGORY_URLS = {
    "top":     "https://us.shein.com/Women-Tops-cat-1738.html",
    "blouse":  "https://us.shein.com/Women-Blouses-Shirts-cat-1786.html",
    "jeans":   "https://us.shein.com/Women-Jeans-cat-1740.html",
    "dress":   "https://us.shein.com/Women-Dresses-cat-1727.html",
    "skirt":   "https://us.shein.com/Women-Skirts-cat-1732.html",
    "jacket":  "https://us.shein.com/Women-Jackets-Coats-cat-1739.html",
    "shorts":  "https://us.shein.com/Women-Shorts-cat-1741.html",
    "sweater": "https://us.shein.com/Women-Sweaters-cat-1742.html",
}

CSV_FIELDS = [
    "image_path", "category", "title",
    "color", "neckline", "sleeve_length",
    "clothing_length", "fit_type", "pattern_type",
    "waist_line", "style", "product_url",
]


async def get_product_links(page, category_url, limit):
    """Scroll a Shein category page and collect product URLs."""
    print(f"  Loading category page: {category_url}")
    await page.goto(category_url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(3000)

    links = set()
    scroll_attempts = 0

    while len(links) < limit and scroll_attempts < 15:
        # Collect all product links currently visible
        anchors = await page.query_selector_all("a[href*='/p-']")
        for a in anchors:
            href = await a.get_attribute("href")
            if href and "/p-" in href:
                if href.startswith("/"):
                    href = "https://us.shein.com" + href
                links.add(href.split("?")[0])  # strip query params

        if len(links) >= limit:
            break

        await page.evaluate("window.scrollBy(0, 1200)")
        await page.wait_for_timeout(1500)
        scroll_attempts += 1

    print(f"  Found {len(links)} product links")
    return list(links)[:limit]


async def scrape_product(page, url, category):
    """Visit a product page and extract image + all attributes."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        # Product title
        title = ""
        title_el = await page.query_selector(".product-intro__head-name, h1.goods-name")
        if title_el:
            title = (await title_el.text_content() or "").strip()

        # Main product image — find highest quality image src
        image_url = ""
        img_el = await page.query_selector(
            ".crop-image-container__img, .product-intro__image img, .goods-detail__images img"
        )
        if img_el:
            image_url = await img_el.get_attribute("src") or await img_el.get_attribute("data-src") or ""
            # Shein serves resized images — request a larger version
            if image_url and "_thumbnail_" in image_url:
                image_url = image_url.replace("_thumbnail_", "_squick_")

        if not image_url:
            return None

        # Product attributes — Shein renders a table of label: value pairs
        attrs = {}
        attr_rows = await page.query_selector_all(
            ".product-intro__attr-item, .goods-attr__item, .product-attr__item"
        )
        for row in attr_rows:
            label_el = await row.query_selector(
                ".product-intro__attr-label, .goods-attr__label, label"
            )
            value_el = await row.query_selector(
                ".product-intro__attr-val, .goods-attr__val, span:last-child"
            )
            if label_el and value_el:
                k = (await label_el.text_content() or "").strip().rstrip(":").lower()
                v = (await value_el.text_content() or "").strip()
                if k and v:
                    attrs[k] = v

        return {
            "product_url": url,
            "category": category,
            "title": title,
            "image_url": image_url,
            "color":           attrs.get("color", ""),
            "neckline":        attrs.get("neckline", ""),
            "sleeve_length":   attrs.get("sleeve length", ""),
            "clothing_length": attrs.get("clothing length", attrs.get("length", "")),
            "fit_type":        attrs.get("fit type", attrs.get("type", "")),
            "pattern_type":    attrs.get("pattern type", ""),
            "waist_line":      attrs.get("waist line", ""),
            "style":           attrs.get("style", ""),
        }

    except Exception as e:
        print(f"    Error scraping {url}: {e}")
        return None


async def download_image(client, image_url, product_url):
    """Download a product image and return the local path."""
    try:
        uid = hashlib.md5(product_url.encode()).hexdigest()[:12]
        path = IMAGES_DIR / f"{uid}.jpg"

        if path.exists():
            return str(path)

        headers = {"Referer": "https://us.shein.com/", "User-Agent": "Mozilla/5.0"}
        response = await client.get(image_url, headers=headers, timeout=15)
        if response.status_code == 200:
            path.write_bytes(response.content)
            return str(path)
    except Exception as e:
        print(f"    Image download failed: {e}")
    return None


async def scrape_category(playwright, category, url, per_category, writer):
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 800},
    )
    page = await context.new_page()

    product_links = await get_product_links(page, url, per_category)
    saved = 0

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for i, link in enumerate(product_links):
            print(f"  [{i+1}/{len(product_links)}] {link[:70]}")
            product = await scrape_product(page, link, category)

            if not product or not product["image_url"]:
                continue

            image_path = await download_image(client, product["image_url"], link)
            if not image_path:
                continue

            writer.writerow({
                "image_path":      image_path,
                "category":        product["category"],
                "title":           product["title"],
                "color":           product["color"],
                "neckline":        product["neckline"],
                "sleeve_length":   product["sleeve_length"],
                "clothing_length": product["clothing_length"],
                "fit_type":        product["fit_type"],
                "pattern_type":    product["pattern_type"],
                "waist_line":      product["waist_line"],
                "style":           product["style"],
                "product_url":     product["product_url"],
            })
            saved += 1

            # Polite delay between product pages
            await page.wait_for_timeout(800)

    await browser.close()
    print(f"  Saved {saved} products for '{category}'")
    return saved


async def main(per_category):
    DATA_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    existing = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            reader = csv.DictReader(f)
            existing = {row["product_url"] for row in reader}
        print(f"Resuming — {len(existing)} products already collected")

    mode = "a" if existing else "w"
    with open(CSV_PATH, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()

        async with async_playwright() as playwright:
            for category, url in CATEGORY_URLS.items():
                print(f"\nScraping category: {category}")
                await scrape_category(playwright, category, url, per_category, writer)

    total = sum(1 for _ in open(CSV_PATH)) - 1
    print(f"\nDone. Total products: {total}")
    print(f"Images: {IMAGES_DIR}")
    print(f"CSV:    {CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=100,
                        help="How many products to scrape per category (default: 100)")
    args = parser.parse_args()
    asyncio.run(main(args.per_category))
