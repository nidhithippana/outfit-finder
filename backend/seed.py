from database import init_db, SessionLocal, Product

SEED_PRODUCTS = [
    # --- Shirts ---
    {"name": "Classic White Oxford Shirt", "category": "shirt", "color": "white", "price": 34.99, "brand": "Everlane", "image_url": "", "product_url": ""},
    {"name": "White Linen Button-Down", "category": "shirt", "color": "white", "price": 44.99, "brand": "Uniqlo", "image_url": "", "product_url": ""},
    {"name": "White Oversized Tee", "category": "shirt", "color": "white", "price": 22.99, "brand": "H&M", "image_url": "", "product_url": ""},
    {"name": "White Crop Top", "category": "shirt", "color": "white", "price": 18.99, "brand": "Zara", "image_url": "", "product_url": ""},
    {"name": "White Ribbed Tank Top", "category": "shirt", "color": "white", "price": 14.99, "brand": "ASOS", "image_url": "", "product_url": ""},

    {"name": "Black Slim Fit Tee", "category": "shirt", "color": "black", "price": 19.99, "brand": "Gap", "image_url": "", "product_url": ""},
    {"name": "Black Graphic Tee", "category": "shirt", "color": "black", "price": 25.99, "brand": "Urban Outfitters", "image_url": "", "product_url": ""},
    {"name": "Black Silk Blouse", "category": "shirt", "color": "black", "price": 59.99, "brand": "Mango", "image_url": "", "product_url": ""},
    {"name": "Black Cropped Hoodie", "category": "shirt", "color": "black", "price": 39.99, "brand": "Nike", "image_url": "", "product_url": ""},
    {"name": "Black Striped Long Sleeve", "category": "shirt", "color": "black", "price": 29.99, "brand": "Banana Republic", "image_url": "", "product_url": ""},

    {"name": "Blue Chambray Shirt", "category": "shirt", "color": "blue", "price": 36.99, "brand": "Everlane", "image_url": "", "product_url": ""},
    {"name": "Blue Denim Shirt", "category": "shirt", "color": "blue", "price": 42.99, "brand": "Levi's", "image_url": "", "product_url": ""},
    {"name": "Blue Striped Tee", "category": "shirt", "color": "blue", "price": 24.99, "brand": "Tommy Hilfiger", "image_url": "", "product_url": ""},

    {"name": "Beige Linen Shirt", "category": "shirt", "color": "beige", "price": 38.99, "brand": "COS", "image_url": "", "product_url": ""},
    {"name": "Beige Knit Sweater", "category": "shirt", "color": "beige", "price": 54.99, "brand": "Everlane", "image_url": "", "product_url": ""},

    {"name": "Pink Floral Blouse", "category": "shirt", "color": "pink", "price": 32.99, "brand": "Zara", "image_url": "", "product_url": ""},
    {"name": "Pink Oversized Sweatshirt", "category": "shirt", "color": "pink", "price": 45.99, "brand": "ASOS", "image_url": "", "product_url": ""},

    {"name": "Gray Crew Neck Sweatshirt", "category": "shirt", "color": "gray", "price": 39.99, "brand": "Champion", "image_url": "", "product_url": ""},
    {"name": "Gray Wool Turtleneck", "category": "shirt", "color": "gray", "price": 64.99, "brand": "COS", "image_url": "", "product_url": ""},

    # --- Pants ---
    {"name": "Blue Slim Fit Jeans", "category": "pants", "color": "blue", "price": 49.99, "brand": "Levi's", "image_url": "", "product_url": ""},
    {"name": "Blue Straight Leg Jeans", "category": "pants", "color": "blue", "price": 54.99, "brand": "AG Jeans", "image_url": "", "product_url": ""},
    {"name": "Blue Wide Leg Jeans", "category": "pants", "color": "blue", "price": 59.99, "brand": "Madewell", "image_url": "", "product_url": ""},
    {"name": "Blue Cargo Pants", "category": "pants", "color": "blue", "price": 64.99, "brand": "Dickies", "image_url": "", "product_url": ""},
    {"name": "Blue Jogger Jeans", "category": "pants", "color": "blue", "price": 44.99, "brand": "H&M", "image_url": "", "product_url": ""},

    {"name": "Black High-Waist Trousers", "category": "pants", "color": "black", "price": 69.99, "brand": "Zara", "image_url": "", "product_url": ""},
    {"name": "Black Skinny Jeans", "category": "pants", "color": "black", "price": 54.99, "brand": "Levi's", "image_url": "", "product_url": ""},
    {"name": "Black Cargo Pants", "category": "pants", "color": "black", "price": 59.99, "brand": "Nike", "image_url": "", "product_url": ""},
    {"name": "Black Wide Leg Trousers", "category": "pants", "color": "black", "price": 74.99, "brand": "COS", "image_url": "", "product_url": ""},
    {"name": "Black Joggers", "category": "pants", "color": "black", "price": 39.99, "brand": "Adidas", "image_url": "", "product_url": ""},

    {"name": "White Linen Pants", "category": "pants", "color": "white", "price": 49.99, "brand": "Mango", "image_url": "", "product_url": ""},
    {"name": "White Wide Leg Trousers", "category": "pants", "color": "white", "price": 64.99, "brand": "Banana Republic", "image_url": "", "product_url": ""},

    {"name": "Beige Chinos", "category": "pants", "color": "beige", "price": 54.99, "brand": "J.Crew", "image_url": "", "product_url": ""},
    {"name": "Beige Linen Trousers", "category": "pants", "color": "beige", "price": 59.99, "brand": "COS", "image_url": "", "product_url": ""},

    {"name": "Gray Sweatpants", "category": "pants", "color": "gray", "price": 44.99, "brand": "Champion", "image_url": "", "product_url": ""},
    {"name": "Gray Slim Chinos", "category": "pants", "color": "gray", "price": 54.99, "brand": "Gap", "image_url": "", "product_url": ""},

    # --- Shoes ---
    {"name": "White Low-Top Sneakers", "category": "shoes", "color": "white", "price": 79.99, "brand": "Adidas", "image_url": "", "product_url": ""},
    {"name": "White Leather Sneakers", "category": "shoes", "color": "white", "price": 89.99, "brand": "Common Projects", "image_url": "", "product_url": ""},
    {"name": "White Running Shoes", "category": "shoes", "color": "white", "price": 99.99, "brand": "Nike", "image_url": "", "product_url": ""},
    {"name": "White Platform Sneakers", "category": "shoes", "color": "white", "price": 74.99, "brand": "Steve Madden", "image_url": "", "product_url": ""},

    {"name": "Black Chelsea Boots", "category": "shoes", "color": "black", "price": 119.99, "brand": "Aldo", "image_url": "", "product_url": ""},
    {"name": "Black Ankle Boots", "category": "shoes", "color": "black", "price": 99.99, "brand": "Steve Madden", "image_url": "", "product_url": ""},
    {"name": "Black Leather Sneakers", "category": "shoes", "color": "black", "price": 89.99, "brand": "Vans", "image_url": "", "product_url": ""},
    {"name": "Black Loafers", "category": "shoes", "color": "black", "price": 94.99, "brand": "Sam Edelman", "image_url": "", "product_url": ""},
    {"name": "Black High-Top Sneakers", "category": "shoes", "color": "black", "price": 84.99, "brand": "Converse", "image_url": "", "product_url": ""},

    {"name": "Brown Leather Boots", "category": "shoes", "color": "brown", "price": 129.99, "brand": "Timberland", "image_url": "", "product_url": ""},
    {"name": "Brown Suede Loafers", "category": "shoes", "color": "brown", "price": 109.99, "brand": "Clarks", "image_url": "", "product_url": ""},
    {"name": "Brown Sandals", "category": "shoes", "color": "brown", "price": 59.99, "brand": "Birkenstock", "image_url": "", "product_url": ""},

    {"name": "Beige Slip-On Sandals", "category": "shoes", "color": "beige", "price": 49.99, "brand": "H&M", "image_url": "", "product_url": ""},
    {"name": "Beige Platform Mules", "category": "shoes", "color": "beige", "price": 69.99, "brand": "Zara", "image_url": "", "product_url": ""},

    # --- Dresses ---
    {"name": "Black Mini Dress", "category": "dress", "color": "black", "price": 54.99, "brand": "Zara", "image_url": "", "product_url": ""},
    {"name": "Black Wrap Dress", "category": "dress", "color": "black", "price": 64.99, "brand": "Mango", "image_url": "", "product_url": ""},
    {"name": "Black Slip Dress", "category": "dress", "color": "black", "price": 49.99, "brand": "ASOS", "image_url": "", "product_url": ""},

    {"name": "White Sundress", "category": "dress", "color": "white", "price": 44.99, "brand": "H&M", "image_url": "", "product_url": ""},
    {"name": "White Linen Midi Dress", "category": "dress", "color": "white", "price": 69.99, "brand": "Everlane", "image_url": "", "product_url": ""},

    {"name": "Blue Floral Midi Dress", "category": "dress", "color": "blue", "price": 59.99, "brand": "Free People", "image_url": "", "product_url": ""},
    {"name": "Blue Denim Dress", "category": "dress", "color": "blue", "price": 54.99, "brand": "Levi's", "image_url": "", "product_url": ""},

    {"name": "Pink Satin Slip Dress", "category": "dress", "color": "pink", "price": 74.99, "brand": "Revolve", "image_url": "", "product_url": ""},
    {"name": "Pink Floral Wrap Dress", "category": "dress", "color": "pink", "price": 64.99, "brand": "Anthropologie", "image_url": "", "product_url": ""},

    # --- Skirts ---
    {"name": "Black Mini Skirt", "category": "skirt", "color": "black", "price": 34.99, "brand": "Zara", "image_url": "", "product_url": ""},
    {"name": "Black Pleated Midi Skirt", "category": "skirt", "color": "black", "price": 44.99, "brand": "Mango", "image_url": "", "product_url": ""},
    {"name": "Black Satin Maxi Skirt", "category": "skirt", "color": "black", "price": 54.99, "brand": "ASOS", "image_url": "", "product_url": ""},

    {"name": "White Linen Midi Skirt", "category": "skirt", "color": "white", "price": 39.99, "brand": "COS", "image_url": "", "product_url": ""},
    {"name": "White Mini Tennis Skirt", "category": "skirt", "color": "white", "price": 29.99, "brand": "Urban Outfitters", "image_url": "", "product_url": ""},

    {"name": "Blue Denim Mini Skirt", "category": "skirt", "color": "blue", "price": 39.99, "brand": "Levi's", "image_url": "", "product_url": ""},
    {"name": "Blue Pleated Midi Skirt", "category": "skirt", "color": "blue", "price": 49.99, "brand": "Free People", "image_url": "", "product_url": ""},

    {"name": "Beige Linen Maxi Skirt", "category": "skirt", "color": "beige", "price": 54.99, "brand": "Mango", "image_url": "", "product_url": ""},
    {"name": "Pink Satin Mini Skirt", "category": "skirt", "color": "pink", "price": 44.99, "brand": "Revolve", "image_url": "", "product_url": ""},
]


def seed():
    init_db()
    db = SessionLocal()

    existing = db.query(Product).count()
    if existing > 0:
        print(f"Database already has {existing} products. Skipping seed.")
        db.close()
        return

    for item in SEED_PRODUCTS:
        db.add(Product(**item))

    db.commit()
    db.close()
    print(f"Seeded {len(SEED_PRODUCTS)} products.")


if __name__ == "__main__":
    seed()
