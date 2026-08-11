"""
Catalog Database

A small local SQLite database -- a general multi-category retail catalog,
deliberately spanning unrelated domains (electronics, books, kitchenware,
office supplies, toys, beauty) rather than one theme -- that the agent
queries through a constrained, parameterized interface, not raw SQL.

- the model never sees or writes SQL
- it can only call query_database with a small, typed set of arguments
- those arguments are turned into parameterized queries below, every
  value bound via `?`, never string-interpolated
"""

import sqlite3

DB_PATH = "catalog.db"

_SEED_PRODUCTS = [
    # (name, category, price_usd, in_stock_count, rating)
    ("Noise-Cancelling Wireless Headphones", "electronics", 89.99, 41, 4.5),
    ("27-inch 4K Monitor", "electronics", 329.99, 8, 4.6),
    ("Portable Bluetooth Speaker", "electronics", 44.99, 67, 4.2),
    ("Mechanical Keyboard", "electronics", 74.99, 23, 4.4),
    ("A Brief History of Time", "books", 14.99, 52, 4.7),
    ("The Midnight Library", "books", 12.99, 38, 4.3),
    ("Atomic Habits", "books", 16.99, 90, 4.8),
    ("Cast Iron Skillet, 12-inch", "kitchenware", 34.99, 44, 4.6),
    ("6-Piece Chef Knife Set", "kitchenware", 59.99, 19, 4.4),
    ("Stovetop Espresso Maker", "kitchenware", 27.99, 33, 4.3),
    ("Mechanical Pencil Set (12-pack)", "office_supplies", 9.99, 120, 4.1),
    ("Ergonomic Office Chair", "office_supplies", 189.99, 6, 4.5),
    ("Desk Organizer Tray", "office_supplies", 19.99, 71, 4.0),
    ("Wooden Building Blocks Set", "toys", 29.99, 26, 4.6),
    ("Remote Control Car", "toys", 39.99, 18, 4.2),
    ("SPF 50 Facial Sunscreen", "beauty", 15.99, 88, 4.5),
    ("Vitamin C Serum", "beauty", 22.99, 54, 4.4),
]


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    """creates the products table and seeds it, if not already present"""
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price_usd REAL NOT NULL,
            in_stock_count INTEGER NOT NULL,
            rating REAL NOT NULL
        )
        """
    )
    count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    if count == 0:
        conn.executemany(
            "INSERT INTO products (name, category, price_usd, in_stock_count, rating) VALUES (?, ?, ?, ?, ?)",
            _SEED_PRODUCTS,
        )
        conn.commit()
    conn.close()


def search_products(
    search_term: str = None,
    category: str = None,
    max_price: float = None,
    in_stock_only: bool = False,
) -> list[dict]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    clauses = []
    params = []
    if search_term:
        clauses.append("name LIKE ?")
        params.append(f"%{search_term}%")
    if category:
        clauses.append("category = ?")
        params.append(category)
    if max_price is not None:
        clauses.append("price_usd <= ?")
        params.append(max_price)
    if in_stock_only:
        clauses.append("in_stock_count > 0")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM products {where} ORDER BY rating DESC LIMIT 10", params
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_product_by_id(product_id: int) -> dict:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_categories() -> list[str]:
    conn = _connect()
    rows = conn.execute("SELECT DISTINCT category FROM products ORDER BY category").fetchall()
    conn.close()
    return [r[0] for r in rows]


if __name__ == "__main__":
    init_db()
    print("categories:", list_categories())
    print("search 'headphones' under $100:", search_products(search_term="headphones", max_price=100))
    print("product 1:", get_product_by_id(1))
    print("product 999:", get_product_by_id(999))
