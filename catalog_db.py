"""
Catalog Database
 
A small local SQLite database (an outdoor-gear product catalog, fitting
the project's existing hiking/trail-themed tasks) that the agent queries
through a constrained, parameterized interface, not raw SQL.
 
- the model never sees or writes SQL
- it can only call query_database with a small, typed set of arguments
- those arguments are turned into parameterized queries below, every
  value bound via `?`, never string-interpolated
"""

import sqlite3

DB_PATH = "catalog.db"

_SEED_PRODUCTS = [
    # (name, category, price_usd, in_stock_count, rating)
    ("Trailhead 40L Backpack", "backpack", 129.99, 34, 4.5),
    ("Summit Pro 65L Backpack", "backpack", 219.99, 12, 4.7),
    ("Daywalker 20L Backpack", "backpack", 59.99, 58, 4.2),
    ("Ridgeline Hiking Boots", "footwear", 149.99, 22, 4.6),
    ("Trailrunner Mesh Shoes", "footwear", 89.99, 40, 4.3),
    ("AlpinePeak 2-Person Tent", "tent", 249.99, 9, 4.8),
    ("Basecamp 4-Person Tent", "tent", 349.99, 5, 4.4),
    ("Ultralight Solo Tent", "tent", 179.99, 15, 4.5),
    ("CloudNine 20F Sleeping Bag", "sleeping_bag", 139.99, 27, 4.6),
    ("Frostguard 0F Sleeping Bag", "sleeping_bag", 219.99, 11, 4.7),
    ("Carbon Trekking Poles (pair)", "trekking_poles", 79.99, 45, 4.4),
    ("Aluminum Trekking Poles (pair)", "trekking_poles", 39.99, 63, 4.1),
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
    print("search 'tent' under $300:", search_products(search_term="tent", max_price=300))
    print("product 1:", get_product_by_id(1))
    print("product 999:", get_product_by_id(999))
