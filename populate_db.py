import random
import datetime
import pandas as pd
import sqlite3

products: list[str] = ["computer", "gpu", "router", "laptop", "Smartphone", "Headphones", "Monitor"]
status_options: list[str] = ["delivered", "pending", "cancelled"]
customer_names: list[str] = ["John", "Jane", "Jill", "Joe", "Jennifer", "Josh", "Jason"]


def generate_random_dates(start_date: datetime.date,
                          end_date: datetime.date,
                          n: int) -> list[datetime.date]:
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")

    delta = end_date - start_date
    random_dates = []
    for _ in range(n):
        random_days = random.randint(0, delta.days)
        random_date = start_date + datetime.timedelta(days=random_days)
        random_dates.append(random_date)
    return random_dates


def generate_random_df(start_date: datetime.date,
                       end_date: datetime.date,
                       n: int) -> pd.DataFrame:
    random_dates: list[datetime.date] = generate_random_dates(start_date, end_date, n)

    ids: list[int] = [id for id in range(n)]
    order_nums: list[int] = [int(f"2025{order:05d}") for order in range(n)]
    items: list[str] = [random.choice(products) for _ in range(n)]
    statuses: list[str] = [random.choice(status_options) for _ in range(n)]
    customers: list[str] = [random.choice(customer_names) for _ in range(n)]

    df = pd.DataFrame(
        {
            "id": ids,
            "order_num": order_nums,
            "item": items,
            "status": statuses,
            "customer_name": customers,
            "order_date": random_dates
        }
    )
    return df


if __name__ == "__main__":
    start_date = datetime.date(2025, 1, 1)
    end_date = datetime.date(2025, 12, 31)
    df: pd.DataFrame = generate_random_df(start_date, end_date, 1_000_000)

    conn = sqlite3.connect("orders.db")
    df.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

    print("Database populated successfully!")
