import random
import datetime
import pandas as pd
import sqlite3

# Options to randomly choose for each type
products: list[str] = ["computer", "gpu", "router", "laptop", "Smartphone", "Headphones", "Monitor"]
status_options: list[str] = ["delivered", "pending", "cancelled"]
customer_names: list[str] = ["John", "Jane", "Jill", "Joe", "Jennifer", "Josh", "Jason"]


def generate_random_dates(start_date: datetime.date,
                          end_date: datetime.date,
                          n: int) -> list[datetime.date]:
    """
    Generates a list of random dates within a specified date range.

    This function takes a start date, an end date, and a number of random dates to
    generate. It ensures that the start date is earlier than or equal to the end
    date and then randomly selects dates within the range, up to the specified
    number of dates.

    :param start_date: The start of the date range for generating random dates.
    :param end_date: The end of the date range for generating random dates.
    :param n: The number of random dates to generate.
    :return: A list of randomly generated dates within the specified range.
    :raises ValueError: If start_date is after end_date.
    :raises ValueError: If `start_date` is equal to `end_date`.
    """
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    elif start_date == end_date:
        raise ValueError("start_date and end_date cannot be the same")

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
    """
    Generates a pandas DataFrame containing randomly generated order data. The DataFrame includes fields such
    as order number, item, quantity, price, order date, shipping date, status, customer name, and total amount.
    Randomized data ensures significantly diverse outputs within the provided constraints.

    :param start_date: Start of the date range from which random order dates will be generated.
                       Must be a valid date earlier than or equal to `end_date`.
    :param end_date: End of the date range from which random order dates will be generated.
                     Must be a valid date later than or equal to `start_date`.
    :param n: Number of rows to generate in the resulting DataFrame. Must be a positive integer and should
              not exceed 1,000,000.
    :return: Randomly generated pandas DataFrame containing order data with columns including ID, order number,
             item details, quantities, prices, statuses, customer names, order dates, shipping dates, and total
             amounts calculated from price and quantity.

    :raises ValueError: If `n` is less than 1 or greater than 1,000,000.
    """

    if n > 1_000_000:
        raise ValueError("n cannot be greater than 1,000,000")
    elif n < 1:
        raise ValueError("n must be at least 1")

    random_dates: list[datetime.date] = generate_random_dates(start_date, end_date, n)

    ids: list[int] = [id for id in range(n)]
    order_nums: list[int] = [int(f"2025{order:05d}") for order in range(n)] # format: 202500005
    items: list[str] = [random.choice(products) for _ in range(n)]
    quantities: list[int] = [random.randint(1, 10) for _ in range(n)]
    prices: list[float] = [round(random.uniform(10.00, 1000.00), 2) for _ in range(n)]
    statuses: list[str] = [random.choice(status_options) for _ in range(n)]
    customers: list[str] = [random.choice(customer_names) for _ in range(n)]

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "id": ids,
            "order_num": order_nums,
            "item": items,
            "quantity": quantities,
            "price": prices,
            "status": statuses,
            "customer_name": customers,
            "order_date": random_dates
        }
    )

    # Add a new total_amount column:
    df["total_amount"] = df["price"] * df["quantity"]
    df["total_amount"] = df["total_amount"].round(2)

    # Add a new shipping_date column just for the delivered orders
    df.loc[df["status"] == "delivered", "shipping_date"] = df["order_date"].apply(
        lambda x: x + pd.Timedelta(days=random.randint(1, 7)))

    return df


if __name__ == "__main__":
    """
    Function to populate the SQLite database with random data.
    """
    start_date = datetime.date(2025, 1, 1)
    end_date = datetime.date(2025, 12, 31)
    df: pd.DataFrame = generate_random_df(start_date, end_date, 1_000_000)

    conn = sqlite3.connect("orders.db")
    df.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

    print("Database populated successfully!")
