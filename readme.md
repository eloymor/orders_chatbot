# Orders Chatbot

This project is a chatbot application built with Streamlit, LangGraph, and LangChain. It allows users to inquire about order 
information from a SQLite database. The chatbot can understand different user intents, such as checking an order's 
status, listing all pending orders, or retrieving orders from the current day.

The easiest would be to let the LLM (Large Language Model) to write the SQL queries itself. However, this would not be a safe
approach as easily one could cheat the app to create Data Manipulation Language (DML) queries and 
modify or delete information. In addition, SQL injection attacks would be straightforward to perform.

In this project, the LLM classifies the input question and then call custom tools which will perform
the database retrieval using SQLAlchemy.

Please note that this is an example application, and most of the tools to retrieve data from the database
are limited to 20 results. For example, as the data is randomly generated, filtering by the pending orders
could retrieve more than 3_000 orders, increasing the context prompt and its api cost.

The LLM used is Google gemini-2.5-flash-lite, as Gemini provides some free API requests per day. This model is also
capable of performing mathematical operations correctly (in case you ask for average lead time by costumer, for example).

Feel free to modify the database or the LLM. A typically Ollama local LLM would perform great for this application,
as the LLM only is used for its understanding of the language.

## Features

- Chat with an AI to get order information.
- Check the status of a specific order by providing an order number.
- Get a list of all pending orders.
- Get a list of all orders placed today.
- Get a list of the lead time for delivered orders.
- You can ask for a summary or a detailed report of any of the previous queries.
- Simple and intuitive web interface powered by Streamlit.

## How it Works

The application uses a Streamlit frontend (`app.py`) to provide a user-friendly chat interface. The backend logic is powered by a LangGraph agent (`main.py`) which processes user queries. The agent can classify the user's intent and use different tools to fetch data from the `orders.db` SQLite database. The database itself is populated with dummy data using `populate_db.py`.

## Setup and Installation

Follow these steps to get the project running locally.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/eloymor/orders_chatbot.git
    cd <orders_chatbot>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your Google API key. This is required for the chatbot's language model.
    ```
    GOOGLE_API_KEY="your_google_api_key"
    ```

5.  **Populate the database:**
    Run the following command to create the `orders.db` file and populate it with 1,000,000 sample orders.
    ```bash
    python populate_db.py
    ```

## Usage


1. **To run the Streamlit application, execute the following command in your terminal:**
    ```bash
   streamlit run app.py
    ```

2. Then, open your web browser to the local URL provided by Streamlit (default: http://localhost:8501/). You can now start chatting with the bot about your orders.
3. **Optionally**, you can run this app in the CLI (command line interface) without any UI or frontend:
    ```bash
   python main.py
    ```

## Project Structure

-   `app.py`: The main Streamlit application file that creates the user interface.
-   `main.py`: Contains the core chatbot logic using LangChain and LangGraph to process user input and interact with the database.
-   `populate_db.py`: A script to generate sample data and populate the `orders.db` SQLite database.
-   `requirements.txt`: A list of all the Python packages required to run the project.
-   `.env`: Configuration file for environment variables (you need to create this).
-   `orders.db`: The SQLite database file (created after running `populate_db.py`).
