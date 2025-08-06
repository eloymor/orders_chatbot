import os
from typing import TypedDict, Optional
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Date, Float, func
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import trim_messages
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# load variables inside .env file to environment
load_dotenv()

# initialize the llm model
api_key: str = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=api_key)

# SQLite DB connection
engine = create_engine('sqlite:///orders.db')
# Init base class for SQL Alchemy
Base = declarative_base()


# Class which contains the table name and its schema
class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    order_num = Column(String)
    item = Column(String)
    quantity = Column(Integer)
    price = Column(Float)
    status = Column(String)
    customer_name = Column(String)
    order_date = Column(Date)
    total_amount = Column(Float)
    shipping_date = Column(Date, nullable=True)

    def __repr__(self):
        return (f"<Order(order_num={self.order_num}, item={self.item}, "
                f"status={self.status}, customer_name={self.customer_name}, "
                f"{self.order_date.strftime("%d/%m/%Y")})>, "
                f"quantity={self.quantity}, price={self.price}, total_amount={self.total_amount}, "
                f"shipping_date={self.shipping_date if self.shipping_date else 'Not yet shipped'})>")


# Create the session to the DB
Session = sessionmaker(bind=engine)


@tool
def get_order_status(order_num: str) -> str:
    """
    Retrieves all columns from the 'orders' table that match the given order number.
    :param order_num: The order number to look up.
    :return: str: A string representation of the order details, or a message if not found.
    """
    session = Session()
    try:
        order = session.query(Order).filter(Order.order_num == order_num).first()
        if order:
            details = (
                f"Order Number: {order.order_num}\n"
                f"Item: {order.item}\n"
                f"Quantity: {order.quantity}\n"
                f"Price: {order.price}\n"
                f"Status: {order.status}\n"
                f"Customer Name: {order.customer_name}\n"
                f"Order Date: {order.order_date.strftime('%Y-%m-%d') if order.order_date else 'Not yet scheduled'}\n"
                f"Total Amount: {order.total_amount}"
            )
            if order.shipping_date:  # Only orders with status == delivered will have shipping_date
                details += f"\nShipping Date: {order.shipping_date.strftime('%Y-%m-%d')}"
            return details
        else:
            return f"Order with {order_num} not found."
    finally:
        session.close()


# As an example, we limit this query to the first 20 results (to avoid extra charges with google API)
@tool
def get_pending_orders() -> str:
    """
    Retrieves the 20th pending orders from the 'orders' table, sorted by order date.
    :return: str: A string representation of the pending orders, or a message if no orders are found.
    """
    session = Session()
    try:
        orders = (session.query(Order).filter(Order.status == "pending").order_by(Order.order_date.asc())
                  .limit(20).all())
        if orders:
            details = "\n".join(str(order) for order in orders)
            return details
        else:
            return "No pending orders found."
    finally:
        session.close()


# As an example, we limit this query to the first 20 results (to avoid extra charges with google API)
@tool
def get_today_orders() -> str:
    """
    Retrieves all orders placed today, so the order date is equal to today's date.
    :return: str: A string representation of the today's orders, or a message if no orders are found.
    """
    session = Session()
    try:
        orders = (session.query(Order).filter(Order.order_date == datetime.today().date()).limit(20).all())
        if orders:
            details = "\n".join(str(order) for order in orders)
            return details
        else:
            return "No orders placed today."
    finally:
        session.close()


# As an example, we limit this query to the first 20 results (to avoid extra charges with google API)
@tool
def get_lead_time() -> str:
    """
    Retrieves the calculated lead time (shipping date - order date) plus the orders details for all delivered orders.
    :return: str: A string representation of the calculated lead time and delivered orders,
                    or a message if no orders are found.
    """
    session = Session()
    try:
        orders_with_lead_time = (
            session.query(
                Order.order_num,
                Order.item,
                Order.quantity,
                Order.price,
                Order.customer_name,
                (func.julianday(Order.shipping_date) - func.julianday(Order.order_date)).label("lead_time")
            )
            .filter(Order.status == 'delivered')
            .limit(20)
            .all()
        )
        if orders_with_lead_time:
            details = []
            for order in orders_with_lead_time:
                details.append(
                    f"Order Num: {order.order_num}, Item: {order.item}, Quantity: {order.quantity}, "
                    f"Price: {order.price}, Customer: {order.customer_name}, Lead time: {order.lead_time}"
                )
            return "\n".join(details)
        else:
            return "No delivered orders found with shipping date information."
    finally:
        session.close()


# Create the Agent state which will store some variables and go through all the graph steps
class AgentSate(TypedDict):
    chat_history: list[BaseMessage]
    order_num: Optional[str]
    order_details: Optional[str]
    intent: Optional[str]

# First node, will be the entry point to the graph
def classify_intent(state: AgentSate) -> AgentSate:
    """
    Analyzes the last message from the chat history in the given state and determines
    the user's primary intent based on predefined categories. The identified intent
    is returned within an updated state.

    :param state: The current state of the agent as an `AgentState` object. It includes
        chat history and other contextual information.
    :return: A new dictionary containing the inferred intent as the key `"intent"`.
        Will update the state with the new intent.
    """
    last_message = state["chat_history"][-1]
    prompt = (
        f"Analyze the following user query and determine their primary intent. "
        f"Respond with 'order_status' if they are asking about an order's status or details. "
        f"Respond with 'pending_orders' if they are asking about pending orders. "
        f"Respond with 'today_orders' if they are asking about today's orders. "
        f"Respond with 'lead_time' if they are asking about the calculated lead time for delivered orders. "
        f"Respond with 'general_query' for any other type of question. "
        f"User query: {last_message.content}"
    )
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    print(f"Classified intent: {intent}")
    return {"intent": intent}


# Conditional node
def extract_order_num(state: AgentSate) -> AgentSate:
    """
    Extracts the order number from the last user query in the chat history of
    the given `AgentSate` object.

    The function retrieves the last user message and utilizes a predefined
    large language model (LLM) to extract a 9-digit numerical order number
    present in the message. If no clear order number can be determined, the
    function explicitly sets the return value to `None`.

    :param state: The AgentSate object containing the chat history. It must
                  include the latest user query under the key `chat_history`.
    :return: A dictionary containing the extracted order number under the key
             `order_num`. If no order number is found, the value will be `None`.
             Will update the state with the extracted order number.
    """
    last_message = state["chat_history"][-1]
    prompt = (
        f"Extract the order number from the following user query. "
        f"The order number usually has 9 numerical digits. "
        f"If the order number is found, return only the number. "
        f"If no clear order number is found, return 'None'. "
        f"User query: {last_message.content}"
    )
    response = llm.invoke(prompt)
    order_num = response.content.strip()
    if order_num.lower() == "none":
        order_num = None
    print(f"Extracted order number: {order_num}")
    return {"order_num": order_num}


def call_tool_node(state: AgentSate) -> AgentSate:
    """
    Calls the tool node to fetch order details based on the provided agent state.

    This function retrieves the "order_num" from the given state. If the "order_num"
    is available, it invokes the `get_order_status` tool to get the corresponding
    order details. If the "order_num" is not found, it returns a fallback response
    indicating the absence of an order number.

    :param state: Current agent state containing relevant data for tool invocation
                  (expected to include an "order_num").
    :return: Updated agent state including order details retrieved from the tool
             or a fallback message when order number is not found.
    """
    order_num = state.get("order_num")
    if order_num:
        order_details = get_order_status.invoke({"order_num": order_num})
        print(f"Tool returned order details: {order_details}")
        return {"order_details": order_details}
    else:
        return {"order_details": "No order number found."}


def call_pending_orders_tool(state: AgentSate) -> AgentSate:
    """
    Call the pending orders tool and retrieve order details.

    This function uses the `get_pending_orders` tool to fetch details about all
    pending orders and returns an updated state containing the order details.
    It also logs the order details fetched for reference.

    :param state: Current state of the agent.
    :return: Updated state containing order details.
    """
    pending_order_details = get_pending_orders.invoke({})
    print(f"Tool returned order details: {pending_order_details}")
    return {"order_details": pending_order_details}


def call_today_orders_tool(state: AgentSate) -> AgentSate:
    """
    Calls a tool to fetch today's order details and updates the agent state with
    the retrieved data.

    :param state: Current state of the agent.
    :return: Updated state of the agent with the fetched order details.
    """
    today_order_details = get_today_orders.invoke({})
    print(f"Tool returned order details: {today_order_details}")
    return {"order_details": today_order_details}


def call_lead_time_tool(state: AgentSate) -> AgentSate:
    """
    Call the lead time tool for processing and retrieving order details. This function
    invokes the `get_lead_time` tool and returns the results as part of the state.

    :param state: The current state object of type AgentSate.
    :return: The updated state of type AgentSate containing `order_details` retrieved
        from the lead time tool.
    """
    lead_time_details = get_lead_time.invoke({})
    print(f"Tool returned order details: {lead_time_details}")
    return {"order_details": lead_time_details}


def generate_response(state: AgentSate) -> AgentSate:
    """
    Generates a response based on the user's intent and provided details, using a prompt-driven AI model.
    The method tailors the generated response according to the user's query by utilizing the state of the conversation,
    including chat history, intent, and relevant order details. Various response scenarios are considered based on
    specific intents: `order_status`, `pending_orders`, `today_orders`, and `lead_time`. If the intent is not recognized,
    a general guidance message is provided to inform the user about the supported functionalities.

    :param state: Dictionary containing the context of the conversation. It includes:
        - chat_history (list): List of past interactions between the user and the assistant, including `last_message`.
        - intent (str): Indicates the type of user query such as "order_status", "pending_orders", etc.
        - order_details (str, optional): Additional details about a specific order or orders.
        - order_num (str, optional): Order number related to the user's query.

    :return: Updated state containing the new chat history with the AI response appended.
    """
    chat_history = state["chat_history"]
    last_message = chat_history[-1]
    intent = state.get("intent")
    order_details = state.get("order_details")
    order_num = state.get("order_num")
    if intent == "order_status":
        if order_details and "not found" not in order_details.lower() and "error" not in order_details.lower():
            response_prompt = (
                f"Based on the following order details, try to answer the user question: {last_message.content}. "
                f"If the question is no clear, give a summary of the order to the user. "
                f"Order details:\n{order_details}"
            )
        elif order_details and ("not found" in order_details.lower() or "error" in order_details.lower()):
            response_prompt = (
                f"The order number '{order_num}' was not found or an error occurred. "
                f"Please politely inform the user and ask them to double-check the order number."
            )
        else:
            response_prompt = (
                f"You were asked about an order status, but no specific order number was provided or found. "
                f"Please politely ask the user for the order number so you can assist them. "
                f"The original query was: {last_message.content}"
            )
    elif intent == "pending_orders":
        if order_details and "no pending orders found" not in order_details.lower():
            response_prompt = (
                f"Based on the following pending orders, try to answer the user question: {last_message.content}. "
                f"If the question is no clear, give a summary of the pending orders to the user. "
                f"Here are the pending orders:\n {order_details}."
            )
    elif intent == "today_orders":
        if order_details and "no orders placed today" not in order_details.lower():
            response_prompt = (
                f"Based on the following orders placed today, try to answer the user question: {last_message.content}. "
                f"If the question is no clear, give a summary of the orders placed today to the user. "
                f"Here are the orders placed today:\n {order_details}."
            )
    elif intent == "lead_time":
        if order_details and "no delivered orders found" not in order_details.lower():
            response_prompt = (
                f"Based on the following calculated lead time and delivered orders, "
                f"try to answer the user question: {last_message.content}. "
                f"If the question is no clear, give a summary of the calculated lead time "
                f"and delivered orders to the user. "
                f"Here is the calculated lead time and delivered orders:\n {order_details}."
            )

    else:
        response_prompt = (
            f"You are a helpful assistant. "
            f"Please inform the user that you can only help him with the following topics: "
            f"Get the details of a particular order, "
            f"Get the details of the pending orders, "
            f"Get the details of today's orders, "
            f"Get the effective lead time of the shipped orders."
        )

    ai_response = llm.invoke(response_prompt)
    new_chat_history = chat_history + [AIMessage(content=ai_response.content)]

    # We have to trim the chat history so it doesn't exceed the maximum context window of the LLM
    trimmed_chat_history = trim_messages(
        new_chat_history,
        max_tokens=800_000, # Gemini has around 1_000_000 token window
        token_counter=llm,
        strategy="last", # We will keep only the most recent messages
        include_system=True # also delete the old system messages
    )
    print(f"Generated response: {ai_response.content}")
    return {"chat_history": trimmed_chat_history}


def route_intent(state: AgentSate) -> str:
    """
    Routes the current intent within the agent's state to the appropriate action or tool. Based on the
    provided intent, this function determines and returns a string identifier representing the next
    action the agent should take. If the intent is unrecognized, a default response action is returned.

    :param state: The current state of the agent, including information about the intent.

    :return: A string representing the next action or tool to invoke based on the intent.
    """
    intent = state.get("intent")
    if intent == "order_status":
        return "extract_order_num"
    elif intent == "pending_orders":
        return "call_pending_orders_tool"
    elif intent == "today_orders":
        return "call_today_orders_tool"
    elif intent == "lead_time":
        return "call_lead_time_tool"
    elif intent == "general_query":
        return "generate_response"
    else:
        return "generate_response"




# Init the graph (we will use the StateGraph type)
workflow = StateGraph(AgentSate)

# Add all the nodes to the graph:
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("extract_order_num", extract_order_num)
workflow.add_node("call_tool_node", call_tool_node)
workflow.add_node("call_pending_orders_tool", call_pending_orders_tool)
workflow.add_node("call_today_orders_tool", call_today_orders_tool)
workflow.add_node("call_lead_time_tool", call_lead_time_tool)
workflow.add_node("generate_response", generate_response)

# We set the classifier as the entry point
workflow.set_entry_point("classify_intent")

# Add the conditional edge (routing), will execute the corresponding node based on the LLM response
workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "extract_order_num": "extract_order_num",
        "call_pending_orders_tool": "call_pending_orders_tool",
        "call_today_orders_tool": "call_today_orders_tool",
        "call_lead_time_tool": "call_lead_time_tool",
        "generate_response": "generate_response"
    }
)

# Define the edges (connections between the nodes)
workflow.add_edge("extract_order_num", "call_tool_node")
workflow.add_edge("call_tool_node", "generate_response")
workflow.add_edge("call_pending_orders_tool", "generate_response")
workflow.add_edge("call_today_orders_tool", "generate_response")
workflow.add_edge("call_lead_time_tool", "generate_response")
workflow.add_edge("generate_response", END)

# Compile the graph
app = workflow.compile()


def main():
    """
    Example to run this chatbot from the command line without UI.
    ($ python main.py)
    :return: None
    """
    print("--- LLM Agent for orders status and orders related questions ---")
    print("Type 'exit' to quit.")

    chat_history = []

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        chat_history.append(HumanMessage(content=user_input))

        final_state = None
        for s in app.stream({"chat_history": chat_history}):
            print(s)
            final_state = s

        if final_state and "generate_response" in final_state:
            updated_chat_history = final_state["generate_response"]["chat_history"]
            ai_message = updated_chat_history[-1]
            print(f"AI: {ai_message.content}")
            chat_history = updated_chat_history

        else:
            print("AI: Sorry, I didn't understand that. Please try again.")
            chat_history.pop()


if __name__ == "__main__":
    main() # app entry point
