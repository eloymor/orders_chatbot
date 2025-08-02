import os
from typing import TypedDict, Optional
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

load_dotenv()

api_key: str = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=api_key)

engine = create_engine('sqlite:///orders.db')
Base = declarative_base()


class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    order_num = Column(String)
    item = Column(String)
    status = Column(String)
    customer_name = Column(String)
    order_date = Column(Date)

    def __repr__(self):
        return (f"<Order(order_num='{self.order_num}', item='{self.item}', "
                f"status='{self.status}', customer_name='{self.customer_name}', "
                f"'{self.order_date.strftime("%d/%m/%Y")}')>")


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
                f"Status: {order.status}\n"
                f"Customer Name: {order.customer_name}\n"
                f"Order Date: {order.order_date.strftime('%Y-%m-%d') if order.order_date else 'Not yet scheduled'}"
            )
            return details
        else:
            return f"Order with {order_num} not found."
    finally:
        session.close()


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


class AgentSate(TypedDict):
    chat_history: list[BaseMessage]
    order_num: Optional[str]
    order_details: Optional[str]
    intent: Optional[str]


def classify_intent(state: AgentSate) -> AgentSate:
    last_message = state["chat_history"][-1]
    prompt = (
        f"Analyze the following user query and determine their primary intent. "
        f"Respond with 'order_status' if they are asking about an order's status or details. "
        f"Respond with 'pending_orders' if they are asking about pending orders. "
        f"Respond with 'today_orders' if they are asking about today's orders. "
        f"Respond with 'general_query' for any other type of question. "
        f"User query: {last_message.content}"
    )
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    print(f"Classified intent: {intent}")
    return {"intent": intent}


def extract_order_num(state: AgentSate) -> AgentSate:
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
    order_num = state.get("order_num")
    if order_num:
        order_details = get_order_status.invoke({"order_num": order_num})
        print(f"Tool returned order details: {order_details}")
        return {"order_details": order_details}
    else:
        return {"order_details": "No order number found."}


def call_pending_orders_tool(state: AgentSate) -> AgentSate:
    pending_order_details = get_pending_orders.invoke({})
    print(f"Tool returned order details: {pending_order_details}")
    return {"order_details": pending_order_details}


def call_today_orders_tool(state: AgentSate) -> AgentSate:
    today_order_details = get_today_orders.invoke({})
    print(f"Tool returned order details: {today_order_details}")
    return {"order_details": today_order_details}


def generate_response(state: AgentSate) -> AgentSate:
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

    else:
        response_prompt = (
            f"You are a helpful assistant. Respond to the following general query: "
            f"{last_message.content}"
        )

    ai_response = llm.invoke(response_prompt)
    new_chat_history = chat_history + [AIMessage(content=ai_response.content)]
    print(f"Generated response: {ai_response.content}")
    return {"chat_history": new_chat_history}


def route_intent(state: AgentSate) -> str:
    intent = state.get("intent")
    if intent == "order_status":
        return "extract_order_num"
    elif intent == "pending_orders":
        return "call_pending_orders_tool"
    elif intent == "today_orders":
        return "call_today_orders_tool"
    elif intent == "general_query":
        return "generate_response"
    else:
        return "generate_response"


workflow = StateGraph(AgentSate)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("extract_order_num", extract_order_num)
workflow.add_node("call_tool_node", call_tool_node)
workflow.add_node("call_pending_orders_tool", call_pending_orders_tool)
workflow.add_node("call_today_orders_tool", call_today_orders_tool)
workflow.add_node("generate_response", generate_response)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "extract_order_num": "extract_order_num",
        "call_pending_orders_tool": "call_pending_orders_tool",
        "call_today_orders_tool": "call_today_orders_tool",
        "generate_response": "generate_response"
    }
)

workflow.add_edge("extract_order_num", "call_tool_node")
workflow.add_edge("call_tool_node", "generate_response")
workflow.add_edge("call_pending_orders_tool", "generate_response")
workflow.add_edge("call_today_orders_tool", "generate_response")
workflow.add_edge("generate_response", END)

app = workflow.compile()


def main():
    print("--- LLM Agent for Order Status and General Queries ---")
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
    main()
