import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import the chatbot functionality from main.py
from main import app as chatbot_app

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Orders Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Add custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.bot {
        background-color: #e3f2fd;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .content {
        width: 80%;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("Orders Chatbot")
st.markdown("Ask about your orders or request statistics about them!")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
def display_messages():
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.markdown(f"""
            <div class="chat-message user">
                <div class="content">
                    <p><strong>You:</strong> {message.content}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            # First render the markdown content to HTML
            from streamlit.components.v1 import html
            import markdown
            
            # Convert markdown to HTML
            md_content = markdown.markdown(message.content)
            
            # Create the complete HTML with the chat bubble styling
            content_html = f"""
            <div class="chat-message bot">
                <div class="content">
                    <p><strong>AI:</strong></p>
                    <div>{md_content}</div>
                </div>
            </div>
            """
            st.markdown(content_html, unsafe_allow_html=True)

# Display chat interface
display_messages()

# User input
user_input = st.chat_input("Type your message here...")

# Process user input
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Display updated chat
    display_messages()
    
    # Process with chatbot
    with st.spinner("Thinking..."):
        # Process the message through the chatbot
        final_state = None
        for s in chatbot_app.stream({"chat_history": st.session_state.chat_history}):
            final_state = s


        def _stream_chunks(text: str, chunk_size: int = 20):
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]


        if final_state and "generate_response" in final_state:
            updated_chat_history = final_state["generate_response"]["chat_history"]

            # Stream the last AI message progressively to the UI
            last_ai_message = updated_chat_history[-1] if updated_chat_history else None
            if isinstance(last_ai_message, AIMessage):
                with st.chat_message("assistant"):
                    st.write_stream(_stream_chunks(last_ai_message.content, chunk_size=32))

            # Persist the full chat history after streaming
            st.session_state.chat_history = updated_chat_history

            # Force a rerun to render the message in the styled bubbles as well
            st.rerun()
        else:
            st.session_state.chat_history.append(
                AIMessage(content="Sorry, I didn't understand that. Please try again."))
            st.session_state.chat_history.pop(-2)  # Remove the user message that caused the error
            st.rerun()

# Add a button to clear the chat history
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
