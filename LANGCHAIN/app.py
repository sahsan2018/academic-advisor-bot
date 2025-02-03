import streamlit as st
from chains import rag_chain
from history import get_session_history
from langchain_core.runnables.history import RunnableWithMessageHistory
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default_session"
    if "last_input" not in st.session_state:
        st.session_state.last_input = None

def format_message(text, is_user=False):
    if is_user:
        align = "right"
        bg_color = "linear-gradient(135deg, #6366f1, #4f46e5)"
        border_radius = "20px 20px 5px 20px"
    else:
        align = "left"
        bg_color = "linear-gradient(135deg, #1e1e38, #242447)"
        border_radius = "20px 20px 20px 5px"
    
    return f"""
    <div style="display: flex; justify-content: {align}; margin: 15px 0;">
        <div style="background: {bg_color}; 
                    padding: 15px 20px; 
                    border-radius: {border_radius}; 
                    max-width: 80%; 
                    font-size: 16px; 
                    line-height: 1.6;
                    color: white;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                    animation: fadeIn 0.5s ease-out;
                    border: 1px solid rgba(255, 255, 255, 0.1);">
            {text}
        </div>
    </div>
    """

def custom_css():
    return """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .stTextInput > div > div > input {
            background-color: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 15px;
            padding: 15px 20px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #4f46e5;
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 15px 30px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(79, 70, 229, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(79, 70, 229, 0.3);
        }
        
        .main {
            background: linear-gradient(135deg, #f8fafc, #eef2ff);
        }
    </style>
    """

def handle_submit():
    if st.session_state.user_input and len(st.session_state.user_input.strip()) > 0:
        user_message = st.session_state.user_input.strip()
        
        # Prevent duplicate messages
        if st.session_state.last_input != user_message:
            st.session_state.last_input = user_message
            
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Get response from the chain
            result = conversational_rag_chain.invoke(
                {"input": user_message},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )["answer"]
            
            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })

        # Reset input box
        st.session_state.user_input = ""

def main():
    st.set_page_config(page_title="Course Assistant", layout="wide")
    init_session_state()
    
    # Apply custom CSS
    st.markdown(custom_css(), unsafe_allow_html=True)
    
    # Header with improved styling
    st.markdown("""
        <div style="text-align: center; padding: 30px 0; background: linear-gradient(135deg, #6366f1, #4f46e5); margin: -50px -50px 20px -50px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            <h1 style="color: white; font-size: 40px; font-weight: 700; margin-bottom: 10px; font-family: 'Arial', sans-serif;">
                Course Assistant
            </h1>
            <p style="color: rgba(255, 255, 255, 0.9); font-size: 18px; font-weight: 400;">
                Ask questions about your course material!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            st.markdown(
                format_message(
                    message["content"],
                    message["role"] == "user"
                ),
                unsafe_allow_html=True
            )
    
    # Input area with improved styling
    st.markdown("<div style='position: fixed; bottom: 0; left: 0; right: 0; padding: 20px; background: white; box-shadow: 0 -4px 15px rgba(0, 0, 0, 0.1);'>", unsafe_allow_html=True)
    cols = st.columns([7, 1])
    with cols[0]:
        st.text_input(
            "",
            key="user_input",
            placeholder="Ask about courses...",
            on_change=handle_submit,
            label_visibility="collapsed"
        )
    with cols[1]:
        st.button("Send", on_click=handle_submit, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 20px 0; color: #6b7280; font-size: 14px;">
            Built with ❤️ to help students succeed
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()