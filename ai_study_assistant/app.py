import io
import os
import time
import streamlit as st
import PyPDF2
from openai import OpenAI, AuthenticationError

from agents import StudyAssistant

# Set page config
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize session state
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "materials_processed" not in st.session_state:
    st.session_state.materials_processed = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "chat"
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

# Function to validate API key
def validate_api_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True
    except AuthenticationError:
        return False
    except Exception as e:
        print(f"Error validating API key: {e}")
        return False

# API Key Popup
def show_api_key_popup():
    with st.container():
        st.markdown('<div class="api-popup">', unsafe_allow_html=True)
        st.image(
            "https://media.licdn.com/dms/image/v2/C5612AQFjOmJ_0UmzIQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1635826425447?e=2147483647&v=beta&t=JtdnhvzjN_5dVKkR6toDC1VEng56AVwtwAGQCu3pRYI",
            width=80,
        )
        st.header("AI Study Assistant 📚", anchor=False)
        st.header("Enter your OpenAI API key to start")
        with st.form("api_key_form"):
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-...",
                help="Find your key at https://platform.openai.com/api-keys",
            )
            submitted = st.form_submit_button("Submit")
            if submitted:
                if not api_key:
                    st.error("Please enter an API key.")
                elif validate_api_key(api_key):
                    st.session_state.api_key = api_key
                    st.session_state.api_key_valid = True
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.session_state.assistant = StudyAssistant()
                    st.success("API key validated. Loading...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid API key. Please try again.")
        
        st.markdown("### Why do I need an API key?")
        st.caption("##### This app uses OpenAI's models for study assistance, requiring a valid API key.")
        st.markdown("### How to get one?")
        st.caption("1. Go to [OpenAI's API Keys page](https://platform.openai.com/api-keys)\n2. Sign up or log in\n3. Generate a new key\n4. Copy and paste it here")
        st.markdown('</div>', unsafe_allow_html=True)

# Block app if API key is not valid
if not st.session_state.api_key_valid:
    show_api_key_popup()
    st.stop()

assistant = st.session_state.assistant

# Utility Functions
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            for char in ["\n\n", ".", "?", "!"]:
                pos = text.rfind(char, start, end)
                if pos != -1:
                    end = pos + 1
                    break
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

def clear_chat():
    st.session_state.chat_history = []

def clear_materials():
    try:
        assistant.vector_store.drop()
        assistant.setup_vector_store()
        st.session_state.materials_processed = {}
        st.success("Study materials cleared.")
    except Exception as e:
        st.error(f"Error clearing materials: {str(e)}")

# Sidebar
with st.sidebar:
    st.image(
        "https://media.licdn.com/dms/image/v2/C5612AQFjOmJ_0UmzIQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1635826425447?e=2147483647&v=beta&t=JtdnhvzjN_5dVKkR6toDC1VEng56AVwtwAGQCu3pRYI",
        width=40,
    )
    st.header("AI Study Assistant", anchor=False)
    st.caption("Your personal tutor")
    
    st.markdown("**Tech Stack**")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Model:")
        st.caption("Vector DB:")
    with col2:
        st.caption("GPT-4o")
        st.caption("Milvus Lite")
    
    st.divider()
    st.subheader("📚 Materials", anchor=False)
    if st.button("Clear Materials", key="clear_materials", use_container_width=True):
        clear_materials()
    
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    if uploaded_files:
        for pdf_file in uploaded_files:
            file_id = f"{pdf_file.name}_{pdf_file.size}"
            if file_id not in st.session_state.materials_processed:
                with st.spinner(f"Processing {pdf_file.name}..."):
                    try:
                        text = extract_text_from_pdf(pdf_file)
                        chunks = chunk_text(text)
                        chunk_progress = st.progress(0)
                        for i, chunk in enumerate(chunks):
                            metadata = {
                                "source": pdf_file.name,
                                "chunk": i,
                                "total_chunks": len(chunks),
                            }
                            assistant.store_material(chunk, metadata)
                            chunk_progress.progress((i + 1) / len(chunks))
                        st.session_state.materials_processed[file_id] = {
                            "name": pdf_file.name,
                            "chunks": len(chunks),
                        }
                        st.success(f"{pdf_file.name} ({len(chunks)} chunks)")
                    except Exception as e:
                        st.error(f"Error processing {pdf_file.name}: {str(e)}")
            else:
                processed = st.session_state.materials_processed[file_id]
                st.info(f"{processed['name']} ({processed['chunks']} chunks)")
    
    st.divider()
    st.subheader("🧭 Navigation", anchor=False)
    nav_buttons = [
        ("Chat", "chat"),
        ("Concepts", "concepts"),
        ("Flashcards", "flashcards"),
        ("Quiz", "quiz"),
        ("Summarizer", "summarizer"),
    ]
    for label, tab in nav_buttons:
        if st.button(label, key=f"nav_{tab}", use_container_width=True):
            st.session_state.current_tab = tab
            st.rerun()

# Tab Functions
def render_chat():
    st.header("Chat", anchor=False)
    st.caption("Ask about your study materials")
    
    chat_container = st.container()
    with chat_container:
        for exchange in st.session_state.chat_history:
            st.markdown(f'<div class="chat-message-user">You: {exchange["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message-ai">AI: {exchange["assistant"]}</div>', unsafe_allow_html=True)
    
    with st.form("chat_form"):
        user_input = st.text_area("Question:", placeholder="Ask about your materials...", height=80)
        col1, col2 = st.columns([1, 1])
        with col1:
            submitted = st.form_submit_button("Send")
        with col2:
            cleared = st.form_submit_button("Clear")
        
        if submitted:
            if user_input:
                with st.spinner("Thinking..."):
                    try:
                        response = assistant.generate_response(user_input)
                        st.session_state.chat_history.append({"user": user_input, "assistant": response})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question.")
        if cleared:
            clear_chat()
            st.rerun()

def render_concepts():
    st.header("Concepts", anchor=False)
    st.caption("Explore detailed explanations")
    
    with st.form("concept_form"):
        concept = st.text_input("Concept:", placeholder="e.g., Photosynthesis...")
        submitted = st.form_submit_button("Explain")
        if submitted:
            if concept:
                with st.spinner(f"Explaining {concept}..."):
                    try:
                        explanation = assistant.explain_concept(concept)
                        st.subheader(concept, anchor=False)
                        st.markdown(explanation)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a concept.")

def render_flashcards():
    st.header("Flashcards", anchor=False)
    st.caption("Test your knowledge")
    
    with st.form("flashcards_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            topic = st.text_input("Topic:", placeholder="e.g., Cell Biology...")
        with col2:
            num_cards = st.number_input("Cards:", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Generate")
        
        if submitted:
            if topic:
                with st.spinner(f"Generating {num_cards} flashcards..."):
                    try:
                        flashcards = assistant.generate_flashcards(topic, num_cards)
                        st.session_state.flashcards = flashcards
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a topic.")
    
    if st.session_state.flashcards:
        st.subheader(f"Flashcards: {topic}", anchor=False)
        for i, card in enumerate(st.session_state.flashcards):
            with st.expander(f"Card {i+1}: {card['question']}"):
                st.write(f"**Answer:** {card['answer']}")

def render_quiz():
    st.header("Quiz", anchor=False)
    st.caption("Challenge yourself")
    
    with st.form("quiz_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            topic = st.text_input("Topic:", placeholder="e.g., Ancient Rome...")
        with col2:
            num_questions = st.number_input("Questions:", min_value=1, max_value=10, value=5)
        submitted = st.form_submit_button("Generate")
        
        if submitted:
            if topic:
                with st.spinner(f"Generating {num_questions} questions..."):
                    try:
                        quiz = assistant.generate_quiz(topic, num_questions)
                        st.session_state.quiz = quiz
                        st.session_state.quiz_answers = {}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a topic.")
    
    if st.session_state.quiz:
        st.subheader(f"Quiz: {topic}", anchor=False)
        for i, question in enumerate(st.session_state.quiz):
            with st.container():
                st.write(f"**Question {i+1}:** {question['question']}")
                selected_option = st.radio(
                    f"Answer for question {i+1}:",
                    question["options"],
                    key=f"q{i}",
                )
                selected_index = question["options"].index(selected_option)
                st.session_state.quiz_answers[i] = selected_index
                if st.button(f"Check Answer {i+1}", key=f"check_{i}"):
                    if selected_index == question["correct_index"]:
                        st.success("Correct!")
                    else:
                        st.error(f"Wrong. Correct: {question['options'][question['correct_index']]}")
                    st.info(f"**Explanation:** {question['explanation']}")
                st.divider()
        
        if st.button("Check All", key="check_all"):
            score = sum(
                1 for i, q in enumerate(st.session_state.quiz)
                if i in st.session_state.quiz_answers and
                st.session_state.quiz_answers[i] == q["correct_index"]
            )
            st.success(f"Score: {score}/{len(st.session_state.quiz)}")

def render_summarizer():
    st.header("Summarizer", anchor=False)
    st.caption("Condense your notes")
    
    with st.form("summarizer_form"):
        text_to_summarize = st.text_area(
            "Text:",
            placeholder="Paste your material...",
            height=160,
        )
        col1, col2 = st.columns([2, 1])
        with col2:
            max_length = st.number_input(
                "Max length:",
                min_value=100,
                max_value=2000,
                value=500,
            )
        submitted = st.form_submit_button("Summarize")
        
        if submitted:
            if text_to_summarize:
                with st.spinner("Summarizing..."):
                    try:
                        summary = assistant.summarize_material(text_to_summarize, max_length)
                        st.subheader("Summary", anchor=False)
                        st.markdown(summary)
                        st.info(f"Length: {len(summary)} characters")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter text.")

# Main Content
with st.container():
    if st.session_state.current_tab == "chat":
        render_chat()
    elif st.session_state.current_tab == "concepts":
        render_concepts()
    elif st.session_state.current_tab == "flashcards":
        render_flashcards()
    elif st.session_state.current_tab == "quiz":
        render_quiz()
    elif st.session_state.current_tab == "summarizer":
        render_summarizer()

# Footer
st.markdown("---")
st.markdown("AI Study Assistant | Created by Manish Sharma")
st.markdown("[Follow @lucifer_x007 on X](https://x.com/lucifer_x007)")
st.markdown("Powered by OpenAI")