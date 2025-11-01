import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from datetime import datetime

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ENSA Tétouan Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Chat container */
    .stChatFloatingInputContainer {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling */
    h1 {
        color: white;
        text-align: center;
        font-size: 3em;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        margin-bottom: 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: white;
        font-size: 2em;
        font-weight: 700;
    }
    
    /* Info box styling */
    .info-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Input field styling */
    .stTextInput input {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Pinecone and models
@st.cache_resource
def init_components():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store

# Sidebar
with st.sidebar:
    st.title("🎓 ENSA Assistant")
    st.markdown("---")
    
    st.markdown("### 📊 Statistiques")
    if "messages" in st.session_state:
        msg_count = len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])
        st.metric("Messages envoyés", msg_count)
    
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.info(
        "Assistant virtuel spécialisé dans le Règlement intérieur "
        "des études de l'ENSA de Tétouan.\n\n"
        "💡 **Posez vos questions sur:**\n"
        "- Assiduité et absences\n"
        "- Notation et examens\n"
        "- Passage d'année\n"
        "- Stages et projets\n"
        "- Discipline académique"
    )
    
    st.markdown("---")
    
    # Temperature slider
    temperature = st.slider(
        "🌡️ Créativité des réponses",
        min_value=0.0,
        max_value=2.0,
        value=0.3,
        step=0.1,
        help="Plus la valeur est élevée, plus les réponses sont créatives"
    )
    
    # Clear chat button
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; padding: 10px;'>"
        "<small>Powered by OpenAI & Pinecone</small><br>"
        f"<small>{datetime.now().strftime('%d/%m/%Y')}</small>"
        "</div>",
        unsafe_allow_html=True
    )

# Main content
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("💬 Assistant ENSA Tétouan")
    st.markdown(
        "<p style='text-align: center; color: white; font-size: 1.2em;'>"
        "Posez vos questions sur le règlement intérieur des études"
        "</p>",
        unsafe_allow_html=True
    )

# Initialize components
vector_store = init_components()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        SystemMessage(
            "Tu es un assistant virtuel conçu pour répondre avec précision et "
            "clarté aux questions des étudiants de l'École Nationale des Sciences "
            "Appliquées de Tétouan (ENSA Tétouan) concernant le Règlement intérieur des études."
        )
    )

# Chat container
chat_container = st.container()

with chat_container:
    # Display welcome message if no messages yet
    if len(st.session_state.messages) == 1:
        st.markdown(
            """
            <div class='info-box'>
                <h3 style='color: #667eea; margin-top: 0;'>👋 Bienvenue !</h3>
                <p style='color: #333;'>
                    Je suis votre assistant virtuel pour toutes vos questions concernant 
                    le règlement intérieur de l'ENSA Tétouan.
                </p>
                <p style='color: #666;'>
                    <strong>Exemples de questions :</strong><br>
                    • "Quel est le taux d'absence autorisé ?"<br>
                    • "Comment se calcule la moyenne semestrielle ?"<br>
                    • "Quelles sont les conditions de passage en année supérieure ?"
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🎓"):
                st.markdown(message.content)

# Chat input
prompt = st.chat_input("Tapez votre question ici...", key="chat_input")

if prompt:
    # Add user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))
    
    # Show thinking spinner
    with st.spinner("🤔 Réflexion en cours..."):
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Retrieve relevant documents
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5},
        )
        
        docs = retriever.invoke(prompt)
        docs_text = "\n\n".join(d.page_content for d in docs)
        
        # Create system prompt
        system_prompt = """Tu es un assistant spécialisé dans les questions relatives au Règlement intérieur des études de l'ENSA de Tétouan.
Utilise les éléments de contexte suivants pour répondre avec exactitude et clarté.
Si l'information ne se trouve pas dans le contexte, indique simplement que tu ne le sais pas et suggère à l'étudiant de contacter le service de scolarité pour plus de précisions.
Reste toujours formel, respectueux et concis (maximum trois phrases).
Tes réponses doivent concerner uniquement les règles académiques : assiduité, notation, examens, passage d'année, redoublement, stages, projets ou discipline.

Contexte : {context}"""
        
        system_prompt_fmt = system_prompt.format(context=docs_text)
        st.session_state.messages.append(SystemMessage(system_prompt_fmt))
        
        # Get response
        result = llm.invoke(st.session_state.messages).content
        
        # Display assistant response
        with st.chat_message("assistant", avatar="🎓"):
            st.markdown(result)
        
        st.session_state.messages.append(AIMessage(result))
        
        # Show source documents count
        if docs:
            st.caption(f"📚 Réponse basée sur {len(docs)} document(s) pertinent(s)")