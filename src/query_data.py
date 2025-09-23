################################################STREAMLIT APP BELOW with GROQ VERSION################################################33


import streamlit as st
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load env variables
load_dotenv()

# Retrieval settings
K_RETRIEVE = 25   # get more candidates first
K_RETURN   = 8     # send more top docs to the LLM for richer context
THRESHOLD  = 0.4   # include more relevant matches

# LLM (Groq via LangChain)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="deepseek-r1-distill-llama-70b",
    temperature=0.0
)

# Prompt template
PROMPT_TEMPLATE = """
You are a friendly legal assistant specialized in anti-narcotics criminal law.
Your job is to answer questions using only the information provided in the context below.

Instructions:
- If the user's question is a greeting or general inquiry (like "hi", "hello", "what can you do?"), respond politely and explain your role and how you can help.
- If the answer is not present in the context, say "I don't know based on the provided documents." Do NOT make up information or guess.
- If the answer is present, give a full, accurate, and to-the-point answer using all relevant details from the context. If the answer is a list, enumerate all items clearly.
- Format your response with a clear "Answer:" section.
- When citing legal information, reference the specific statute, regulation, or case from the provided materials.

Context:
{context}

Conversation history:
{history}

User's question: {question}
"""

# Embedding function (FREE HuggingFace)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"}
)

# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "anf-bot-2"
db = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_function
)

# Streamlit UI header
st.markdown("""
<style>
/* Hide Streamlit default UI elements */
header, [data-testid="stSidebar"], footer, .stDeployButton, .stActionButton, .stStatusWidget, .stSpinner, .stNotification, .stToast {
    display: none !important;
}
[data-testid="stAppViewContainer"], .block-container {
    background: #f6fff8 !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 100vw !important;
    max-width: 100vw !important;
}
.chat {
    max-width: 600px;
    margin: 0 auto;
    padding-bottom: 80px;
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
}
.header {
    width: 100%;
    background: #14532d;
    color: #fff;
    text-align: center;
    padding: 32px 0 20px 0;
    margin-bottom: 16px;
    border-radius: 0 0 24px 24px;
    box-shadow: 0 2px 8px rgba(20,83,45,0.08);
}
.header img {
    width: 56px;
    height: 56px;
    border-radius: 14px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(20,83,45,0.10);
}
.header .title h1 {
    font-size: 28px;
    margin: 0;
    font-weight: 700;
    letter-spacing: 1px;
}
.message-row {
    display: flex;
    width: 100%;
    margin-bottom: 18px;
}
.message-row.assistant {
    justify-content: flex-start;
}
.message-row.user {
    justify-content: flex-end;
}
.message-row.user .avatar {
    order: 2;           /* Avatar appears after bubble */
    margin-left: 5px;   /* Space between bubble and avatar */
    margin-right: 2;
}
.message-row.user .bubble.user {
    order: 1;           /* Bubble appears first */
}
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #eee;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(20,83,45,0.10);
    flex-shrink: 0;
    margin-bottom: auto;
    margin-top: 2px;
}
.avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.bubble {
    background: #14532d;
    color: #fff;
    border-radius: 18px;
    padding: 16px 20px;
    max-width: 70%;
    font-size: 18px;
    line-height: 1.7;
    box-shadow: 0 2px 8px rgba(20,83,45,0.08);
    word-break: break-word;
    margin: 0 10px;
    text-align: left;
    box-sizing: border-box;
    overflow-wrap: anywhere;
}
.bubble.user {
    background: #1e7c4c;
}
.bubble.assistant {
    background: #14532d;
}
.stChatInputContainer {
    background: #fff !important;
    border-top: 1px solid #e0e0e0 !important;
    box-shadow: 0 -2px 8px rgba(20,83,45,0.05);
    padding: 12px 0 !important;
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 100;
}
.stChatInput {
    background: #f6fff8 !important;
    border-radius: 18px !important;
    border: 2px solid #14532d !important;
    font-size: 18px !important;
    color: #14532d !important;
    padding: 14px 20px !important;
    margin: 0 auto !important;
    width: 95% !important;
    max-width: 570px !important;
    box-sizing: border-box;
    box-shadow: 0 2px 8px rgba(20,83,45,0.08);
}
@media (max-width: 600px) {
    .chat {
        max-width: 100vw;
        padding-left: 0;
        padding-right: 0;
    }
    .header .title h1 {
        font-size: 20px;
    }
    .header img {
        width: 40px;
        height: 40px;
    }
    .avatar {
        width: 28px;
        height: 28px;
    }
    .bubble {
        font-size: 16px;
        padding: 10px 12px;
        max-width: 90vw;
    }
    .stChatInput {
        font-size: 16px !important;
        padding: 10px 12px !important;
        width: 98vw !important;
        max-width: 98vw !important;
    }
}
</style>
<div class="chat">
  <div class="header">
    <img src="https://upload.wikimedia.org/wikipedia/en/7/70/Anti-Narcotics_Force_Logo.png"/>
    <div class="title"><h1>ANF Academy Educational Chatbot</h1></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Avatar URLs
assistant_avatar_url = "https://upload.wikimedia.org/wikipedia/en/7/70/Anti-Narcotics_Force_Logo.png"
user_avatar_url = "https://cdn-icons-png.flaticon.com/512/9131/9131529.png"  # Replace with your preferred user icon

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history with custom bubbles and avatars
st.markdown('<div class="chat">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    avatar_url = user_avatar_url if role_class == "user" else assistant_avatar_url
    row_class = f"message-row {role_class}"
    st.markdown(
        f'''
        <div class="{row_class}">
            <div class="avatar"><img src="{avatar_url}" alt="{role_class} avatar"></div>
            <div class="bubble {role_class}">{msg["content"]}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# User input
if query := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(
        f'''
        <div class="message-row user">
            <div class="avatar"><img src="{user_avatar_url}" alt="user avatar"></div>
            <div class="bubble user">{query}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Build history
    # Only keep last 3 messages for history
    history = ""
    for msg in st.session_state.messages[-3:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    # Step 1: Retrieve many candidates
    raw_results = db.similarity_search_with_relevance_scores(query, k=K_RETRIEVE)

    # Step 2: Filter by threshold
    filtered = [(doc, score) for doc, score in raw_results if score >= THRESHOLD]

    # Step 3: Re-rank by cosine similarity (more accurate)
    if filtered:
        query_emb = embedding_function.embed_query(query)
        doc_embs = [embedding_function.embed_query(doc.page_content) for doc, _ in filtered]
        sims = cosine_similarity([query_emb], doc_embs)[0]
        reranked = sorted(zip(filtered, sims), key=lambda x: x[1], reverse=True)
        top_docs = [doc.page_content for (doc, _), _ in reranked[:K_RETURN]]

        MAX_DOC_LENGTH = 1200  # or lower
        top_docs = [doc[:MAX_DOC_LENGTH] for doc in top_docs]
        context_text = "\n\n".join(top_docs)
    else:
        context_text = ""

    # Step 4: Build prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, history=history, question=query)

    # Step 5: Get answer from LLM
    response_text = llm.predict(prompt)

    # Remove <think>...</think> blocks from a string
    def remove_think_tags(response: str) -> str:
        return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    response_text = remove_think_tags(response_text)

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.markdown(
        f'''
        <div class="message-row assistant">
            <div class="avatar"><img src="{assistant_avatar_url}" alt="assistant avatar"></div>
            <div class="bubble assistant">{response_text}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )