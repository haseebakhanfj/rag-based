import os
import tempfile
from typing import List, Tuple

import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f7ff;
        border: 1px solid #cfe5ff;
        margin-bottom: 15px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f8f9fa;
        border: 1px solid #dddddd;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📚 AI PDF RAG Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Upload a PDF and ask questions using Retrieval-Augmented Generation"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 5


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# GET GROQ CLIENT
# ============================================================

def get_groq_client():
    """
    Gets Groq API key from Streamlit secrets or environment variables.
    """

    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf_text(uploaded_file) -> Tuple[str, int]:
    """
    Extract text from an uploaded PDF.

    Returns:
        full_text
        number_of_pages
    """

    try:
        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        full_text = "\n\n".join(pages)

        return full_text, len(reader.pages)

    except Exception as e:
        raise RuntimeError(f"Could not read PDF: {str(e)}")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    """

    text = text.replace("\x00", " ")

    lines = []

    for line in text.splitlines():
        cleaned = " ".join(line.split())

        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines)


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:

    if not text:
        return []

    words = text.split()

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

        if start < 0:
            start = 0

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    chunks: List[str],
    model: SentenceTransformer,
) -> np.ndarray:

    if not chunks:
        return np.array([], dtype="float32")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype("float32")


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings: np.ndarray):

    if embeddings.size == 0:
        return None

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# SEARCH VECTOR DATABASE
# ============================================================

def search_faiss(
    query: str,
    index,
    chunks: List[str],
    model: SentenceTransformer,
    top_k: int = TOP_K,
):

    if index is None or not chunks:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0 or idx >= len(chunks):
            continue

        results.append(
            {
                "text": chunks[idx],
                "score": float(score),
                "index": int(idx),
            }
        )

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results) -> str:

    if not results:
        return ""

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"--- Retrieved Passage {i} ---\n"
            f"{result['text']}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    client: Groq,
    question: str,
    context: str,
):

    system_prompt = """
You are an intelligent document question-answering assistant.

Your job is to answer the user's question using the retrieved
information from the uploaded document.

Rules:

1. Use the provided context as the primary source.
2. Do not invent facts that are not supported by the context.
3. If the answer cannot be found in the context, clearly say:
   "The answer was not found in the uploaded document."
4. Give a clear and useful answer.
5. Use simple language unless technical terminology is necessary.
6. When appropriate, organize the answer using bullet points.
"""

    user_prompt = f"""
Retrieved document context:

{context}

User question:

{question}

Answer the question based on the retrieved document context.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    return response.choices[0].message.content


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "processed" not in st.session_state:
    st.session_state.processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("### RAG Configuration")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.write("---")

    st.write("### Embedding Model")

    st.caption(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    st.write("---")

    if st.session_state.processed:

        st.success("PDF processed successfully.")

        st.write(
            f"**Document:** "
            f"{st.session_state.document_name}"
        )

        st.write(
            f"**Pages:** "
            f"{st.session_state.page_count}"
        )

        st.write(
            f"**Chunks:** "
            f"{len(st.session_state.chunks)}"
        )

    else:

        st.info(
            "Upload a PDF and click "
            "'Process PDF' to begin."
        )


# ============================================================
# API KEY CHECK
# ============================================================

client = get_groq_client()

if client is None:

    st.warning(
        "⚠️ Groq API key is not configured."
    )

    st.info(
        "For Streamlit Cloud, add GROQ_API_KEY "
        "under Settings → Secrets."
    )

    st.code(
        'GROQ_API_KEY = "your_groq_api_key"',
        language="toml",
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader("📄 Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )

    process_button = st.button(
        "🔍 Process PDF",
        type="primary",
        use_container_width=True,
    )

    if process_button:

        with st.spinner(
            "Extracting and processing the PDF..."
        ):

            try:

                # --------------------------------------------
                # Extract PDF
                # --------------------------------------------

                raw_text, page_count = extract_pdf_text(
                    uploaded_file
                )

                if not raw_text.strip():

                    st.error(
                        "No readable text was found in this PDF."
                    )

                    st.stop()

                # --------------------------------------------
                # Clean text
                # --------------------------------------------

                cleaned_text = clean_text(raw_text)

                # --------------------------------------------
                # Create chunks
                # --------------------------------------------

                chunks = create_chunks(
                    cleaned_text,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                )

                if not chunks:

                    st.error(
                        "Could not create text chunks from the PDF."
                    )

                    st.stop()

                # --------------------------------------------
                # Load embedding model
                # --------------------------------------------

                embedding_model = load_embedding_model()

                # --------------------------------------------
                # Create embeddings
                # --------------------------------------------

                embeddings = create_embeddings(
                    chunks,
                    embedding_model,
                )

                if embeddings.size == 0:

                    st.error(
                        "Could not create embeddings."
                    )

                    st.stop()

                # --------------------------------------------
                # Create FAISS database
                # --------------------------------------------

                faiss_index = create_faiss_index(
                    embeddings
                )

                if faiss_index is None:

                    st.error(
                        "Could not create FAISS index."
                    )

                    st.stop()

                # --------------------------------------------
                # Save to session state
                # --------------------------------------------

                st.session_state.chunks = chunks

                st.session_state.faiss_index = (
                    faiss_index
                )

                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.page_count = (
                    page_count
                )

                st.session_state.processed = True

                st.session_state.messages = []

                st.success(
                    "✅ PDF processed successfully!"
                )

                st.write(
                    f"**Pages:** {page_count}"
                )

                st.write(
                    f"**Text chunks:** {len(chunks)}"
                )

            except Exception as e:

                st.error(
                    "An error occurred while processing the PDF."
                )

                st.exception(e)


# ============================================================
# QUESTION ANSWERING
# ============================================================

if st.session_state.processed:

    st.write("---")

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Ask something about your PDF:",
        placeholder="Example: What is the main topic of this document?",
    )

    ask_button = st.button(
        "🚀 Ask Question",
        type="primary",
        use_container_width=True,
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif client is None:

            st.error(
                "Groq API key is missing. "
                "Please configure GROQ_API_KEY in Streamlit Secrets."
            )

        else:

            embedding_model = load_embedding_model()

            with st.spinner(
                "Searching the document..."
            ):

                try:

                    # ----------------------------------------
                    # Retrieve relevant chunks
                    # ----------------------------------------

                    results = search_faiss(
                        query=question,
                        index=st.session_state.faiss_index,
                        chunks=st.session_state.chunks,
                        model=embedding_model,
                        top_k=top_k,
                    )

                    if not results:

                        st.warning(
                            "No relevant information was found."
                        )

                        st.stop()

                    # ----------------------------------------
                    # Build context
                    # ----------------------------------------

                    context = build_context(
                        results
                    )

                    # ----------------------------------------
                    # Generate answer
                    # ----------------------------------------

                    with st.spinner(
                        "Generating answer..."
                    ):

                        answer = generate_answer(
                            client=client,
                            question=question,
                            context=context,
                        )

                    # ----------------------------------------
                    # Display answer
                    # ----------------------------------------

                    st.markdown(
                        '<div class="answer-box">',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "### 🤖 Answer"
                    )

                    st.write(answer)

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    # ----------------------------------------
                    # Show retrieved passages
                    # ----------------------------------------

                    with st.expander(
                        "🔎 View retrieved passages"
                    ):

                        for i, result in enumerate(
                            results,
                            start=1,
                        ):

                            st.markdown(
                                f"**Passage {i}** "
                                f"(similarity: "
                                f"{result['score']:.3f})"
                            )

                            st.write(
                                result["text"]
                            )

                            st.divider()

                except Exception as e:

                    st.error(
                        "An error occurred while generating the answer."
                    )

                    st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.write("---")

st.caption(
    "Built with Streamlit • FAISS • "
    "Sentence Transformers • Groq"
)
