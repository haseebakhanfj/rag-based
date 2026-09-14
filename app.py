import os
import re
from typing import List, Dict

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Chunking settings
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the open-source embedding model once per Streamlit session."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str):
    """Create the Groq client once for the supplied API key."""
    return Groq(api_key=api_key)


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from every page of a PDF."""
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if text:
            pages.append(f"[Page {page_number}]\n{text}")

    return "\n\n".join(pages)


# -----------------------------
# Text chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Word-based chunking is simple, stable and avoids cutting text in the
    middle of a word. The embedding model performs tokenization internally.
    """
    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start += step

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def create_faiss_index(chunks: List[str], embedding_model):
    """Create normalized embeddings and a FAISS inner-product index."""
    embeddings = embedding_model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # With normalized vectors, inner product is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Retrieval
# -----------------------------
def retrieve_chunks(question: str, index, chunks: List[str],
                     embedding_model, top_k: int = TOP_K):
    """Retrieve the most semantically relevant chunks."""
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "text": chunks[int(idx)],
                "score": float(score),
                "index": int(idx),
            }
        )

    return results


# -----------------------------
# Generation
# -----------------------------
def generate_answer(question: str, retrieved_chunks: List[Dict], client):
    """Generate a grounded answer using only retrieved PDF context."""
    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"--- Retrieved Context {i} ---\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a document question-answering assistant.

Answer the user's question using ONLY the retrieved context from the uploaded
PDF.

Rules:
1. Do not invent facts that are not supported by the context.
2. If the answer is not available in the context, clearly say:
   "I could not find that information in the uploaded document."
3. Give a concise but useful answer.
4. When possible, mention the PDF page number shown in the retrieved context.
5. Do not use outside knowledge to fill missing information.
"""

    user_prompt = f"""Retrieved PDF context:

{context}

User question:
{question}

Answer based only on the retrieved context."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return response.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.caption(
    "Upload a PDF → extract text → chunk → open-source embeddings → "
    "FAISS retrieval → Groq Llama answer"
)

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=TOP_K,
        help="Number of relevant chunks sent to the language model.",
    )

    st.markdown("---")
    st.write("**Embedding model**")
    st.code(EMBEDDING_MODEL_NAME)

    st.write("**LLM**")
    st.code(GROQ_MODEL)

    st.markdown("---")
    st.info(
        "The embedding model and FAISS index run in the app. "
        "Groq is used only for final answer generation."
    )


# API key: Streamlit Cloud secret first, environment variable second.
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

if not groq_api_key:
    st.warning(
        "GROQ_API_KEY is not configured. Add it to Streamlit Cloud Secrets "
        "or your local environment before asking questions."
    )

uploaded_file = st.file_uploader(
    "📄 Upload a PDF document",
    type=["pdf"],
    help="For the first test, use a text-based PDF rather than a scanned image-only PDF.",
)

if uploaded_file is not None:
    file_signature = (
        uploaded_file.name,
        uploaded_file.size,
        getattr(uploaded_file, "file_id", None),
    )

    # Rebuild the index only when a different PDF is uploaded.
    if st.session_state.get("file_signature") != file_signature:
        st.session_state.clear()
        st.session_state["file_signature"] = file_signature

        with st.spinner("Reading and indexing your PDF..."):
            extracted_text = extract_pdf_text(uploaded_file)

            if not extracted_text.strip():
                st.error(
                    "No selectable text was extracted from this PDF. "
                    "It may be a scanned/image-only PDF. Use a text-based PDF "
                    "or add OCR in a future version."
                )
                st.stop()

            chunks = chunk_text(extracted_text)

            if not chunks:
                st.error("The PDF did not produce usable text chunks.")
                st.stop()

            embedding_model = load_embedding_model()
            index, embeddings = create_faiss_index(chunks, embedding_model)

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["embedding_count"] = len(embeddings)
            st.session_state["extracted_chars"] = len(extracted_text)

    chunks = st.session_state["chunks"]
    index = st.session_state["index"]

    col1, col2, col3 = st.columns(3)
    col1.metric("PDF", uploaded_file.name)
    col2.metric("Chunks", len(chunks))
    col3.metric("Characters extracted", f"{st.session_state['extracted_chars']:,}")

    st.success("✅ PDF indexed successfully. You can now ask questions.")

    question = st.text_input(
        "💬 Ask a question about your PDF",
        placeholder="Example: What are the main objectives described in the document?",
    )

    ask = st.button("🔎 Ask", type="primary", use_container_width=True)

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not groq_api_key:
            st.error("Please configure GROQ_API_KEY first.")
        else:
            try:
                embedding_model = load_embedding_model()

                with st.spinner("Searching the document..."):
                    retrieved = retrieve_chunks(
                        question,
                        index,
                        chunks,
                        embedding_model,
                        top_k=top_k,
                    )

                if not retrieved:
                    st.warning("No relevant context was found.")
                else:
                    client = get_groq_client(groq_api_key)

                    with st.spinner("Generating answer..."):
                        answer = generate_answer(
                            question,
                            retrieved,
                            client,
                        )

                    st.subheader("🤖 Answer")
                    st.write(answer)

                    with st.expander("📌 Retrieved context"):
                        for i, item in enumerate(retrieved, start=1):
                            st.markdown(
                                f"**Chunk {i} — similarity: {item['score']:.3f}**"
                            )
                            st.text(item["text"])

else:
    st.info("Upload a PDF to build your local FAISS knowledge base.")
