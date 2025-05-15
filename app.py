import streamlit as st
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from typing import List
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from pinecone import Pinecone
from langchain_pinecone import Pinecone as LangchainPinecone

# === Streamlit page setup ===
st.set_page_config(page_title="AlturaTech policy Assistant", layout="wide")
st.title("📄 AlturaTech policy Assistant")
st.markdown("Ask questions based on internal policy documents.")

# === Custom Embeddings Wrapper ===
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(text, convert_to_numpy=True).tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

# === Setup ===
@st.cache_resource
def init_models():
    # Embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_model = SentenceTransformerEmbeddings(model=model)

    # Pinecone & Vectorstore
    pc = Pinecone(api_key="pcsk_4TdFoY_7YMQNwoFz26iF5dc59ZVLuPAzxeqsFpuyhGWk6FyNhqApFyQWQYn9LoqrjhEFcW", environment="us-east-1")
    index = pc.Index("policychatbot")
    vectorstore = LangchainPinecone(index, embedding=embedding_model, text_key="text")
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        groq_api_key="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"
    )

    return retriever, llm

retriever, llm = init_models()

# === Prompts ===
router_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a routing agent. Based on the user question below, identify the type of source that would best answer it: "pdf", "web", or "general".
    Question: {question}
    Answer (just one word - pdf, web, general):
    """
)
router_chain = LLMChain(llm=llm, prompt=router_prompt)

reasoning_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
    Analyze the following context and resolve any conflicts. Provide a coherent and complete answer.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
)
reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt)

compliance_prompt = PromptTemplate(
    input_variables=["answer"],
    template="""
    Review the answer for any compliance or policy-sensitive content.
    - If risky content is found, redact it.
    - Optionally, add a footnote with a compliance warning.

    Answer:
    {answer}

    Compliant Answer:
    """
)
compliance_chain = LLMChain(llm=llm, prompt=compliance_prompt)

# === Core Logic ===
def route_fn(query):
    return router_chain.invoke({"question": query})["text"].strip().lower()

def filtered_retrieval(query, route):
    if route in ["pdf", "web"]:
        return retriever.get_relevant_documents(query, filter={"source": route})
    return retriever.get_relevant_documents(query)

def format_docs_with_metadata(docs):
    return "\n\n".join(f"{doc.page_content}\n(Source: {doc.metadata.get('filename', doc.metadata.get('url', 'N/A'))})" for doc in docs)

def group_chunks_by_document(docs):
    grouped_docs = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("filename") or doc.metadata.get("url") or "Unknown"
        grouped_docs[source].append((doc.metadata.get("chunk_id", "N/A"), doc.page_content.strip()))
    return grouped_docs

def run_query_with_sources(query):
    route = route_fn(query)
    docs = filtered_retrieval(query, route)
    context = format_docs_with_metadata(docs)
    answer = reasoning_chain.invoke({"context": context, "query": query})["text"]
    compliant = compliance_chain.invoke({"answer": answer})["text"]
    return compliant, docs

# === Streamlit UI ===




query = st.text_input("Enter your query:")
if query:
    with st.spinner("Thinking..."):
        final_answer, retrieved_docs = run_query_with_sources(query)

    st.markdown("### 🧠 Final Answer")
    st.write(final_answer)

    st.markdown("### 📚 Retrieved Chunks by Document")
    grouped = group_chunks_by_document(retrieved_docs)
    for source, chunks in grouped.items():
        st.markdown(f"**Source: {source}**")
        for chunk_id, content in chunks:
            with st.expander(f"🔹 Chunk ID: {chunk_id}"):
                st.write(content)
