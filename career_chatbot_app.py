#imported required libraries
import os
import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pypdf import PdfReader

#setup API key and LLM

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=api_key,
    temperature=0,
    model_name="openai/gpt-oss-20b"
)


#Initialize vector DB

client = chromadb.Client()
collection=client.get_or_create_collection("career_knowledge_base")
#Function to Ingest PDF

def ingest_pdf(file):
    reader=PdfReader(file)
    text=""
    for page in reader.pages:
        text+=page.extract_text()+"\n"
    return text
#Streamlit App

st.title("Career Guidance Chatbot")
st.markdown_files=("Get personalized grounded career advice")
#Upload PDFs or text files

uploaded_files=st.file_uploader("upload career resources (PDFs)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        text=ingest_pdf(file)
           #chunking

        splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks=splitter.split_text(text)
    #Add chunks to DB

        collection.add(
                       documents=chunks,
                       ids=[f"{file.name}_{i}" for i in range(len(chunks))])

        st.success(f"{len(uploaded_files)} PDFs ingested successfully!")
        
#RAG process

# --------------------------------------------------
# USER QUERY
# --------------------------------------------------

user_query = st.text_input("Ask your career question:")


# --------------------------------------------------
# GET ADVICE
# --------------------------------------------------

if st.button("Get Advice") and user_query:

    # Get all documents from ChromaDB
    collection_data = collection.get()

    all_documents = collection_data.get("documents", [])

    if not all_documents:
        st.warning("Please upload a career resource PDF first.")
        st.stop()

    # --------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------

    vector_results = collection.query(
        query_texts=[user_query],
        n_results=min(5, len(all_documents))
    )

    vector_docs = vector_results["documents"][0]


    # --------------------------------------------------
    # KEYWORD SEARCH
    # --------------------------------------------------

    keywords = user_query.lower().split()

    keyword_docs = [
        doc
        for doc in all_documents
        if any(keyword in doc.lower() for keyword in keywords)
    ]


    # --------------------------------------------------
    # HYBRID SEARCH
    # --------------------------------------------------

    hybrid_docs = list(
        dict.fromkeys(vector_docs + keyword_docs)
    )

    hybrid_docs = hybrid_docs[:10]


    if not hybrid_docs:
        st.warning("No relevant documents found.")
        st.stop()


    # --------------------------------------------------
    # RERANKING PROMPT
    # --------------------------------------------------

    rerank_prompt = PromptTemplate.from_template(
        """
You are a career document ranking assistant.

User Query:
{user_query}

Documents:
{docs}

Rank the documents from most relevant to least relevant
for answering the user's career question.

Focus on:
- Skills
- Career paths
- Job roles
- Companies
- Education
- Industry requirements

Return the most relevant information first.
"""
    )


    # --------------------------------------------------
    # RERANKING
    # --------------------------------------------------

    rerank_chain = rerank_prompt | llm

    reranked_output = rerank_chain.invoke({
        "user_query": user_query,
        "docs": "\n\n--- DOCUMENT ---\n\n".join(hybrid_docs)
    })


    # --------------------------------------------------
    # TOP CONTEXT
    # --------------------------------------------------

    top_context = [
        line.strip()
        for line in reranked_output.content.split("\n")
        if line.strip()
    ][:3]


    # Fallback if the LLM returns nothing
    if not top_context:
        top_context = hybrid_docs[:3]


    # --------------------------------------------------
    # FINAL PROMPT
    # --------------------------------------------------

    final_prompt = PromptTemplate.from_template(
        """
You are a career guidance AI assistant.

Use the following career resources to answer the
user's question.

Career Resources:
{context}

User Query:
{query}

Provide a personalized career roadmap.

Include:
1. Skills to learn
2. Recommended job roles
3. Relevant companies
4. Steps to improve career readiness
5. Suggested learning progression

Use the provided resources as the main source of information.
Do not invent unsupported information.
"""
    )


    # --------------------------------------------------
    # FINAL RAG CHAIN
    # --------------------------------------------------

    rag_chain = final_prompt | llm

    career_advice = rag_chain.invoke({
        "context": "\n\n".join(top_context),
        "query": user_query
    })


    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    st.subheader("Top Retrieved Context")

    for doc in top_context:
        st.write("•", doc)


    st.subheader("Personalized Career Advice")

    st.write(career_advice.content)