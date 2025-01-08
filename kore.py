import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables

# Configuration for Gemini Embeddings
class GeminiEmbeddings:
    def __init__(self, model_name="models/text-embedding-004", api_key="AIzaSyB3YqnrS9jBzDA8EuASkd6gDNI8UQkTQJw"):
        if api_key is None:
            raise ValueError("API key is required for Gemini embeddings.")
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def embed_query(self, text):
        result = genai.embed_content(model=self.model_name, content=text)
        return result['embedding']

    def embed_documents(self, texts):
        embeddings = [genai.embed_content(model=self.model_name, content=text)['embedding'] for text in texts]
        return embeddings

# Load the PDF
@st.cache_resource
def load_data(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

# Initialize components
@st.cache_resource
def initialize_chain():
    # Load documents and split them
    docs = load_data("/Users/jaydaksharora/Downloads/SRB-2023-24_compressed.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Initialize embeddings and vector store
    embeddings = GeminiEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings,persist_directory="./chroma_db")
    
    # Set up retriever and chain
    retriever = vectorstore.as_retriever()
    llm = ChatGroq(model="llama-3.3-70b-versatile",api="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")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Answer in detail and concise."
        "\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

rag_chain = initialize_chain()

# Streamlit UI
st.title("Student Resource Book Chatbot")
st.write("Ask me any question about the rules or guidelines in the Student Resource Book!")

query = st.text_input("Enter your question:", "")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            results = rag_chain.invoke({"input": query})
            answer = results.get('answer', "Sorry, I couldn't find an answer.")
            st.markdown(f"### Answer:\n{answer}")
    else:
        st.warning("Please enter a question!")
