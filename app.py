import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Generate FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Build conversational chain for answering questions
def get_conversational_chain():
    prompt_template = """
    You are an AI assistant that extracts precise details from the provided context. 
    Answer the question strictly based on the available context and provide concise and accurate information.

    Ensure the answer is in a complete sentence.
    If the answer is unavailable in the context, respond with: "The answer is not available in the provided context."

    Maintain clarity and accuracy without adding extra information beyond the provided context.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input and generate response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="💬", layout="wide")

    # Header section
    st.title("💬 Chat with PDF using Gemini 🤖")
    
    # Sidebar for PDF upload
    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your PDF files here",
            accept_multiple_files=True,
        )
        if st.button("📥 Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing your files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Processing complete! You can now ask questions.")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

    # User input section
    user_question = st.text_input("💡 Ask a question based on the uploaded PDF content:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()