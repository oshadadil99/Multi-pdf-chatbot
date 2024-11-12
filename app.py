import streamlit as st
import os

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub



def get_pdf_text(pdf_file):
    text = ""
    for pdf in pdf_file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()  
    return text  

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_converstation_chain(vectorstore):
    
    hf_api_token = os.getenv("HUGGINGFACE_API_KEY")
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        
    llm = HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token=hf_api_token)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    converstation_chain = ConversationalRetrievalChain.from_llml(
        llm =llm,
        retriever = vectorstore.as_retriever(),
        memory = memory,
    )
    return converstation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask from multi-PDFs",page_icon=":books:")

    if "converstation" not in st.session_state:
        st.session_state.converstation = None

    st.header("Ask from multi-PDFs :books:")
    st.text_input("Ask anything from your docs:")

    with st.sidebar:
        st.subheader("Select your PDF")
        pdf_file = st.file_uploader("Choose a PDF file", accept_multiple_files=True)
        if st.button("Start"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_file)
                
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.converstation = get_converstation_chain(vectorstore)



if __name__ == "__main__":
    main()