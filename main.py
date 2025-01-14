# Get all required tools
import os
import requests
import tempfile
import numpy as np
import streamlit as st
from PyPDF2 import  PdfReader
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.agents import create_csv_agent
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# Streamlit app setting
st.set_page_config(page_title="Chatbot", layout="wide")

# Load environment variables
load_dotenv()

# Code for PDF chatbot
def pdf_chatbot():
    try:
        st.header("Chat with PDF Files")

        pdf_uploader = st.file_uploader("Choose a PDF file", type="pdf")

        if pdf_uploader is not None:
            pdf_file = PdfReader(pdf_uploader)
            text_file = " "
            for page in pdf_file.pages:
                text_file += page.extract_text()
 
            
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text_file)
           

            # Use OpenAI embeddings
            embeddings = OpenAIEmbeddings()
           
            
            vector_db = FAISS.from_texts(chunks, embeddings)
           

            user_query = st.text_input("Enter your question here")

            if user_query:
                documents = vector_db.similarity_search(user_query)
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                answer_output = chain.run(input_documents=documents, question=user_query)
                st.write(answer_output)
                
    except Exception as e:
        st.error(f"An error occurred in the PDF chatbot: {e}")


# Code for CSV Chatbot
def csv_chatbot():

    st.header("Chat with CSV Files")

    csv_uploader = st.file_uploader("Upload your CSV document", type='csv')
        
    if csv_uploader is not None:
        user_query = st.chat_input("Enter your question here")
    
        llm = OpenAI(temperature=0)

        # Create a csv agent
        csv_agent = create_csv_agent(llm, csv_uploader, verbose=True)
        
        if user_query:
            answer_output = csv_agent.run(user_query)
            st.write(answer_output)
   

# Code for Youtube Chatbot
def youtube_chatbot():    
    
    st.header("Chat with YouTube Videos")

    video_url = st.text_input("Enter YouTube video URL")

    if video_url:
        try:
            loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
            document = loader.load()

            text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len
        )
            
            text_chunks = text_splitter.split_documents(document)
            total_tokens = sum(len(chunk.page_content.split()) for chunk in text_chunks)

            if total_tokens > 4097:
                st.write("Sorry, this application cannot process longer videos at the moment. Please upload a shorter video.")
            else:
                embeddings = OpenAIEmbeddings()
                vector_db = Chroma.from_documents(text_chunks, embeddings)

                llm = OpenAI(temperature=0)

                user_input = st.chat_input("Enter your question here")

                if user_input:
                    chain = load_qa_chain(llm, chain_type="stuff")
                    answer_output = chain.run(input_documents=text_chunks, question=user_input)
                    st.write(answer_output)
                
        except ValueError as e:
            st.error(f"Error loading YouTube video: {e}")


# Code for website chatbot
def website_chatbot():

    st.header("Chat with Website")

    website_url = st.text_input("Enter your website URL")

    if website_url:

        try:
            loader = WebBaseLoader(website_url)
            website_text = loader.load()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=4000,
                chunk_overlap=200,
                length_function=len
            )
            
            text_chunks = text_splitter.split_documents(website_text)
            
            embeddings = OpenAIEmbeddings()
            
            vector_db = Chroma.from_documents(text_chunks, embeddings)

            llm = OpenAI(temperature=0.5)
             
            user_input = st.chat_input("Enter your question here")

            if user_input:
                for chunk in text_chunks:
                    chain = load_qa_chain(llm, chain_type="stuff")
                    answer_output = chain.run(input_documents=[chunk], question=user_input)
                st.write(answer_output)

        except Exception as e:
            print(f"Oops, an error occurred. Please try again.: {e}")

# Code for word chatbot
def word_chatbot():
    try:
        st.header("Chat with Word Documents")
        
        # Word file uploader
        word_uploader = st.file_uploader('Upload your word document ', type=".docx")

        if word_uploader is not None:
            # Save the uploaded file to disk
            with open("uploaded_file.docx", "wb") as f:
                f.write(word_uploader.read())

            loader = UnstructuredWordDocumentLoader("uploaded_file.docx")
            word_data = loader.load()

            if word_data: 
                full_text = ""
                for doc in word_data:
                    full_text += doc.page_content

                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                # Splitting text into smaller chunks
                chunks = text_splitter.split_text(full_text)
                
                embeddings = OpenAIEmbeddings()
                vector_db = FAISS.from_texts(chunks, embeddings)

                user_query = st.text_input("Enter your question here")

                if user_query:
                    documents = vector_db.similarity_search(user_query)

                    llm = OpenAI(temperature=0)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    answer_output = chain.run(input_documents=documents, question=user_query)
                    st.write(answer_output)
            else:
                st.error("Error: Unable to load Word document. Please make sure you have uploaded a valid .docx file.")
    except Exception as e:
        st.error(f"An error occurred in the Word chatbot: {e}")


# Code for excel chatbot
def excel_chatbot():
    try:
        st.header("Chat with Excel Documents")

        # Excel file uploader
        excel_uploader = st.file_uploader('Upload your Excel document', type=[".xlsx", ".xls"])
        
        if excel_uploader is not None:
            # Save the uploaded file to disk
            with open("uploaded_file.xlsx", "wb") as f:
                f.write(excel_uploader.read())

            loader = UnstructuredExcelLoader("uploaded_file.xlsx")
            excel_data = loader.load()
             
            # Check if data is not empty
            if excel_data: 
                full_text = ""
                for sheet in excel_data:
                    full_text += sheet.page_content
                    
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(full_text)

                embeddings = OpenAIEmbeddings()

                vector_db = FAISS.from_texts(chunks, embeddings)
                
                # Ask for user input
                user_query = st.text_input("Enter your question here")

                if user_query:
                    documents = vector_db.similarity_search(user_query)
                    llm = OpenAI(temperature=0)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    answer_output = chain.run(input_documents=documents, question=user_query)
                    st.write(answer_output)
            else:
                st.error("Error: Unable to load Excel document. Please make sure you have uploaded a valid .xlsx or .xls file.")
    except Exception as e:
        st.error(f"Oops, an error occurred. Please try again.: {e}")

# Code for PowerPoint Chatbot
def power_point():
    try:
        st.header("Chat with PowerPoint Documents")

        ppt_uploader = st.file_uploader('Upload your PowerPoint file', type=[".pptx", ".ppt"])

        if ppt_uploader is not None:
            # Save the uploaded file to disk
            with open("uploaded_file.pptx", "wb") as f:
                f.write(ppt_uploader.read())

         
            loader = UnstructuredPowerPointLoader("uploaded_file.pptx")
            ppt_data = loader.load()

            if ppt_data:  
                full_text = ""
                for slide in ppt_data:
                    full_text += slide.page_content


                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
               
                chunks = text_splitter.split_text(full_text)
    
                embeddings = OpenAIEmbeddings()

                vector_db = FAISS.from_texts(chunks, embeddings)
                
                user_query = st.text_input("Enter your question here")

                if user_query:
                    documents = vector_db.similarity_search(user_query)
                    llm = OpenAI(temperature=0)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    answer_output = chain.run(input_documents=documents, question=user_query)
                    st.write(answer_output)
            else:
                st.error("Error: Unable to load PowerPoint document. Please make sure you have uploaded a valid .pptx or .ppt file.")
    except Exception as e:
        st.error(f"Oops, an error occurred. Please try again.: {e}")


# Create Navigation bar for the App
with st.sidebar:
    st.markdown('<h1 style="color:#a647d6">Q&A with your files</h1>', unsafe_allow_html=True)

nav = st.sidebar.radio("Select Chatbot", ["PDF Chatbot", "CSV Chatbot", "YouTube Chatbot", "Website Chatbot", "Word Chatbot", "Excel Chatbot", "PowerPoint Chatbot"])

# Display the selected chatbot
if nav == "PDF Chatbot":
    pdf_chatbot()
elif nav == "CSV Chatbot":
    csv_chatbot()
elif nav == "YouTube Chatbot":
    youtube_chatbot()
elif nav == "Website Chatbot":
    website_chatbot()
elif nav == "Word Chatbot":
    word_chatbot()
elif nav == "Excel Chatbot":
    excel_chatbot()
elif nav == "PowerPoint Chatbot":
    power_point()


