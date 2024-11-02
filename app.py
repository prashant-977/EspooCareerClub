import streamlit as st
import pandas as pd
import os
import base64
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, ListIndex
from dotenv import load_dotenv

def save_uploaded_file(uploaded_file):
    with open(os.path.join("data", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.success("Saved file:{} to directory".format(uploaded_file.name))
	
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    #Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    #Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)
	
def semantic_search(query):
    # Load documents from the 'data' directory
    documents = SimpleDirectoryReader('data').load_data()

    # Create a vector store index from the documents
    index = GPTVectorStoreIndex.from_documents(documents)

    # Create a query engine from the index
    query_engine = index.as_query_engine()

    # Use the query engine to perform the query
    response = query_engine.query(query)

    return response
  
def summarize(file):
    # Load documents from the 'data' directory
    documents = SimpleDirectoryReader('data').load_data()

    # Create a list index from the documents
    index = ListIndex.from_documents(documents)

    # Create a query engine from the index
    query_engine = index.as_query_engine()

    # Use the query engine to query/summarize the file
    response = query_engine.query(file)

    return response
  
st.set_page_config(layout="wide")
st.title("Gen AI Powered Jobseekers' Guide")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_csv is not None:
	df = pd.read_csv(uploaded_csv)


if uploaded_pdf is not None:
    # Set up columns with specified width adjustments
    col1, col2 = st.columns([1,3])
    
    # Column 1: Display the uploaded PDF and CSV file if available
    with col1:
        input_file = save_uploaded_file(uploaded_pdf)
        pdf_file = "data/" + uploaded_pdf.name
        st.write("Uploaded PDF:", uploaded_pdf.name)
        
        # Display CSV file if uploaded and save it
        if uploaded_csv is not None:
            csv_file = save_uploaded_file(uploaded_csv)
            st.write("Uploaded CSV:", uploaded_csv.name)
    
    # Column 2: Search area for profession-specific insights
    with col2:
        st.success("Search Area")
        query_search = st.text_area("Enter your designation")
        
        # Handle case where multiple questions are provided in a CSV file
        if st.checkbox("Search"):
            st.info("Your query: " + query_search)
            if uploaded_csv is not None:
                # Extract and display responses for each question from the CSV file
                questions = df["insight"].tolist()
                responses = [semantic_search(q) for q in questions]
                for idx, (q, r) in enumerate(zip(questions, responses), 1):
                    st.write(f"Question {idx}: {q.split('(')[0]}\nAnswer: {r}")
            else:
                # Display response for single entered question
                result = semantic_search(query_search)
                st.write(result)
    


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")	
os.environ["OPENAI_API_KEY"] = openai_api_key