# Import libraries
from openai import OpenAI
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
import time

# Set the directory where you want to save the uploaded files
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to the specified directory
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Load the PDF and create embeddings only once
        if "vectordb_retriever" not in st.session_state:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            # Split the document into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            print(f"Your {len(documents)} document(s) have been split into {len(splits)} chunks")
            # Create a vector store from the document chunks
            vectordb = Chroma.from_documents(splits, embeddings)
            st.session_state.vectordb_retriever = vectordb.as_retriever(search_kwargs={"k": 2}, search_type="mmr")

            # Initialize the BM25 retriever and ensemble retriever
            bm25_retriever = BM25Retriever.from_documents(splits)
            bm25_retriever.k = 2
            st.session_state.ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, st.session_state.vectordb_retriever],
                weights=[0.5, 0.5]
            )

        st.sidebar.success(f"PDF '{uploaded_file.name}' uploaded and processed successfully! You can start chat.")
    except Exception as e:
        st.sidebar.error(f"An error occurred while processing the PDF: {e}")

# Initialize the language model only if ensemble_retriever is in session_state
if "ensemble_retriever" in st.session_state:
    llm = ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],  # Replace with your actual API key
        temperature=0.17,
        # model="gpt-4-0613",
        model = "gpt-4o",
        # model = "gpt-3.5-turbo-0125"
    )

    # Initialize MultiQueryRetriever with the ensemble retriever
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=st.session_state.ensemble_retriever, llm=llm)

    # Define the prompt template
    prompt_template = """Use the following pieces of context to provide a detailed 1000-word answer to the question.
    Ensure that your response is thorough and well-structured. If the context does not fully answer the question,
    supplement your answer with your own knowledge. Conclude your response by mentioning the source heading from the provided context.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Streamed response emulator
    def response_generator(res):
        response = res
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    # Initialize an empty list to store the history
    history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question related to the uploaded PDF..."):
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            # Retrieve relevant documents
            ensemble_docs = retriever_from_llm.get_relevant_documents(query=prompt)
            # Combine historical context with current ensemble_docs
            all_context = history[-3:] + ensemble_docs
            resp = llm.predict(PROMPT.format(context=all_context, question=prompt))
            
            # Save the question and answer in history
            history.append({
                "question": prompt,
                "answer": resp
            })

            # Display the assistant's response
            with st.chat_message("assistant"):
                st.markdown(resp)  # Use markdown to display the response

            # Append the assistant's response to session state
            st.session_state.messages.append({"role": "assistant", "content": resp})
        except Exception as e:
            # Handle any exceptions that occur during processing
            st.chat_message("assistant").markdown(f"An error occurred: {e}")
else:
    st.sidebar.info("Please upload a PDF to start the chat.")