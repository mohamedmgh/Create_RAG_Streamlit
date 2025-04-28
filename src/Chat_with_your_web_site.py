import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.llms import HuggingFaceHub
import os




# Load environment variables
load_dotenv()

# Set Hugging Face API token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "your APi key"

# App configuration
st.set_page_config(page_title="RAG Chatbot Use Streamlit", page_icon="🤖")
st.title("RAG Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db" not in st.session_state:
    st.session_state.db = None

# File uploader for PDF

st.header("Settings")
website_url = st.text_input("Website URL")

# Initialize the vector database when a file is uploaded
if website_url and st.session_state.db is None:
    with st.sidebar:
        with st.spinner("Processing document..."):
            
            # Load the document
            loader = WebBaseLoader(website_url )
            data = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=0,
            )
            chunks = text_splitter.split_documents(data)
            
            # Create embeddings
            embedding_model = FastEmbedEmbeddings(model_name="thenlper/gte-large")
            
            # Initialize vector store
            st.session_state.db = Chroma.from_documents(
                chunks, 
                embedding_model, 
                persist_directory="./chroma_db"
            )
            
            

# Function to get RAG response
def get_rag_response(query, chat_history):
    # Check if database is initialized
    if st.session_state.db is None:
        return "Please upload a document first."
    
    # Create retriever
    retriever = st.session_state.db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 4}
    )
    
    # Initialize LLM
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={
            "temperature": 0.1,
            "max_new_tokens": 512,
            "return_full_text": False,
            "repetition_penalty": 1.1,
            "top_p": 0.9
        }
    )
    
    # Create RAG prompt template
    rag_template = """
    <s>[INST] You are an AI Assistant that follows instructions extremely well. Answer based on the context provided.
    
    Chat history: {chat_history}
    
    Context: {context}
    
    If the answer cannot be found in the context, please say "I don't have that information in the document." [/INST]
    
    User question: {query} </s>
    """
    
    # Format chat history for the prompt
    formatted_chat_history = "\n".join([
        f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
        for msg in chat_history
    ])
    
    # Create prompt and chain
    prompt = ChatPromptTemplate.from_template(rag_template)
    output_parser = StrOutputParser()
    
    chain = (
        {
            "context": retriever, 
            "query": RunnablePassthrough(),
            "chat_history": lambda x: formatted_chat_history
        } 
        | prompt 
        | llm 
        | output_parser
    )
    
    return chain.invoke(query)

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.markdown(message.content)
    else:
        with st.chat_message("ai"):
            st.markdown(message.content)

# Get user input
user_query = st.chat_input("Ask a question about your document...")

# Process user input
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))
    
    with st.chat_message("human"):
        st.markdown(user_query)
    
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            ai_response = get_rag_response(user_query, st.session_state.chat_history)
            st.markdown(ai_response)
    
    st.session_state.chat_history.append(AIMessage(ai_response))                
