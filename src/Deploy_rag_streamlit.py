import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain_core.runnables import RunnablePassthrough


# import the question-answering chain and Huggingface Hub LLM
from langchain.llms import HuggingFaceHub
import os



 
# import PDF reader
from langchain_community.document_loaders.pdf import PyPDFLoader

# load the document
# Source credits: https://ncert.nic.in/textbook/pdf/lekl101.pdf
loader = PyPDFLoader('C:/Users/PC/Desktop/VS code File/chatbot_strealit/src/guide.pdf')
data = loader.load()


 
# import text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
chunk_size = 512,
chunk_overlap = 0,
)

chunks = text_splitter.split_documents(data)



from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

model_name = "thenlper/gte-large"
embedding_model = FastEmbedEmbeddings(model_name="thenlper/gte-large")


 
from langchain.vectorstores import Chroma

# initialize the vector store (save to disk)
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db3")

 
# Let's define the query, since we are going to use it multiple times.
query = "Give me the main comptetence of mohamed"



# retrieve from vector db (load from disk) with query
retrieved_docs = db.similarity_search(query)
print(retrieved_docs[0].page_content)


 
# initialize the retriever
retriever = db.as_retriever(
search_type="mmr", #similarity
search_kwargs={'k': 4}
)




import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

load_dotenv()

#from langchain.llms import HuggingFaceHub
import os


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# app config
st.set_page_config(page_title="Streaming bot", page_icon="🤖")
st.title("Streaming bot")


#get response 
def get_response(query,chat_history):
    template = """
    <s>[INST]
    You are an AI Assistant that follows instructions extremely well.
    Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT
    [/INST]
    CONTEXT: {context}
    </s>
    [INST]
    Chat history: {chat_history}
    User question: {user_question}
    [/INST]

    
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={
    "temperature":0.1,
    "max_new_tokens":512,
    "return_full_text":False,
    "repetition_penalty":1.1,
    "top_p":0.9
    })
        
    chain = ( {"context": retriever, "user_question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    
    return chain.invoke({
        "chat_history": chat_history,
        "user_question": query,
        "context": retriever,
    })



#Conversation 
for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message("human"):
            st.markdown(message.content)
    else: 
        with st.chat_message("AI"):
            st.markdown(message.content)

#user input 

user_query = st.chat_input("Your messsage")

if user_query is not None and user_query!="":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        ai_response= get_response(user_query,st.session_state.chat_history)
        st.markdown(ai_response)  
    st.session_state.chat_history.append(AIMessage(ai_response))   