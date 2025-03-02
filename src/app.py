import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

load_dotenv()

from langchain.llms import HuggingFaceHub
import os





if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    

# app config 
st.set_page_config(page_title="Streaming bot", page_icon="🤖")
st.title("Streaming bot")


#get response 
def get_response(query,chat_history):
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
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
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "chat_history": chat_history,
        "user_question": query,
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

     


