
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI


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


 
# define the llm
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
model_kwargs={
"temperature":0.1,
"max_new_tokens":512,
"return_full_text":False,
"repetition_penalty":1.1,
"top_p":0.9
})


 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """
<s>[INST]
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT
[/INST]
CONTEXT: {context}
</s>
[INST]
{query}
[/INST]
"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain = (
{"context": retriever, "query": RunnablePassthrough()}
| prompt
| llm
| output_parser
)

 
response = chain.invoke(query)
response


response3 = chain.invoke("donne moi c'est quoi le resme in 3 line")
print(response3)