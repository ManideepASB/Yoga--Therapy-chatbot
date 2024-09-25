#!/usr/bin/env python
# coding: utf-8

# In[2]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS 
import torch

# Define paths
DATA_PATH = "C:\\Mani C drive\\CPS quarter 5\\6980 capstone\\DATA articles"
DB_FAISS_PATH = "C:\\Mani C drive\\CPS quarter 5\\6980 capstone\\Yoga chat bit V1\\vectorstores\\db_faiss"

# Function to create vector DB
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Set the device to MPS if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    create_vector_db()




# In[9]:


import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import chainlit as cl
import torch
import streamlit as st
# Define the path to your FAISS database
DB_FAISS_PATH = 'C:\\Users\\manid\\Downloads\\Finance\\vectorstores\\db_faiss'

# Custom prompt template
custom_prompt_template = """You are a sophisticated and diligent YOGA assistant, programmed to deliver accurate, respectful, and constructive responses. Your primary function is to provide information and insights related to finance and economics, with a focus on financial knowledge.

You are designed to adhere strictly to ethical guidelines, ensuring all your responses are free from harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. You maintain a socially unbiased stance and promote positivity in all interactions.

If you encounter a question that is unclear, nonsensical, or factually inconsistent, you are to clarify the confusion respectfully and guide the inquirer towards a coherent understanding, instead of providing incorrect or misleading information. In instances where you lack sufficient data or knowledge to respond accurately, you are to acknowledge the limitation openly, avoiding speculation or the dissemination of falsehoods.

Your ultimate aim is to educate, inform, and assist users in understanding yOGA concepts, and yoga benifits, empowering them with reliable information to make informed decisions.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "sk-proj-PhcrSSXTBn7IuOoTaYjoT3BlbkFJV3K2xRKR6O4DHc7n69vg"

# Initialize the ChatOpenAI model
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo', max_tokens=512, temperature=0.5
)

# Function to create the Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Function to initialize the QA bot
def qa_bot():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = chat
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Function to get the final result for a query and provide a generalized output
def final_result(query):
    qa = qa_bot()
    response = qa({'query': query})
    
    # Extracting and summarizing the key points from the documents
    summarized_response = response['result']
    return summarized_response

def main():
    st.title("Yoga Assistant Bot")
    st.write("Hi, Welcome to Yoga Assistant Bot. What is your query?")
    
    query = st.text_input("Enter your query here:")

    if st.button("Submit"):
        if query:
            response = final_result(query)
            st.write(response)
        else:
            st.write("Please enter a query.")
            
if __name__ == "__main__":
    main()


# In[ ]:


# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Yoga Assistant Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    query = message.content
    response = final_result(query)
    
    await cl.Message(content=response).send()


# In[ ]:




