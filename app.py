import flask as Flask
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import requests
import json
import os
import sys


import os
import pandas as pd
import numpy as np
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from pinecone import Pinecone
# from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pathlib
import textwrap
from groq import Groq
from IPython.display import display, Markdown
import os
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

import pineconeRetrive as pr



# configuration
GROQ_API_KEY = "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"
PINECONE_API_KEY = "1307be7e-dd9f-41e4-af3d-70166f43ba98"
index_name = "innovent1"


# groq congif
ai_role = "You are a AI assistant that helps people resolve their error in industrial machinary the user has give the error or problem they are facing and the relevent solution is given convert this entire query into a step by step guide for solving the error"

groq_chat = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192"
    )

print("Hello! I'm your here to solve any error you face while handling a Machine. I can help answer your questions and solve errors!")

system_prompt = 'You are a friendly conversational chatbot'
conversational_memory_length = 5

memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)




# embedding model
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device="cpu")


# pinecone config
pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        pod_type='p1'
    )

index = pc.Index(index_name)



# query retrival









app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', botStatus="0",user_message="", response="")


@app.route('/response', methods=['POST'])
def response():
    # Parse JSON data from the request
    data = request.get_json()
    user_message = data.get('user_message', '')

    if user_message.lower() == "end":
        return jsonify({"response": "Chat ended."})

    # Process the user message with the chatbot logic
    bot_response = pr.generateResponse(user_message)

    # Return the bot's response as JSON
    return jsonify({"response": bot_response})

@app.route('/Toolinfo')
def Toolinfo():
    return render_template('ToolInfo.html')





if __name__ == '__main__':
    app.run(debug=True)