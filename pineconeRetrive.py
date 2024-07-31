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


GROQ_API_KEY = "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"
PINECONE_API_KEY = "1307be7e-dd9f-41e4-af3d-70166f43ba98"
index_name = "innovent1"

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device="cpu")


pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        pod_type='p1'
    )

index = pc.Index(index_name)


def retrieve_query(query_embedding, k=1):
    matching_results = index.query(
        vector=query_embedding, top_k=k, include_metadata=True)
    return matching_results


def generateResponse(query):
    ai_role = "You are a AI assistant that helps people resolve their error in industrial machinary the user has give the error or problem they are facing and the relevent solution is given convert this entire query into a step by step guide for solving the error"

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

    print(
        "Hello! I'm your here to solve any error you face while handling a Machine. I can help answer your questions and solve errors!")

    system_prompt = 'You are a friendly conversational chatbot'
    conversational_memory_length = 5

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history",
                                            return_messages=True)
    user_question = query
    query_embedding = embedding_model.encode(user_question)
    query_embedding = query_embedding.tolist() if hasattr(
        query_embedding, 'tolist') else query_embedding
    results = retrieve_query(query_embedding)
    print(results)

    text = results['matches'][0]['metadata']['text']
    cleaned_text = text.replace('\t', ' ').replace('\r', ' ').replace('\xa0', ' ').replace('§\uf0a7',
                                                                                           '').replace('\xad‐',
                                                                                                       '-')
    cleaned_text = ' '.join(cleaned_text.split())
    user_question = ai_role + user_question + cleaned_text
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=system_prompt
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),
        ]
    )
    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    response = conversation.predict(human_input=user_question)
    print("Chatbot:", response)
    return response
