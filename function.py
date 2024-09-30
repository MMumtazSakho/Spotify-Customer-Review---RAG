from openai import OpenAI
import os
import chromadb
from chromadb.config import Settings
import numpy as np
from langchain_chroma import Chroma
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain_mistralai import MistralAIEmbeddings
import streamlit as st
import numpy as np
import sqlite3
import pandas as pd


api_key = os.environ["OPENAI_API_KEY"]=API
client = OpenAI()
api_key_mistral = os.environ["MISTRAL_API_KEY"] = "5DZg7ZyUoJiJ1FGrbXjfzp6Bad40aSrp"
client_mistral = MistralClient(api_key=api_key_mistral)
embedding_model = MistralAIEmbeddings(model = "mistral-embed")




def retriever(q):
    persistent_client = chromadb.PersistentClient(path='spotify_review_db2')
    vector_store_from_client = Chroma(
        client=persistent_client,
        collection_name='spotify_review2',
        embedding_function=embedding_model,
        )
    
    results = vector_store_from_client.similarity_search(
        q,
        k=5,
    )
    final_results = [results[i].page_content for i in range(len(results))]
    final_results = ', '.join(final_results)
    return final_results

def retriever_list(path, collection, q, k):
    persistent_client = chromadb.PersistentClient(path=path)
    vector_store_from_client = Chroma(
        client=persistent_client,
        collection_name=collection,
        embedding_function=embedding_model,
        )
    
    results = vector_store_from_client.similarity_search(
        q,
        k=k,
    )
    final_results = [results[i].page_content for i in range(len(results))]
    # final_results = ', '.join(final_results)
    return final_results

def retrieve_and_summary(q):
    text = retriever(q)
    return text


def retrieval_augmented_generation(user_message):
    retrieved_chunk = retrieve_and_summary(q=user_message)
    prompt = f"""
        retrieved review is below:
        ---------------------
        {retrieved_chunk}
        ---------------------
        Given the context about spotify review. answer the query.
        Query: {user_message}
        Answer:
        """

    return  prompt

def sql_generator(user_message):
    prompt = f"""
        Given table name "review" with columns:
        1. "review_text": user review (text).
        2. "rating" : user rating (Excellent,Good,Average,Below Average,Poor,very poor)
        3. "author_app_version" : spotify app version
        4. "review_timestamp" : timestamp of user that give a review
        you have to translate user question to sql query with template:
        <sql> your code <sql>. don't use newline in sql code and do not give another response!
        Query: {user_message}
        Answer:
        """
    return  prompt

def clean_query(query: str) -> str:
    cleaned_query = query.replace('\n', ' ').replace('\\', '').replace('/',',')
    cleaned_query = ' '.join(cleaned_query.split())
    return cleaned_query



tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieval_augmented_generation",
            "description": "get the data only about spotify music review or related",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the query of user if they asking related to spotify music review",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "this function is to translate question to sql query",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "the question of user if they asking for filter, table report etc.. about spotify review.",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        }
    }
]
