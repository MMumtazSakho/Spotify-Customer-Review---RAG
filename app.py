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
import json

api_key = os.environ["OPENAI_API_KEY"]="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"
client = OpenAI()


api_key_mistral = os.environ["MISTRAL_API_KEY"] = "5DZg7ZyUoJiJ1FGrbXjfzp6Bad40aSrp"
client_mistral = MistralClient(api_key=api_key_mistral)
embedding_model = MistralAIEmbeddings(model = "mistral-embed")



persistent_client = chromadb.PersistentClient(path='spotify_review_db2')
collection = persistent_client.get_or_create_collection("spotify_review2")
vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="spotify_review2",
    embedding_function=embedding_model,
)


# from transformers import pipeline
# summarizer = pipeline("summarization", model="bart-large-cnn", device='cuda')
# from transformers import BartTokenizer
# tokenizer = BartTokenizer.from_pretrained("bart-large-cnn")

# def summary(text):
#     # Tokenizing the text to ensure it's not too long
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
#     input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
#     summary_text = summarizer(input_text, max_length=200, min_length=60, do_sample=False)
#     return summary_text

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
    return final_results

def retrieve_and_summary(q):
    text = retriever(q)
    # summary_text = summary(text)
    # return summary_text[0]['summary_text']
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


tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieval_augmented_generation",
            "description": "get the data about spotify review",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the query of user if they asking for spotify review",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }
    }
]

import streamlit as st
import json

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.retrieval = []
    st.session_state.total_tokens = 0  
    st.session_state.messages.append({"role": "system", "content": "You are a spotify user review assistant for developer team. Use the supplied tools to assist the user if needed."})
    st.session_state.dummy_messages = []

st.title('Spotify Review Assistant - by Sakho')
for i, message in enumerate(st.session_state.dummy_messages):
    with st.chat_message(message["role"]):
        if message["role"] != 'system':
            st.markdown(message["content"])

if prompt := st.chat_input("I want to know the user review about spotify...?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.dummy_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            tools=tools
        )
        try:
            # Mengambil data tool_call
            tool_call = stream.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            query = arguments.get('query')
            print(query)
            with st.spinner(f"Mencari review terkait {query}..."):
                retrieve_chunk = retrieval_augmented_generation(query)
                st.session_state.messages.append({"role": "system", "content": retrieve_chunk})
                
                # Completion setelah retrieval
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    tools=tools
                )
                rag_response = stream.choices[0].message.content
                st.session_state.retrieval.append(retrieve_chunk)

                option = st.selectbox(options=retrieve_chunk, label="hasil pencarian:")
                st.session_state.messages.append({"role": "assistant", "content": rag_response})
                st.session_state.dummy_messages.append({"role": "assistant", "content": rag_response})
                response = st.write(rag_response)

                total_tokens = stream.usage.total_tokens
                st.session_state.total_tokens += total_tokens  # Tambahkan total token
                st.write(f"Total Tokens: {st.session_state.total_tokens}")

        except Exception as e:
            print('assistant: ', stream.choices[0].message.content)
            response = st.write(stream.choices[0].message.content)
            st.session_state.messages.append({"role": "assistant", "content": stream.choices[0].message.content})
            st.session_state.dummy_messages.append({"role": "assistant", "content": stream.choices[0].message.content})

            total_tokens = stream.usage.total_tokens
            st.session_state.total_tokens += total_tokens  # Tambahkan total token
            st.write(f"Total Tokens: {st.session_state.total_tokens}")







