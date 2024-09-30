import streamlit as st
import json
import pandas as pd
import sqlite3
from function import tools,client, retrieval_augmented_generation, sql_generator, clean_query

conn = sqlite3.connect('review.db')

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.retrieval = []
    st.session_state.total_tokens = 0  
    st.session_state.messages.append({"role": "system", 
                                      "content": "You are a spotify user review assistant for developer team. Use the supplied tools to assist the user if needed."})
    st.session_state.dummy_messages = []
    st.session_state.option1 = True
    st.session_state.option1 = True
    st.session_state.llm_tools = tools

st.session_state.option1 = st.checkbox('Retrieval generation',value=True)
st.session_state.option2 = st.checkbox('Table generation', value=True) 
for i, message in enumerate(st.session_state.dummy_messages):
    with st.chat_message(message["role"]):
        if message["role"] != 'system':
            st.markdown(message["content"],unsafe_allow_html=True)

if prompt := st.chat_input("I want to know the user review about spotify...?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.dummy_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if (st.session_state.option1 and st.session_state.option2):
        st.session_state.llm_tools = tools
        print('dua duanya')
    elif st.session_state.option1 and st.session_state.option2==False:
        st.session_state.llm_tools = [tools[0]]
        print('option1')
    elif st.session_state.option2 and st.session_state.option1==False:
        st.session_state.llm_tools = [tools[1]]
        print('option2')
    elif st.session_state.option2==False and st.session_state.option1==False:
        st.session_state.llm_tools = None
        print('option2')

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            tools=st.session_state.llm_tools
        )
        try:
            # ngambil data tool_call
            tool_call = stream.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            # query = arguments.get('query')
            # print(query)
            if tool_call.function.name =='retrieval_augmented_generation':
                query = arguments.get('query')
                with st.spinner(f"Mencari review terkait {query}..."):
                    retrieve_chunk = retrieval_augmented_generation(query)
                    st.session_state.messages.append({"role": "system", "content": retrieve_chunk})
                    
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0.0,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        # tools=st.session_state.llm_tools
                    )
                    rag_response = stream.choices[0].message.content
                    st.session_state.retrieval.append(retrieve_chunk)

                    option = st.selectbox(options=retrieve_chunk, label="hasil pencarian:")
                    st.session_state.messages.append({"role": "assistant", "content": rag_response})
                    st.session_state.dummy_messages.append({"role": "assistant", "content": rag_response})
                    response = st.write(rag_response)
                    total_tokens = stream.usage.total_tokens
                    st.session_state.total_tokens += total_tokens  

            elif tool_call.function.name =='sql_query':
                question = arguments.get('question')
                prompt = sql_generator(question)
                with st.spinner(f"membuat tabel..."):
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                        
                            {
                                "role": "user",
                                "content": prompt
                            },
                        ]
                    )
                    sql_query = completion.choices[0].message.content
                    sql_query = sql_query.split('<sql>')[1]
                    sql_query = clean_query(sql_query)
                    print(sql_query)
                    df = pd.read_sql_query(sql_query, conn)
                    print(df)
                    df = df.to_html()
                    response = st.markdown(df, unsafe_allow_html=True)
                    st.session_state.dummy_messages.append({"role": "assistant", "content":df})
                    st.session_state.messages.append({"role": "assistant", "content":"tabel sudah jadi (respon dummy)"})
                   

        except Exception as e:
            print('assistant: ', stream.choices[0].message.content)
            response = st.write(stream.choices[0].message.content)
            st.session_state.messages.append({"role": "assistant", "content": stream.choices[0].message.content})
            st.session_state.dummy_messages.append({"role": "assistant", "content": stream.choices[0].message.content})
            total_tokens = stream.usage.total_tokens
            st.session_state.total_tokens += total_tokens 









