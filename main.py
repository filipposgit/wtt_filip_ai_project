import streamlit as st
from langchain import OpenAI
from streamlit_chat import message

st.set_page_config(page_title=" Document retrieval and question answering with langchain library", page_icon=":robot:")
st.header(" Document retrieval and question answering with langchain library")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        "This code demonstrates how to use the langchain library to perform document retrieval and question "
        "answering. It loads a directory of documents, splits them into smaller text chunks, embeds them using OpenAI "
        "embeddings, and indexes them using FAISS. It then retrieves relevant documents based on a query and uses "
        "OpenAI language model to answer a given question based on the retrieved documents.")

with col2:
    st.image(image="streamlit.png", width=200)
    st.image(image="langchain.png", width=200)

with col3:
    st.image(image="chat_gpt.png", width=200)


def load_llm():
    llm2 = OpenAI(temperature=0.5)
    return llm2


llm = load_llm()

uploaded_file = st.file_uploader("Upload your file to analyze. ")


def question():
    input_text = st.text_area("Enter your question", key='my_new_key')
    return input_text


final_input = question()


def answer():
    the_answer = llm(final_input)
    return the_answer


final_answer = answer()
message_final_answer = message(final_answer)

st.write(message_final_answer)
