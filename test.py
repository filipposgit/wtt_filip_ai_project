import os
import io
import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = ''

st.set_page_config(page_title="Document retrieval and question answering with langchain library", page_icon=":robot:")
st.header("Document retrieval and question answering with langchain library")

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

uploaded_file = st.file_uploader("Upload your file to analyze.")


def generate_response(query):
    if uploaded_file is None:
        return "No file uploaded."

    file_stream = io.TextIOWrapper(uploaded_file, encoding='utf-8')
    text_splitter = CharacterTextSplitter(chunk_size=1600, chunk_overlap=0)
    texts = text_splitter.split_documents(file_stream)

    if not texts:
        return "No documents found."

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    docs = retriever.get_relevant_documents(query)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever,
                                     return_source_documents=True)
    result = qa({"query": query, "docs": docs})

    return result


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = generate_response(user_input)
    # store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
