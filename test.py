import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Setting up the page
st.set_page_config(page_title=" Document retrieval and question answering with langchain library", page_icon=":robot:")
st.header(" Document retrieval and question answering with langchain library")

# Creating columns
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


# Loading the llm
def load_llm():
    llm2 = OpenAI(temperature=0.5)
    return llm2


llm = load_llm()

# Upload file function
uploaded_file = st.file_uploader("Upload your file to analyze. ")

documents = uploaded_file.load()

text_splitter = CharacterTextSplitter(chunk_size=1600, chunk_overlap=0)

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

docs = retriever.get_relevant_documents("What is the text about")

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)


def question():
    query = st.text_area("Enter your question", key='my_new_key')
    return query


final_question = question()


def answer():
    result = qa({"final_question": final_question})

    return result


final_answer = answer()

st.write(final_answer)
