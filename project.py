import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = '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'
llm = OpenAI(temperature=0.9)

loader = DirectoryLoader('C:/Users/filip.balsewicz/Desktop/my_folder')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1600, chunk_overlap=0)

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

docs = retriever.get_relevant_documents("What is the text about")

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
query = "Is UC Berkeley the University of California? "
result = qa({"query": query})

print(result)
