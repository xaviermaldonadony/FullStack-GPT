from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import  CacheBackedEmbeddings, OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableLambda
from langchain.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import time as t
import streamlit as st

st.set_page_config(
   page_title="PrivateGPT",
   page_icon="🤖"
)
st.title("PrivateGPT")

class ChatCallbackHandler(BaseCallbackHandler):
   message = ""

   def on_llm_start(self, *args, **kwargs):
      self.message_box = st.empty()

   def on_llm_end(self, *args, **kwargs):
      save_message(self.message, "ai")

   def on_llm_new_token(self,token, *args, **kwargs):
      self.message += token
      self.message_box.markdown(self.message)


llm = ChatOllama(
   model="llama3.2:latest",
   temperature = 0.1,
   streaming = True,
   callbacks=[ChatCallbackHandler(),]
   )

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
   file_content = file.read();
   file_path = f"./.cache/private_files/{file.name}"
   with open(file_path, "wb") as f:
      f.write(file_content)

   cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
   splitter = CharacterTextSplitter.from_tiktoken_encoder(
      separator="\n",
      chunk_size=600,
      chunk_overlap=100
   )
   # load files
   loader = UnstructuredFileLoader(file_path)

   # split files, into small docs makes it easier for llm to read 
   docs = loader.load_and_split(text_splitter=splitter)

   # embeddings are vector representation, the meaning behind the text docsk
   embeddings = OllamaEmbeddings(model="llama3.2:latest")

   cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

   # takes docs and embeddings. we use the store to later make seraches and get related data/info
   vectorstore = FAISS.from_documents(docs, cached_embeddings)
   retriever = vectorstore.as_retriever()
   return retriever

def send_message(message, role, save=True):
   with st.chat_message(role):
      st.markdown(message)
   if save:
      save_message(message, role)

def save_message(message, role):
   st.session_state["messages"].append({"message": message, "role": role})

def paint_history():
   for message in st.session_state["messages"]:
      send_message(message["message"], message["role"], save=False)

def format_docs(docs):
   return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     Answer the question using ONLY the following context. If you don't know the answer just say you
     don't know. DON'T make anything up.

     Context: {context}
     """),
     ("human", "{question}"),
 ])

st.markdown(
   """
   Welcome?\n
   Use this chatbot to ask questionto an AI about your files!\n
   UPload your files on the sidebar.\n
   """
)

with st.sidebar:
   file = st.file_uploader("Upload .txt .pdf or .docx file", type=["txt", "pdf", "docx"])

if file:
   retriever = embed_file(file)
   send_message("I'm ready? Ask away!", "ai", save=False)
   paint_history()
   message = st.chat_input("Ask me anything about your file...")

   if message:
      send_message(message, "human")
      chain = ({
         "context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
      } | prompt | llm)
      with st.chat_message("ai"):
         chain.invoke(message)

else:
   st.session_state["messages"] = []