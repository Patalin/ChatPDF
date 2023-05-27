import streamlit as st
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Sidebar contents
with st.sidebar:
    st.title('ChatPDF App')
    st.markdown(''' 
    ## About
    Chat with any PDF file.
    ''')
    add_vertical_space(1)

    # Let users input their OpenAI key
    user_key = st.text_input("Enter your OpenAI key: ")
    os.environ['OPENAI_API_KEY'] = user_key

    st.write('Made by Catalin')

def main():
    st.header("Chat with any PDF: ")
    # upload PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
          text += page.extract_text()

      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000,
          chunk_overlap=200,
          length_function=len
      )
      chunks = text_splitter.split_text(text=text)

      # embeddings
      store_name = pdf.name[:4]

      if os.path.exists(f"{store_name}.pkl"):
          with open(f"{store_name}.pkl", "rb") as f:
              VectorStore = pickle.load(f)
          # st.write('Embeddings loaded from the disk.')
      else:
          embeddings = OpenAIEmbeddings()
          VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
          with open(f"{store_name}.pkl", "wb") as f:
              pickle.dump(VectorStore, f)
          # st.write('Embeddings Computation Completed')

      # Accept use question/query
      query = st.text_input("Ask questions bout your PDF file: ")
      
      if query:  
        docs = VectorStore.similarity_search(query=query, k=3)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb: 
          response = chain.run(input_documents=docs, question=query)
          print(cb)

          #test
        st.write(response)


if __name__ == '__main__':
    main()


