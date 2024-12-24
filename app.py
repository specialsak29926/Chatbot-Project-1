import streamlit as st
import pickle 
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain


# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ##  About
    This app is an LLM-powered chatbot built using:
    -[Streamlit](https://streamlit.io/)
    -[LangChain](https://python.langchain.com/)
    ''')
    add_vertical_space(5)
    st.write('Made by Sakshi from IIT Bombay')

def main():
    st.header("Chat With your PDF") 
    load_dotenv()
    # upload a pdf file
    pdf = st.file_uploader("Upload yourPDF",type = 'pdf')
    
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
            )
        chunks = text_splitter.split_text(text=text)

        ## embeddings
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
             embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
             VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
             with open(f"{store_name}.pkl","wb") as f:
                  pickle.dump(VectorStore,f)
             st.write('Embeddings Computation Completed ')

    # Accept user question/query
    query = st.text_input("Ask questions about your PDF file:")
    #st.write(query)

    if query:
        docs = VectorStore.similarity_search(query=query,k=3)
        #llm =  HuggingFaceHub(repo_id="google/flan-t5-small",huggingfacehub_api_token= "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", model_kwargs={"temperature": 0.3, "max_length": 256})
        #chain = load_qa_chain(llm=llm,chain_type='stuff')
        #response = chain.run(input_documents=docs,question=query)
        st.write(docs)    
 




    
        st.write(chunks)

if __name__ == '__main__':
   main()