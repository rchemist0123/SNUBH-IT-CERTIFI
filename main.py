import os
import polars as pl
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_teddynote.messages import stream_response
from langchain_teddynote import logging
from langchain.document_loaders import DataFrameLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnablePassthrough
from Constants import *
from Functions import format_document_dynamically, print_message, add_message

load_dotenv()
logging.langsmith("HIMSS RAG")

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings_for_load = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.load_local(
    DB_SAVE_PATH, embeddings_for_load, allow_dangerous_deserialization=True
)
# print(f"✅ '{DB_SAVE_PATH}' 폴더에서 VectorDB를 성공적으로 불러왔습니다.")

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

# query = "PHR"
# retrieved_docs = vector_store.similarity_search(query)

# print("\n--- 검색 결과 ---")
# print(retrieved_docs[0].page_content)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def create_chain():
    prompt = load_prompt(PROMPT_PATH, encoding="utf-8")
    rag_chain = (
        {
            "context": retriever | format_document_dynamically,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


st.title("SNUBH 데이터융합팀 IT 인증 검색 💬")

print_message()

user_input = st.chat_input("궁금한 것을 물어보세요.")

if user_input:

    # 대화 출력
    st.chat_message("user").write(user_input)

    # chain 생성
    chain = create_chain()

    # 스트리밍 호출
    response = chain.stream(user_input)
    # response = retrieve_and_rerank_answer(user_input)
    with st.chat_message("assistant"):
        container = st.empty()
        answer = ""
        for token in response:
            answer += token
            container.markdown(answer)

    # 대화기록 저장.
    add_message("user", user_input)
    add_message("assistant", answer)

    # if answer != "":
    #     with st.popover("참고문서 보기"):
    #         st.write(answer)
