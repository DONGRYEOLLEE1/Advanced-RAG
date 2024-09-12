"""
Self-RAG Implementation
paper: Self-RAG: Learning to Retrieve, Generate and Critique through Self-reflection
link: https://arxiv.org/abs/2310.11511
"""

import argparse

from typing import List

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults

from util.chain import return_chain
from util.grader import GradeDocuments
from util.prompt import PromptHelper


def mani(query: str):
    
    # init
    emb = OpenAIEmbeddings(model = "text-embedding-3-small")
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    
    # load vector-data
    vectorstore = Chroma(
        collection_name = "rag-chroma", 
        persist_directory = "/data/dev/Advanced-RAG/data", 
        embedding_function = emb)
    retriever = vectorstore.as_retriever(search_kwargs = {"k": 3})
    
    # grader
    