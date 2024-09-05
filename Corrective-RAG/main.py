"""
Corrective-RAG (CRAG) Implementation
paper: Corrective Retrieval Augmented Generation 
link: https://arxiv.org/abs/2401.15884
"""

import os
import abc
import argparse

from string import Template
from typing import Literal, Any, Optional, List
from langchain_core.runnables import RunnableSequence

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults


def return_chain(prompt, llm, etc: Optional[Any] = None) -> RunnableSequence:
    if etc == None:
        return (prompt | llm)
    else:
        return (prompt | llm | etc)

class PromptHelper(abc.ABC):
    
    prompts = {
        "grader_system": {
            "1.0": Template("You are a grader assessing relevance of a retrieved document to a user question. If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.")
        },
        "grader_human": {
            "1.0": Template("Retrieved document:\n\n{document}\n\nUser question:\n\n{question}")
        },
        "grader_human_cot": {
            "1.0", Template("Retrieved document:\n\n{document}\n\nUser question:\n\n{question}\nThink Step by step, and answer with yes or no only.")
        },
        "question_rewrite_system": {
            "1.0": Template("You are a question re-writer that converts as input question to a better version that is optimized for web search. Look at the input and try to reason about the underlying semantic intent / meaning.")
        },
        "question_rewrite_human": {
            "1.0": Template("Here is the initial question:\n\n{question}\nFormulate an improved question.")
        },
        "generation_system": {
            "1.0": Template("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion:{question}\nContext:{context}\nAnswer:")
        }
    }
    
    @classmethod
    def get_prompt(cls, prompt_type, version = "1.0", **kwargs):
        prompt_template = cls.prompts[prompt_type][version]
        
        return prompt_template.substitute(kwargs)

    
class GradeDocuments(BaseModel):
    """Binary score for relevant check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description = "Documents are relevant to the question, 'yes' or 'no'."
    )


def main(query: str):
    
    # Initialization
    emb = OpenAIEmbeddings(model = "text-embedding-3-small")
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    
    """
    Save Vector Data to `Chroma`
    
    >>> text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20, length_function = len, is_separator_regex = False)
    >>> urls = ["https://namu.wiki/w/%ED%94%84%EB%9E%91%ED%82%A4"]
    >>> docs = [WebBaseLoader(url).load() for url in urls]
    >>> docs_list = [item for sublist in docs for item in sublist]
    >>> docs_splits = text_splitter.split_documents(docs_list)
    >>> vectorstore = Chroma.from_documents(documents = docs_splits, collection_name = "rag-chroma", embedding = emb, persist_directory = "/data/dev/Advanced-RAG")
    """
    
    ## load vector-data
    vectorstore = Chroma(collection_name = "rag-chroma", persist_directory = "/data/dev/Advanced-RAG", embedding_function = emb)
    retriever = vectorstore.as_retriever()  # k = 4
    
    # grader
    system_grader = PromptHelper.get_prompt("grader_system")
    human_grader = PromptHelper.get_prompt("grader_human")
    prompt_grader = ChatPromptTemplate.from_messages(
        [("system", system_grader), ("human", human_grader)]
    )
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    grader_chain = return_chain(prompt_grader, structured_llm_grader)
    
    # rewritter (Knowledge Refinement)
    system_rewritter = PromptHelper.get_prompt("question_rewrite_system")
    human_rewritter = PromptHelper.get_prompt("question_rewrite_human")
    prompt_rewrite = ChatPromptTemplate.from_messages(
        [("system", system_rewritter), ("human", human_rewritter)]
    )
    question_rewriter = return_chain(prompt_rewrite, llm, StrOutputParser())
    
    # generation
    system_generation = PromptHelper.get_prompt("generation_system")
    prompt_generation = ChatPromptTemplate.from_messages([("system", system_generation)])
    generation_chain = return_chain(prompt_generation, llm, StrOutputParser())

    
    ## CRAG pipeline
    relevant_docs: List[Document] = retriever.invoke(query)
    
    grader_relevant_docs = []
    web_search = "No"
    for document in relevant_docs:
        confidence_degree = grader_chain.invoke({"question": query, "document": document.page_content}) # Confidence Degree
        
        if confidence_degree.binary_score == "yes":
            grader_relevant_docs.append(document)
        else:
            web_search = "Yes"
            
    if web_search == "Yes":
        print("Web Search!")
        rewritted_query = question_rewriter.invoke({"question": query})
        
        web_search_tool = TavilySearchResults(
            name = "tavily_search_results", 
            max_results = 3
        )
        web_results = web_search_tool.invoke({"query": rewritted_query})
        
        web_search_docs = [
            Document(
                metadata = {"url": web_res['url']},
                page_content = web_res['content']
            ) for web_res in web_results
        ]
        
        grader_relevant_docs.extend(web_search_docs)
        
        final_result = generation_chain.invoke({"question": rewritted_query, "context": grader_relevant_docs})
    else:
        final_result = generation_chain.invoke({"question": query, "context": grader_relevant_docs})
        
    print(final_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type = str)
    args = parser.parse_args()
    
    main(query = args.query)