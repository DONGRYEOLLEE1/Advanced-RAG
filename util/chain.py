from typing import Optional, Any

from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def return_chain(
    prompt: ChatPromptTemplate,
    llm: ChatOpenAI,
    **kwargs
) -> RunnableSequence:
    
    """
    This function takes a ChatPromptTemplate and a ChatOpenAI and returns
    a RunnableSequence. The RunnableSequence is created by chaining the
    prompt and the llm together.

    Args:
        prompt (ChatPromptTemplate): The ChatPromptTemplate to pass to the
            llm.
        llm (ChatOpenAI): The llm to use to generate text.
        **kwargs: Additional keyword arguments to pass to the RunnableSequence.
            These can include a 'retrieval_dict' to use for retrieval, and/or
            a 'parser' to use to parse the output of the llm.

    Returns:
        RunnableSequence: The RunnableSequence created by chaining the prompt
            and llm together.
    """
    
    chain = prompt | llm
    
    if 'retrieval_dict' in kwargs:
        chain = kwargs['retrieval_dict'] | chain
    if 'parser' in kwargs:
        chain = chain | kwargs['parser']
        
    return chain