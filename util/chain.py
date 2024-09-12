from typing import Optional, Any

from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def return_chain(
    prompt: ChatPromptTemplate,
    llm: ChatOpenAI,
    parser: Optional[Any] = None
) -> RunnableSequence:
    """
    Return a RunnableSequence that first runs the prompt, then the llm, and finally the parser.
    
    Args:
        prompt (ChatPromptTemplate): The prompt to run first
        llm (ChatOpenAI): The llm to run second
        parser (Optional[Any]): The parser to run third. If None, the parser is skipped.
    
    Returns:
        RunnableSequence: A RunnableSequence that runs the prompt, llm, and parser in order.
    """
    
    if parser == None:
        return (prompt | llm)
    else:
        return (prompt | llm | parser)