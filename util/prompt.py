import abc

from string import Template

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