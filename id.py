from llama_cpp import Llama, LlamaGrammar
from tools import make_enum_grammar
#from tools import make_enum_grammar

prompt_PATH = "/Users/silei/Code/local_code/intent_detection/system_prompt.txt"
labels = {
    "Process structure",
    "Distribution of cases over paths",
    "Throughput time of cases",
    "Resource utilization rate",
    "other",
}

def intent_detection(
        prompt_from_user: str,
        llm: Llama
) -> str:
    semantic_tag = ""
    messages = []
    grammar = make_enum_grammar(labels, True)
    
    with open(prompt_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_from_user},
    ]
    
    res = llm.create_chat_completion(
	    messages,
        max_tokens=512,
        temperature=0.1,
        repeat_penalty=1.00,
        grammar=grammar,
        )

    """
    TODO: add memory storage in this part + monitoring maybe.
    """
    """
    TODO: try to solve the multiple intent input query.
    """
    
    semantic_tag = res["choices"][0]["message"]["content"]

    return semantic_tag 