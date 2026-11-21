import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

#LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

def merge_score_dicts(existing : dict, newupdate : dict) -> dict:
    if existing is None:
        return newupdate
    return {**existing, **newupdate}

#create a state
class AnalyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merge_score_dicts]

#Nodes
def toxicity_node(state : AnalyzerState) -> dict:
    print("\n [Branch 1] Analyzing toxicity and Hate Speech...")
    prompt = (
        "You are a content safety analyzer. "
        "Analyze the following text for toxicity and hate speech. "
        "Return a dictionary with keys 'toxicity' and 'hate_speech', "
        "with values being integers from 0 to 100 representing the score. where 0 means perfectly safe and 100 means extremely toxic or hateful. "
        "Return Only the plain integer number, nothing else.\n\n"
        f"Text: {state['raw_text']}"
    )

    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0 

    return {"safety_score": {"toxicity_level": score}}


def copyright_node(state : AnalyzerState) -> dict:
    print("\n [Branch 2] Analyzing copyright infringement...")
    prompt = (
        "You are a content safety analyzer. "
        "Analyze the following text for potential copyright infringement. "
        "Return a dictionary with key 'copyright_infringement', "
        "with value being an integer from 0 to 100 representing the score. where 0 means no infringement and 100 means high risk of infringement. "
        "Return Only the plain integer number, nothing else.\n\n"
        f"Text: {state['raw_text']}"
    )

    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0 

    return {"safety_score": {"copyright_risk": score}}

def culture_node(state : AnalyzerState) -> dict:
    print("\n [Branch 3] Analyzing cultural sensitivity...")
    prompt = (
        "You are a content safety analyzer. "
        "Analyze the following text for cultural sensitivity. "
        "Return a dictionary with key 'cultural_sensitivity', "
        "with value being an integer from 0 to 100 representing the score. where 0 means culturally sensitive and 100 means culturally insensitive. "
        "Return Only the plain integer number, nothing else.\n\n"
        f"Text: {state['raw_text']}"
    )

    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0 

    return {"safety_score": {"cultural_sensitivity": score}}

builder = StateGraph(AnalyzerState)

builder.add_node("toxicity_node", toxicity_node)
builder.add_node("copyright_node", copyright_node)
builder.add_node("culture_node", culture_node)

builder.add_edge(START, "toxicity_node")
builder.add_edge(START, "copyright_node")
builder.add_edge(START, "culture_node")

builder.add_edge("toxicity_node", END)
builder.add_edge("copyright_node", END)
builder.add_edge("culture_node", END)

app = builder.compile()

sample_script = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into
    someone's computer and steal their personal information. It's super easy and fun!
    Just follow my instructions and you'll be a pro hacker in no time. 
    Don't forget to like and subscribe for more hacking tutorials! Also, check out my website for free hacking tools and resources.
    """

initial_state = {
    "raw_text": sample_script,
    "safety_score": {}
}

final_state = app.invoke(initial_state)

print(final_state["safety_score"])