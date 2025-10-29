import os
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from tools import search_web, write_summary
# import gurardrails libraries
from guardrails.hub import GibberishText
from guardrails import Guard


import os
from dotenv import load_dotenv
load_dotenv()



# Use the Guard with the validator
guard = Guard().use(
    GibberishText, threshold=0.5, validation_method="sentence", on_fail="exception"
)


os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


## Define the state
class AgentState(MessagesState):
    next_agent:str #which agent should go next 




llm=init_chat_model("groq:llama-3.1-8b-instant")
llm

# Define agent functions (simpler approach)
def researcher_agent(state: AgentState):
    """Researcher agent that searches for information"""
    
    messages = state["messages"]
    
    # Add system message for context
    system_msg = SystemMessage(content="You are a research assistant. Use the search_web tool to find information about the user's request.")
    
    # Call LLM with tools
    researcher_llm = llm.bind_tools([search_web])
    response = researcher_llm.invoke([system_msg] + messages)
    
    # Return the response and route to writer
    return {
        "messages": [response],
        "next_agent": "writer"
    }

def writer_agent(state: AgentState):
    """Writer agent that creates summaries"""
    
    messages = state["messages"]
    
    # Add system message
    system_msg = SystemMessage(content="You are a technical writer. Review the conversation and create a clear, concise summary of the findings.")
    
    # Simple completion without tools
    response = llm.invoke([system_msg] + messages)
    
    return {
        "messages": [response],
        "next_agent": "end"
    }


# Build graph
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)

# Define flow
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)
final_workflow=workflow.compile()

final_workflow



user_text = "Research about the use case of agentic AI in business."

try:
    validated_text = guard.validate(user_text)

    # Ensure it's a string for HumanMessage
    if isinstance(validated_text, dict):
        validated_text = validated_text.get("text", user_text)
    elif not isinstance(validated_text, str):
        validated_text = str(validated_text)

except Exception as e:
    print("Guard validation failed:", e)
    validated_text = user_text

# Always pass a string inside HumanMessage
response = final_workflow.invoke({
    "messages": [HumanMessage(content=validated_text)]
})


print(response)
response["messages"][-1].content