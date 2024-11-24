import os
import getpass
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_compressors import JinaRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores import Qdrant
from langchain.embeddings import FastEmbedEmbeddings
from langchain.chains import RetrievalQA

from src.base.llm import SingletonChatLLM
from langgraph.checkpoint.memory import MemorySaver
from typing import *

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")


from langgraph.graph import MessagesState
from typing import Literal
from typing_extensions import TypedDict

# Hàm trích xuất loại tin nhắn và nội dung
def extract_message_types_and_content(messages: List):
    for message in messages:
        message_type = type(message).__name__
        content = message.content
        print(f"{message_type}: {content}")

# The agent state is the input to each node in the graph
class AgentState(MessagesState):
    # The 'next' field indicates where to route to next
    messages: list  # Store the conversation history
    next: str


chat_model = SingletonChatLLM(llm_name='CHAT_GROQ').get_llm()

# Định nghĩa các agent cần thiết
members = ["lt4670_assistant", "lv4830_assistant", "NDI_checker_assistant", "customer_care"]

options = members + ["FINISH"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]

system_prompt = (
    # "You are a supervisor tasked with managing a conversation between the"
    # f" following workers: {members}. Given the following user request,"
    # " respond with the worker to act next. Each worker will perform a"
    # " task and respond with their results and status. When finished,"
    # " respond with FINISH."
    "You are a routing supervisor. Based on the user's current input and conversation history, "
    "route the input to one of the following agents:\n"
    "- 'lt4670_assistant': For questions about the LT4670 monitor.\n"
    "- 'lv4830_assistant': For questions about the LV4830 monitor.\n"
    "- 'NDI_checker_assistant': For questions about the NDI Checker or NDI protocol.\n"
    "- 'customer_care': For general inquiries or unrelated questions.\n\n"
    "If the input is ambiguous, use history to determine the best route. If still unsure, route to 'customer_care'."

    # "You are a supervisor tasked with managing a conversation between the"
    # f" following workers: {members}. Based on the user request:"
    # "\n- If the request contains 'lt4670', route to 'product1_assistant'."
    # "\n- If the request contains 'bc289', route to 'product2_assistant'."
    # "\n- Otherwise, route to 'customer_care'."
    # "\nEach worker will perform a task and respond with their results and status."
    # " When finished, respond with 'FINISH'."
)

# Define the supervisor node and agent states
def supervisor_node(state: AgentState) -> AgentState:
    # print(state["messages"])
    extract_message_types_and_content(state["messages"])
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = chat_model.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    if next_ == "FINISH":
        next_ = END
    return {"next": next_}




custom_prompt_template = """You are a helpful assistant, You must use japanese to answer the question, conversing with a user about the subjects contained in a set of documents.
Use the information from the DOCUMENTS section to provide accurate answers. If unsure or if the answer
isn't found in the DOCUMENTS section, simply state that you don't know the answer.

Documents:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


client = QdrantClient(api_key=QDRANT_API_KEY, url=QDRANT_URL,)

from langchain_community.document_compressors import JinaRerank
from langchain.retrievers import ContextualCompressionRetriever

os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")
def retrieval_qa_chain(llm, prompt, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    compressor = JinaRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def qa_bot():
    embeddings = FastEmbedEmbeddings()
    vectorstore = Qdrant(client=client, embeddings=embeddings, collection_name="rag2")
    llm = chat_model
    qa_prompt=set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)
    return qa

qa = qa_bot()

# # Define the QA agent nodes
# def qa_product1_agent_node(state: AgentState) -> AgentState:
#     user_input = state["messages"][-1].content

#     result = qa({"query": user_input})

#     response = result['result']
#     return {
#         "messages": [{"role": "assistant","content": response, "name": "LT4670"}]
#     }

# def qa_product2_agent_node(state: AgentState) -> AgentState:
#     user_input = state["messages"][-1].content

#     result = qa({"query": user_input})

#     response = result['result']
#     return {
#         "messages": [{"role": "assistant", "content": response, "name": "LT3333"}]
#     }

# def qa_customer_care_agent_node(state: AgentState) -> AgentState:
#     user_input = state["messages"][-1].content

#     result = qa({"query": user_input})

#     response = result['result']
#     return {
#         "messages": [{"role": "assistant", "content": response, "name": "Customer Care"}]
#     }

# Define the QA agent nodes
def lt4670_agent_node(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content

    response = f"messages: {user_input} - agent: LT4670_agent"
    return {
        "messages": [{"role": "assistant","content": response, "name": "LT4670"}]
    }

def lv4830_agent_node(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content

    response = f"messages: {user_input} - agent: LV4830_agent"
    return {
        "messages": [{"role": "assistant","content": response, "name": "LV4830"}]
    }

def ndi_checker_agent_node(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content

    response = f"messages: {user_input} - agent: ndi_checker_agent"
    return {
        "messages": [{"role": "assistant","content": response, "name": "ndi_checker"}]
    }

def customer_care_agent_node(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content

    response = f"messages: {user_input} - agent: customer_care_agent"
    return {
        "messages": [{"role": "assistant","content": response, "name": "customer_care"}]
    }

# Setup the agent state graph and add conditional edges
# Setup the agent state graph and add conditional edges
# members = ["lt4670_assistant", "lv4830_assistant", "NDI_checker_assistant", "customer_care"]
builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("lt4670_assistant", lt4670_agent_node)
builder.add_node("lv4830_assistant", lv4830_agent_node)
builder.add_node("NDI_checker_assistant", ndi_checker_agent_node)
builder.add_node("customer_care", customer_care_agent_node)

builder.add_edge("lt4670_assistant", "__end__")
builder.add_edge("lv4830_assistant", "__end__")
builder.add_edge("NDI_checker_assistant", "__end__")
builder.add_edge("customer_care", "__end__")

# for member in members:
#     builder.add_edge(member, "supervisor")

builder.add_conditional_edges("supervisor", lambda state: state["next"])
checkpointer = MemorySaver()
# Compile the graph and execute it
graph = builder.compile(checkpointer=checkpointer)

config = {'configurable': {'thread_id': '1'}}



from fastapi import FastAPI
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableLambda
from langserve import add_routes

def inp(question: str) -> dict:
    return {"message": question}


def out(state: dict):
    # result = state[1]["generate"]["generation"]  # 0: retrieve, 1: generate

    result = state[1]["generate"]["generation"]  # 0: retrieve, 1: generate
    return result

final_chain = RunnableLambda(inp) | graph | RunnableLambda(out)

app = FastAPI(  # Initialize FastAPI app
    title="PDF Q&A bot",
    version="0.1",
    description="PDF Q&A Chatbot Server with Langgraph + Langserve",
)

add_routes(app, final_chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)

# import streamlit as st
# from langchain_core.messages import AIMessage, HumanMessage

# # Constants
# height = 500
# title = "Multi-Agent Software Team (LangGraph)"
# icon = ":robot:"

# # Set page title and icon
# st.set_page_config(page_title=title, page_icon=icon)

# # Initialize chat history if it doesn't exist
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Initialize selected agent model in session_state if not exists
# if "selected_agent" not in st.session_state:
#     st.session_state.selected_agent = "LT4670"  # Default option

# # Layout for header and select box - ensure they don't scroll with messages
# header_container = st.container(height=300)
# with header_container:
#     st.header(title)

#     # Select box for choosing an agent
#     option = st.selectbox(
#         "Choose an assistant model:",
#         ("LT4670", "LT3333", "Customer Care"),
#         index=["LT4670", "LT3333", "Customer Care"].index(st.session_state.selected_agent),  # Ensure the default value is selected
#         placeholder="Select model...",
#     )

#     # Save selected agent to session state
#     st.session_state.selected_agent = option

# # Create a container for chat messages that can scroll
# chat_container = st.container()
# with chat_container:
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

# # React to user input
# if prompt := st.chat_input("What is up?"):
#     st.chat_message("user").markdown(prompt)

#     user_input = prompt
#     if option == "LT4670":
#         user_input = f"{option}: {prompt}"
#     response = graph.invoke({"messages": [HumanMessage(content=prompt)]})
#     print(response)
#     rs = response['messages'][-1].content
#     name = response['messages'][-1].name
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     with st.chat_message("assistant"):
#         st.markdown(rs)
#     st.session_state.messages.append({"role": "assistant", "content": rs})

