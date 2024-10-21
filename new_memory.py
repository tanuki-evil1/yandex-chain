import time
import uuid

from typing import Sequence

import psycopg
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatYandexGPT
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import SystemMessage, trim_messages

load_dotenv()


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

model = ChatYandexGPT(temperature=0.7)
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Define a new graph
workflow = StateGraph(state_schema=State)


# Define the function that calls the model
def call_model(state: State):
    chain = prompt | model
    trimmed_messages = trimmer.invoke(state["messages"])
    response = chain.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

sync_connection = psycopg.connect("postgresql://tanuki:26032003@localhost:5432/customgpt")
memory = PostgresSaver(conn=sync_connection)
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem did I ask?"
language = "Russian"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",

):
    while True:
        user_input = input("You: ")
        if user_input == 'q':
            break

    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="|")