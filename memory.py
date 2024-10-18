import time

import psycopg
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatYandexGPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

import uuid

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_postgres import PostgresChatMessageHistory

load_dotenv()

sync_connection = psycopg.connect("postgresql://tanuki:26032003@localhost:5432/customgpt")
session_id = str(uuid.uuid4())

# chat_history = PostgresChatMessageHistory(
#     "chat_history",
#     session_id,
#     sync_connection=sync_connection
# )
#
# # Add messages to the chat history
# chat_history.add_messages([
#     SystemMessage(content="Meow"),
#     AIMessage(content="woof"),
#     HumanMessage(content="bark"),
# ])

workflow = StateGraph(state_schema=MessagesState)
model = ChatYandexGPT(temperature=0.4)


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = PostgresSaver(sync_connection)
app = workflow.compile(checkpointer=memory)



thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

time.sleep(1)
input_message = HumanMessage(content="what was my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a friendly AI assistant."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}")
# ])
#
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )
#
# chain = LLMChain(
#     llm=model,
#     prompt=prompt,
#     verbose=True,
#     memory=memory
# )
