from dotenv import load_dotenv
from langchain_community.llms import YandexGPT
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = YandexGPT(temperature=0.4, max_tokens=100, verbose=True)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ты AI повар. Создаешь уникальный рецепт который основан на главный ингридиентах"),
        ("human", "{input}")
    ]
)

chain = prompt | llm

print(chain.invoke({"input": "Помидор"}))
