from dotenv import load_dotenv
from langchain_community.llms import YandexGPT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser

load_dotenv()

llm = YandexGPT(temperature=0.4, max_tokens=100, verbose=True)


def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Ты AI повар. Создаешь уникальный рецепт который основан на главный ингридиентах"),
            ("human", "{input}")
        ]
    )
    parser = StrOutputParser(prompt)
    chain = prompt | llm | parser

    return chain.invoke({"input": "Помидор"})


def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Напиши 10 синонимов к слову раздели их запятыми"),
            ("human", "{input}")
        ]
    )
    parser = CommaSeparatedListOutputParser( )
    chain = prompt | llm | parser

    return chain.invoke({"input": "Помидор"})


print(call_list_output_parser())
