from langchain_community.chat_models import ChatYandexGPT
from langchain_core.messages import HumanMessage, SystemMessage

folder_id = 'b1gedtgh5bepjo28uf9p'
yandexgpt_key = 'AQVN33EPOxrFvr3udf35bbaQv-PbY0wGEIQIzeok'
chat_model = ChatYandexGPT(api_key=yandexgpt_key, folder_id=folder_id)

answer = chat_model.invoke(
    [
        SystemMessage(
            content="You are a helpful assistant that translates English to Russian."
        ),
        HumanMessage(content="I love programming."),
    ]
)
print(answer.content) # Я люблю программирование.

