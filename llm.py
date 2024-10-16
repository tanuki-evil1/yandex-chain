from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate

folder_id = 'b1gedtgh5bepjo28uf9p'
yandexgpt_key = 'AQVN33EPOxrFvr3udf35bbaQv-PbY0wGEIQIzeok'


template = "What is the capital of {country}?"
prompt = PromptTemplate.from_template(template)

llm = YandexGPT(api_key=yandexgpt_key, folder_id=folder_id)
llm_chain = prompt | llm

country = "Russia"

print(llm_chain.invoke(country))
