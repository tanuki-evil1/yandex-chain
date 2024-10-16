from dotenv import load_dotenv
from langchain_community.llms import YandexGPT
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import YandexGPTEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

load_dotenv()


def get_documents_from_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    return split_docs


def create_db(docs):
    texts = [doc.page_content for doc in docs]
    embeddings = YandexGPTEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store


def create_chain(vector_store):
    model = YandexGPT(temperature=0.4, max_tokens=150, verbose=True)
    prompt = ChatPromptTemplate.from_template(
        """Отвечай на вопросы пользователя только по контексту и в конце ты должен задавать вопрос и вести диалог:
        Context: {context}
        Question: {input}
        """
    )

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,

    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retriever_chain


docs = get_documents_from_pdf('amo_info.pdf')
vector_store = create_db(docs)
chain = create_chain(vector_store)
response = chain.invoke({"input": "Рассылка интересует и мне очень нужна"})

print(response["answer"])
