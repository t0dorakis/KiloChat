
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain import LLMChain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, ConversationalAgent
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
import requests
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# LLM SETTING
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.9)

# Document Loader
loader = TextLoader('./200kilo.txt')
documents = loader.load()

# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = OpenAIEmbeddings()


def embed_data():
    embeddings = OpenAIEmbeddings()
    # Indexing
    # Save in a Vector DB
    print("Indexing...")
    index = FAISS.from_documents(docs, embeddings)
    print("Embeddings done. ✅", index)

    return index


index = embed_data()

# TOOLS

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.as_retriever(),
)


tools = [
    Tool(
        name="Search Information",
        func=qa.run,
        description="Useful to find information",
    )
]

# MEMORY

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# The chat fails on finish if the memory is not read only
readonlymemory = ReadOnlySharedMemory(memory=memory)


# PROMPT

prefix = """
Tools:
    """
suffix = """ Have a conversation with a human, answering the following questions as best you can. You are a body builder.\
    You are working for 200kilo as a personal trainer and you also have good knwoldedge about them and their work.\
    You will answer the questions of visitors of the website. Answer in the tone of voice of a body builder. Begin!"
{chat_history}
Question: {input}
{agent_scratchpad}"""

human_prefix = "Answer in the tone of voice a super nice and friendly body builder-bro:"


prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    human_prefix=human_prefix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

# AGENT

# TODO: dthis does not Model does not dream enough

llm_chain = LLMChain(llm=ChatOpenAI(
    temperature=0.9, model='gpt-3.5-turbo'), prompt=prompt)

agent_chain = initialize_agent(
    tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

# Answering a question given its context


def answer_question(question):
    result = agent_chain.run(
        input=question)

    # TODO: add a token tracker
    return result
