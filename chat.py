
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain import LLMChain
from langchain.document_loaders import NotionDirectoryLoader
from langchain.llms import OpenAI
from langchain import OpenAI, ConversationChain
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import requests
import os
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# PROMPT TEMPLATE
template = """
Context: {context}

Question: {question}

Answer like a heavy weight body builder
Answer: """

# prompt = PromptTemplate(input_variables=["question", "context"], template=template)
# prompt.format(question="Can Barack Obama have a conversation with George Washington?")


# LLM SETTING
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.9)

# CONVERSATION CHAIN

# conversation = ConversationChain(llm=llm, verbose=True)
# conversation.predict(input="Hi there!")
# conversation.predict(input="Can we talk about AI?")
# conversation.predict(input="I'm interested in Reinforcement Learning.")


# AGENTS

# tools = load_tools(["wikipedia", "llm-math"], llm=llm)
# agent = initialize_agent(
#     tools, llm, agent="zero-shot-react-description", verbose=True)
# agent.run("Can Barack Obama have a conversation with George Washington?")


# LOADING DOCUMENTS

# loader = NotionDirectoryLoader("Notion_DB")

# docs = loader.load()

# url = "https://raw.githubusercontent.com/hwchase17/langchain/master/docs/modules/state_of_the_union.txt"
# res = requests.get(url)
# with open("state_of_the_union.txt", "w") as f:
#   f.write(res.text)

# Document Loader
loader = TextLoader('./200kilo.txt')
documents = loader.load()

# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings()

# text = "This is a test document."
# query_result = embeddings.embed_query(text)
# doc_result = embeddings.embed_documents([text])
db = FAISS.from_documents(docs, embeddings)

# Save and load:
db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings)
# print(docs[0].page_content)

# query = "What did the president say about Ketanji Brown Jackson"
# docs = db.similarity_search(query)


# PROMPT EXECUTION
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template,
)

messages = [
    SystemMessage(content="You are a body builder.\
    You are working for 200kilo as a personal trainer and you also have good knwoldedge about them and their work.\
    You will answer the questions of visitors of the website.  "),
]

chain = LLMChain(llm=llm, prompt=prompt)


# question = "Can Barack Obama have a conversation with George Washington?"


# print(llm_chain.run(prompt.format(
#     question="Can Barack Obama have a conversation with George Washington?")))


# write a function that takes in a question and returns an answer to be serverd over the web


# Answering a question given its context
def answer_question(question):
  return chain.run(question=question, context=docs[0].page_content)
