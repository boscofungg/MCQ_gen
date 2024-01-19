# -*- coding: utf-8 -*-


# %pip install langchain
# %pip install sentence-transformers
# %pip install pinecone-client
# %pip install cohere

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["COHERE_API_KEY"] = "WRbXNn3iT1qFDKHMBVUa81w39VXT8oVhJOSFgiR9"

from langchain.document_loaders import PyPDFDirectoryLoader
#Function to read documents
def load_docs(directory):
  loader = PyPDFDirectoryLoader(directory)
  documents = loader.load()
  return documents
# Passing the directory to the 'load_docs' function
directory = '/Users/boscofung/Desktop/AI project/MCQcreator/Sample Data'
documents = load_docs(directory)
len(documents)
documents[0]

from langchain.text_splitter import RecursiveCharacterTextSplitter
#This function will split the documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs
docs = split_docs(documents)
print(len(docs))

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

import pinecone
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key="6cc8c083-3a48-48c1-b7e2-2a1d09b37dc2",
    environment="gcp-starter"
)

index_name = "mcq-creator"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

#This function will help us in fetching the top relevent documents from our vector store - Pinecone
def get_similiar_docs(query, k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
llm=HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
llm
chain = load_qa_chain(llm, chain_type="stuff")

#This function will help us get the answer to the question that we raise
def get_answer(query):
  relevant_docs = get_similiar_docs(query)
  print(relevant_docs)
  response = chain.run(input_documents=relevant_docs, question=query)
  return response

our_query = "what is oppurtunity cost?"
answer = get_answer(our_query)
print(answer)

import re
import json
# from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="question", description="Question generated from provided input text data."),
    ResponseSchema(name="choices", description="Available options for a multiple-choice question in comma separated."),
    ResponseSchema(name="answer", description="Correct answer for the asked question.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser

# This helps us fetch the instructions the langchain creates to fetch the response in desired format
format_instructions = output_parser.get_format_instructions()

print(format_instructions)

# Implement a chatmodel
from langchain_community.chat_models import ChatCohere
chat_model = ChatCohere(model="command", max_tokens=256, temperature=0.75)
chat_model

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("""When a text input is given by the user, please generate multiple choice questions
        from it along with the correct answer.
        \n{format_instructions}\n{user_prompt}""")
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)

final_query = prompt.format_prompt(user_prompt = answer)
print(final_query)

final_query.to_messages()

final_query_output = chat_model(final_query.to_messages())
print(final_query_output.content)

# Let's extract JSON data from Markdown text that we have
markdown_text = final_query_output.content
json_string = re.search(r'{(.*?)}', markdown_text, re.DOTALL).group(1)
print(json_string)