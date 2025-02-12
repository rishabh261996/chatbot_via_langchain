import os
from flask import Flask, render_template
from flask import request, jsonify, abort

from langchain import LLMChain, PromptTemplate
from langchain.llms import Cohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage, AIMessage



app = Flask(__name__)

def answer_from_knowledgebase(message):
    res = qa({"query": message})
    return res['result']

def search_knowledgebase(message):
    res = qa({"query": message})
    sources = ""
    for count, source in enumerate(res['source_documents'],1):
       sources += "Source " + str(count) + "\n"
       sources += source.page_content + "\n"

from langchain.schema import HumanMessage, AIMessage
import os

def answer_as_chatbot(message):
    # Initialize memory
    memory = ConversationBufferMemory()

    # Define the prompt template
    template = """You are an expert Python developer. 
    Answer the following question in a clear and informative manner:
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Add user message to memory
    memory.chat_memory.add_message(HumanMessage(content=message))

    # Initialize the language model (Cohere) and the LLM chain
    try:
        llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
    except KeyError:
        return "Error: COHERE_API_KEY not found in environment variables."

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Run the LLM chain and generate the response
    try:
        res = llm_chain.run(message)
    except Exception as e:
        return f"Error generating response: {e}"

    # Add AI response to memory
    memory.chat_memory.add_message(AIMessage(content=res))

    # Return the response
    return res


def load_db():
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"])
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=Cohere(),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        return qa
    except Exception as e:
        print("Error:", e)

qa = load_db() 

@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json['message']
    
    response_message = answer_from_knowledgebase(message)
    
    return jsonify({'message': response_message}), 200

@app.route('/search', methods=['POST'])
def search():
    message = request.json['message']
    response_message = search_knowledgebase(message)
    return jsonify({'message': response_message}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json['message']
    
    # Generate a response
    response_message = answer_as_chatbot(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="")

if __name__ == "__main__":
    app.run()
