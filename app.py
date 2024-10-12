import os
from flask import Flask, render_template
from flask import request, jsonify, abort

from langchain import LLMChain, PromptTemplate
from langchain.llms import Cohere
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)


def answer_from_knowledgebase(message):
    # TODO: Write your code here
    return ""


def search_knowledgebase(message):
    # TODO: Write your code here
    sources = ""
    return sources


def answer_as_chatbot(message):
    template = """You are an expert Python developer. 
Answer the following question in a clear and informative manner:

Question: {question}

Answer:"""
    memory = ConversationBufferMemory()
    memory.add_user_message(message)
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    res = llm_chain.run(message)
    return res


@app.route("/kbanswer", methods=["POST"])
def kbanswer():
    # TODO: Write your code here

    # call answer_from_knowledebase(message)

    # Return the response as JSON
    return


@app.route("/search", methods=["POST"])
def search():
    # Search the knowledgebase and generate a response
    # (call search_knowledgebase())

    # Return the response as JSON
    return


@app.route("/answer", methods=["POST"])
def answer():
    message = request.json["message"]

    # Generate a response
    response_message = answer_as_chatbot(message)

    # Return the response as JSON
    return jsonify({"message": response_message}), 200


@app.route("/")
def index():
    return render_template("index.html", title="")


if __name__ == "__main__":
    app.run()
