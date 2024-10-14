import os
from flask import Flask, render_template, request, jsonify, abort

from langchain import LLMChain, PromptTemplate
from langchain.llms import Cohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage, AIMessage

app = Flask(__name__)

def answer_from_knowledgebase(message):
    try:
        res = qa({"query": message})
        return res['result']
    except Exception as e:
        print(f"Error in answer_from_knowledgebase: {e}")
        return "Sorry, I couldn't retrieve an answer from the knowledge base."

def search_knowledgebase(message):
    try:
        res = qa({"query": message})
        sources = ""
        for count, source in enumerate(res['source_documents'], 1):
            sources += f"Source {count}:\n{source.page_content}\n\n"
        return sources
    except Exception as e:
        print(f"Error in search_knowledgebase: {e}")
        return "Sorry, I couldn't search the knowledge base."

def answer_as_chatbot(message):
    memory = ConversationBufferMemory()
    template = """You are an expert Python developer. 
    Answer the following question in a clear and informative manner:
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    # Add user message to memory
    memory.chat_memory.add_message(HumanMessage(content=message))
    
    # Initialize LLM and Chain
    try:
        llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
    except KeyError:
        print("COHERE_API_KEY not found in environment variables.")
        return "Configuration error: COHERE_API_KEY not set."

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # Get the response from the LLM
    try:
        res = llm_chain.run(message)
        print(f"Generated response: {res}")
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating a response."
    
    # Add AI message to memory
    memory.chat_memory.add_message(AIMessage(content=res))
    
    return res

def load_db():
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"])
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=Cohere(cohere_api_key=os.environ["COHERE_API_KEY"]),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        return qa
    except KeyError:
        print("COHERE_API_KEY not found in environment variables.")
    except Exception as e:
        print("Error loading database:", e)
    return None

qa = load_db()

@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json.get('message')
    if not message:
        print("No message provided in /kbanswer.")
        return jsonify({'error': 'No message provided'}), 400
    response_message = answer_from_knowledgebase(message)
    return jsonify({'message': response_message}), 200

@app.route('/search', methods=['POST'])
def search():
    message = request.json.get('message')
    if not message:
        print("No message provided in /search.")
        return jsonify({'error': 'No message provided'}), 400
    response_message = search_knowledgebase(message)
    return jsonify({'message': response_message}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json.get('message')
    if not message:
        print("No message provided in /answer.")
        return jsonify({'error': 'No message provided'}), 400
    print(f"Received message: {message}")
    response_message = answer_as_chatbot(message)
    print(f"Response message: {response_message}")
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="Chatbot")

if __name__ == "__main__":
    app.run()
