from flask import Flask, render_template, request, jsonify
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend to call the API

# Load LLaMA 2 Model
llama2_model = OllamaLLM(model='llama2:latest')

# Store chat history
messages = [
    SystemMessage(content="You are a professional Doctor.Help the human by giving them best advice.")
]

# Home route (Loads the frontend)
@app.route('/')
def home():
    return render_template('index.html')


# API route for chatbot
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a valid question."})

    # Append user's message
    messages.append(HumanMessage(content=user_message))

    # Create prompt template
    template = "Question: {question}\n\nAnswer: Let's think step by step."
    prompt = ChatPromptTemplate.from_template(template)

    # Generate response
    chain = prompt | llama2_model
    response = chain.invoke({"question": user_message})

    # Store AI response in history
    messages.append(AIMessage(content=response))

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
