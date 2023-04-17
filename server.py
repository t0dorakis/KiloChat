from flask import Flask, request, jsonify
from chat import answer_question

app = Flask(__name__)

# allow CORS


@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers',
                       'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods',
                       'GET,PUT,POST,DELETE,OPTIONS')
  return response

# Content_type must be application/json
# POST body must be {"question": "your question here"}
# Response will be {"answer": "your answer here"}
# Example:
# POST http://localhost:5000/chat
# Content-Type: application/json
# Body: {"question": "What is the meaning of life?"}
# Response: {"answer": "42"}


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/chat", methods=['GET', 'POST'])
def chat():
  # get post body
  content = request.json
  question = content.get('question')
  print("content", content)
  response = answer_question(question)
  return response
