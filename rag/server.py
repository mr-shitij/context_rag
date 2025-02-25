from flask import Flask, request, jsonify
from flask_cors import CORS

from rag import RAG

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

rag_system = RAG(
    collection_name="pdf_embeddings",
    raw_dir="../DOCS/raw",
    processed_dir="../DOCS/processed",
)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Simple endpoint that takes a message and returns a response"""

    # Get message from request
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400

    user_message = data['message']

    # Generate a response
    response = generate_response(user_message)

    return jsonify({
        'response': response
    })


def generate_response(message):
    """
    Simple response generator function.
    In a real application, you would integrate with a more sophisticated
    language model or conversation system.
    """

    final_answer = rag_system.query_llm(message)
    return final_answer


if __name__ == '__main__':
    app.run(debug=True)