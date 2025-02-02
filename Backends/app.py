from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from typing import Dict, Any
import uuid
import requests
from groclake.vectorlake import VectorLake
from groclake.modellake import ModelLake
from langchain.text_splitter import CharacterTextSplitter
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize Grocklake
GROCLAKE_API_KEY = 'a0a080f42e6f13b3a2df133f073095dd'
GROCLAKE_ACCOUNT_ID = '6ecda6849eb84c553fb646f987f6f7db'
os.environ['GROCLAKE_API_KEY'] = GROCLAKE_API_KEY
os.environ['GROCLAKE_ACCOUNT_ID'] = GROCLAKE_ACCOUNT_ID

# Initialize Hugging Face access token
HUGGINGFACE_API_KEY = ''
FLUX_API_URL = ""

# Initialize Grocklake tools
vector_lake = VectorLake()
model_lake = ModelLake()

def generate_image_from_text(text: str) -> str:
    """Generate an image from text description using FLUX API."""
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        # Create a detailed prompt for image generation
        payload = {
            "inputs": f"A detailed, vivid visualization of this memory: {text}",
            "parameters": {
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
                "num_inference_steps": 50
            }
        }
        
        response = requests.post(FLUX_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            # Properly handle binary image data
            image_bytes = response.content
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_image}"
        else:
            logger.error(f"Error from FLUX API: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

def split_text(text: str, metadata: Dict[str, Any] = None):
    """Split text into chunks and prepare for vector storage."""
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    document_id = str(uuid.uuid4())
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = {
            "document_id": document_id,
            "chunk_id": i,
            "text": chunk
        }
        if metadata:
            chunk_metadata.update(metadata)
        processed_chunks.append((chunk, chunk_metadata))
    
    return processed_chunks

def store_memory(text: str, metadata: Dict[str, Any] = None):
    """Store memory text as vectors using Vectorlake."""
    try:
        chunks = split_text(text, metadata)
        
        for chunk_text, chunk_metadata in chunks:
            vector_data = vector_lake.generate(chunk_text)
            vector = vector_data['vector']
            
            payload = {
                "vector": vector,
                "vector_type": "text",
                "vector_document": chunk_text,
                "vectorlake_id": GROCLAKE_ACCOUNT_ID,
                "metadata": chunk_metadata
            }
            
            vector_lake.push(payload)
        
        logger.info(f"Successfully stored memory with document ID: {chunks[0][1]['document_id']}")
        return chunks[0][1]['document_id']
    
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise

@app.route("/postMemory", methods=['GET', 'POST'])
def post_memory():
    """Endpoint to store new memories with both GET and POST support."""
    try:
        if request.method == 'POST':
            data = request.json
            text = data.get("text")
            metadata = data.get("metadata", {})
        else:  # GET method
            text = request.args.get("text")
            metadata = {}
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        document_id = store_memory(text, metadata)
        return jsonify({"success": True, "document_id": document_id}), 200
        
    except Exception as e:
        logger.error(f"Error in post_memory: {e}")
        return jsonify({"error": str(e)}), 500

# Keeping the original store endpoint for backward compatibility
@app.route("/store", methods=['POST'])
def store_memory_endpoint():
    """Alternative endpoint to store new memories (POST only)."""
    return post_memory()

def get_memory_response(query: str):
    """Retrieve and process memories based on the query."""
    try:
        query_vector_data = vector_lake.generate(query)
        query_vector = query_vector_data['vector']
        
        search_payload = {
            "vector": query_vector,
            "vectorlake_id": GROCLAKE_ACCOUNT_ID,
            "query": query,
            "vector_type": "text",
            "top_k": 5
        }
        
        results = vector_lake.search(search_payload)
        
        if not results:
            return {"text": "I don't have any memories to share right now. Please add some memories first."}
        
        chat_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """You are an AI meant to help Alzheimer's patients remember their memories.
                    Be kind, considerate, and detailed in your responses. Make the memories feel vivid and real."""
                },
                {
                    "role": "user",
                    "content": f"""The user is asking: "{query}"
                    
                    Here is the memory to recall: "{results}"
                    
                    Please respond with as much detail as possible, making them feel like they are living the memory again.
                    Respond in second person and make it vivid.
                    Do not mention being an AI or reference the context.
                    Only use the information provided - do not make up any false details."""
                }
            ]
        }
        
        response = model_lake.chat_complete(chat_payload)
        text_response = response.get('answer', '').strip()
        
        # Generate image for the memory
        image_data = generate_image_from_text(text_response)
        
        return {
            "text": text_response,
            "image": image_data
        }
        
    except Exception as e:
        logger.error(f"Error retrieving memory response: {e}")
        return {"text": str(e), "image": None}

@app.route("/getMemory", methods=['GET', 'POST'])
def get_memory():
    """Endpoint to query stored memories with both GET and POST support."""
    try:
        if request.method == 'POST':
            data = request.json
            query = data.get("query")
        else:  # GET method
            query = request.args.get("query")
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        response = get_memory_response(query)
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in get_memory: {e}")
        return jsonify({"error": str(e)}), 500

# Keeping the original query endpoint for backward compatibility
@app.route("/query", methods=['GET', 'POST'])
def query_memory():
    """Alternative endpoint to query memories."""
    return get_memory()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)