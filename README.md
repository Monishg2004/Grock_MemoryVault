# MemoryVault - AI-Powered Memory Companion for Alzheimer's Patients

## Inspiration
Alzheimer's disease affects millions of people worldwide, gradually erasing their precious memories and connections to their past. We created MemoryVault to help patients reconnect with their memories through an innovative AI-powered platform that makes remembering an interactive and engaging experience.

## What it does
MemoryVault is an AI-powered memory preservation and retrieval system that helps Alzheimer's patients and their families:
- Store and organize personal memories in a secure digital vault
- Retrieve memories through natural conversation
- Experience memories visually through AI-generated imagery
- Engage with past experiences in an interactive, emotionally connected way
- Share and preserve family histories across generations

## How we built it
MemoryVault combines several cutting-edge technologies to create a seamless memory preservation and retrieval system:

### Backend Architecture (Flask + Python)
- **Flask Server**: RESTful API endpoints for memory storage and retrieval
- **Vector Embeddings**: Using SentenceTransformer for converting memories into semantic vectors
- **Pinecone Integration**: Vector database for efficient memory storage and retrieval
- **Google Gemini Pro**: Advanced language model for natural conversation and memory narration
- **FLUX API**: AI image generation for visual memory representation
- **Error Handling**: Comprehensive logging and error management system

### Core Features
1. **Memory Storage Pipeline**:
   - Text chunking for optimal processing
   - Vector embedding generation
   - Metadata association
   - Pinecone database storage

2. **Memory Retrieval System**:
   - Query vector generation
   - Semantic similarity search
   - Context-aware response generation
   - Image generation for visual representation

3. **API Endpoints**:
   - `/postMemory`: Memory storage endpoint
   - `/query`: Memory retrieval and interaction endpoint

## Challenges we ran into
1. **Vector Database Integration**: Ensuring efficient storage and retrieval of memory embeddings while maintaining semantic relevance
2. **Response Generation**: Creating natural, empathetic responses that accurately reflect stored memories
3. **Image Generation**: Implementing reliable image generation that matches memory contexts
4. **Error Handling**: Building robust error handling for various API integrations

## Accomplishments that we're proud of
1. Created a seamless integration between multiple AI models (Sentence Transformers, Gemini Pro, FLUX)
2. Implemented efficient vector-based memory storage and retrieval
3. Developed a system that maintains high accuracy while being user-friendly
4. Built a scalable architecture that can handle multiple users and memories

## What we learned
- Vector database management for semantic search
- Large Language Model integration and prompt engineering
- Image generation API implementation
- Building production-ready Flask applications
- Error handling in distributed systems

## What's next for MemoryVault
1. **Enhanced Authentication**: Implementing secure user authentication and family account management
2. **Voice Integration**: Adding speech-to-text and text-to-speech capabilities
3. **Advanced Memory Management**: Tools for organizing and categorizing memories
4. **Improved Visualization**: Enhanced image generation and virtual reality experiences
5. **Mobile Application**: Native mobile apps for easier access
6. **Healthcare Integration**: Partnerships with memory care facilities and healthcare providers

## Built With
- Python
- Flask
- Pinecone
- Google Gemini Pro
- Sentence Transformers
- FLUX API
- Docker
- Google Cloud Run
- React
- Vercel

## Try it out
- [Demo Link]
- [GitHub Repository]

## Installation and Local Setup

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env

# Run the application
python app.py
```

## Team
- [Team Member 1] - Backend Development
- [Team Member 2] - Frontend Development
- [Team Member 3] - AI Integration
- [Team Member 4] - Project Management