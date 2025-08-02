# Milvus Setup Guide for RentMind-AI-for-Landlords

This guide helps you set up Milvus for enhanced chat memory and vector search capabilities.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ with pip
- At least 4GB RAM available for Milvus

## Option 1: Local Milvus with Docker (Recommended for Development)

### 1. Download Milvus Docker Compose

```bash
# Create milvus directory
mkdir milvus-docker && cd milvus-docker

# Download docker-compose.yml
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml -o docker-compose.yml

# Start Milvus
docker-compose up -d
```

### 2. Verify Installation

```bash
# Check if containers are running
docker-compose ps

# You should see:
# - milvus-standalone
# - milvus-minio
# - milvus-etcd
```

Milvus will be available at `localhost:19530`

## Option 2: Milvus Cloud (Recommended for Production)

1. Sign up at [https://cloud.zilliz.com/](https://cloud.zilliz.com/)
2. Create a new cluster
3. Get your connection details:
   - Host/Endpoint
   - Port (usually 19530)
   - Username and Password

## Environment Configuration

Add these variables to your `.env` file:

```env
# Milvus Configuration
MILVUS_HOST=localhost          # or your cloud endpoint
MILVUS_PORT=19530             # default port
MILVUS_USER=                  # leave empty for local, set for cloud
MILVUS_PASSWORD=              # leave empty for local, set for cloud  
MILVUS_DB=default             # database name

# OpenAI API Key (required for embeddings and LLM)
OPENAI_API_KEY=your_openai_api_key_here
```

## Installation Steps

### 1. Install Dependencies

```bash
# Navigate to your project directory
cd RentMind-AI-for-Landlords

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model for NER
python -m spacy download en_core_web_sm
```

### 2. Initialize Milvus Collections

```python
# Run this in Python to set up collections
from AI_Assistant.milvus_utils import get_milvus_store

# This will create necessary collections and indexes
milvus_store = get_milvus_store()
print("✅ Milvus collections initialized successfully!")
```

### 3. Migrate Existing Data (Optional)

If you have existing FAISS data, migrate it to Milvus:

```python
from AI_Assistant.chatbot_integration import migrate_existing_data_to_milvus

migrate_existing_data_to_milvus()
```

## Testing the Setup

Run the test function to verify everything works:

```python
# From the AI Assistant directory
python chatbot_integration.py
```

You should see output like:
```
Testing Enhanced Conversational AI...
User: Hello, I want to estimate rent for my 2 bedroom flat
Primary Intent: rent_prediction
Confidence: 0.85
Entities: ['BEDROOMS=2', 'PROPERTY_TYPE=flat']
✅ Enhanced features test completed successfully!
```

## Usage in Your Application

### Basic Chat with Memory

```python
from AI_Assistant.chatbot_integration import enhanced_conversational_engine

# Use the enhanced engine instead of the original
result = enhanced_conversational_engine(
    conversation_history=[],
    user_message="I want to estimate rent for my 3 bedroom house",
    session_id="user123_session456",  # Unique session ID
    user_id="user123"                 # Unique user ID
)

print(result["response"])
```

### Direct Milvus Operations

```python
from AI_Assistant.milvus_utils import get_milvus_store

milvus_store = get_milvus_store()

# Store a chat message
milvus_store.store_chat_message(
    session_id="session123",
    user_id="user456", 
    message_type="user",
    content="Hello, I need help with rent pricing",
    intent="rent_prediction",
    entities={"property_type": "flat", "bedrooms": "2"}
)

# Retrieve relevant chat history
relevant_messages = milvus_store.retrieve_chat_memory(
    session_id="session123",
    query_text="rent pricing help",
    limit=5
)

# Perform semantic search
results = milvus_store.semantic_search(
    query_text="2 bedroom flat in London",
    source_filter="rent_data",
    limit=5
)
```

## Key Features Enabled

### 1. Advanced NER (Named Entity Recognition)
- Extracts property details, financial information, and addresses
- Uses spaCy + custom patterns for real estate domain
- Handles casual language and synonyms

### 2. Multi-Intent Detection
- Detects multiple user intents in a single message
- Uses both keyword matching and LLM analysis
- Handles greetings, small talk, and clarification requests

### 3. Context-Aware Entity Linking
- Links extracted entities to specific fields
- Resolves ambiguous entities using conversation context
- Maintains entity confidence scores

### 4. Persistent Chat Memory
- Stores conversation history in Milvus
- Retrieves relevant context for better responses
- Supports session-based memory management

### 5. Enhanced Semantic Search
- Replaces FAISS with scalable Milvus
- Supports filtered searches by data source
- Better similarity matching with COSINE metrics

## Troubleshooting

### Milvus Connection Issues
```python
# Test connection
from AI_Assistant.milvus_utils import get_milvus_store

try:
    store = get_milvus_store()
    print("✅ Milvus connection successful")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### spaCy Model Issues
```bash
# Reinstall spaCy model
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl --force-reinstall
```

### Memory/Performance Issues
- Increase Docker memory limit to 8GB+
- Use Milvus Cloud for production workloads
- Consider using IVF_PQ index for large datasets

## Production Considerations

1. **Use Milvus Cloud** for better reliability and scaling
2. **Set up monitoring** for collection sizes and performance
3. **Implement cleanup routines** for old chat sessions
4. **Use Redis** for session state management
5. **Configure proper security** with authentication enabled

## Support

- Milvus Documentation: https://milvus.io/docs
- spaCy Documentation: https://spacy.io/usage
- LangChain Documentation: https://python.langchain.com/

For project-specific issues, check the console logs and ensure all environment variables are properly set.
