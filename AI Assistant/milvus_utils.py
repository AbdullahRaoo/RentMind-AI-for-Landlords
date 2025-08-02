"""
Milvus utility functions for vector storage and retrieval.
Handles chat memory, semantic search, and conversation context.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType,
    utility, MilvusException
)
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusVectorStore:
    """
    Milvus-based vector store for chat memory and semantic search.
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: str = "19530",
                 user: str = "",
                 password: str = "",
                 db_name: str = "default"):
        """
        Initialize Milvus connection and embedding model.
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Collection names
        self.chat_collection_name = "chat_memory"
        self.semantic_collection_name = "semantic_search"
        
        # Initialize connection
        self._connect()
        self._setup_collections()
    
    def _connect(self):
        """Connect to Milvus server."""
        try:
            # Check if already connected
            if connections.has_connection("default"):
                connections.remove_connection("default")
            
            # Connect with or without auth
            if self.user and self.password:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    db_name=self.db_name
                )
            else:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    db_name=self.db_name
                )
            
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _setup_collections(self):
        """Create collections if they don't exist."""
        # Chat memory collection schema
        chat_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="message_type", dtype=DataType.VARCHAR, max_length=32),  # 'user' or 'assistant'
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="entities", dtype=DataType.VARCHAR, max_length=1024),  # JSON string
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        
        chat_schema = CollectionSchema(fields=chat_fields, description="Chat memory storage")
        
        # Semantic search collection schema (for property listings)
        semantic_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64),  # 'rent_data', 'maintenance', etc.
            FieldSchema(name="record_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2048),  # JSON string
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        
        semantic_schema = CollectionSchema(fields=semantic_fields, description="Semantic search storage")
        
        # Create collections
        self._create_collection_if_not_exists(self.chat_collection_name, chat_schema)
        self._create_collection_if_not_exists(self.semantic_collection_name, semantic_schema)
        
        # Create indexes
        self._create_indexes()
    
    def _create_collection_if_not_exists(self, name: str, schema: CollectionSchema):
        """Create collection if it doesn't exist."""
        if not utility.has_collection(name):
            Collection(name=name, schema=schema)
            logger.info(f"Created collection: {name}")
        else:
            logger.info(f"Collection already exists: {name}")
    
    def _create_indexes(self):
        """Create vector indexes for efficient search."""
        # Chat memory index
        chat_collection = Collection(self.chat_collection_name)
        if not chat_collection.has_index():
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            chat_collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Created index for {self.chat_collection_name}")
        
        # Semantic search index
        semantic_collection = Collection(self.semantic_collection_name)
        if not semantic_collection.has_index():
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            semantic_collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Created index for {self.semantic_collection_name}")
        
        # Load collections into memory
        chat_collection.load()
        semantic_collection.load()
    
    def store_chat_message(self, 
                          session_id: str,
                          user_id: str,
                          message_type: str,
                          content: str,
                          intent: str = "",
                          entities: Dict = None):
        """
        Store a chat message with its embedding.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            message_type: 'user' or 'assistant'
            content: Message content
            intent: Detected intent
            entities: Extracted entities dict
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Prepare data
            entities_json = json.dumps(entities or {})
            timestamp = int(datetime.now().timestamp())
            
            data = [[
                session_id,
                user_id,
                message_type,
                content,
                intent,
                entities_json,
                timestamp,
                embedding
            ]]
            
            # Insert into collection
            collection = Collection(self.chat_collection_name)
            collection.insert(data)
            collection.flush()
            
            logger.info(f"Stored {message_type} message for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store chat message: {e}")
            raise
    
    def retrieve_chat_memory(self, 
                           session_id: str,
                           query_text: str,
                           limit: int = 5,
                           similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Retrieve relevant chat history based on query similarity.
        
        Args:
            session_id: Session to search within
            query_text: Query to find similar messages
            limit: Maximum number of messages to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant chat messages
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            # Search parameters
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # Search in collection
            collection = Collection(self.chat_collection_name)
            
            # Filter by session_id
            expr = f"session_id == '{session_id}'"
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["session_id", "user_id", "message_type", "content", 
                              "intent", "entities", "timestamp"]
            )
            
            # Process results
            relevant_messages = []
            for hit in results[0]:
                if hit.score >= similarity_threshold:
                    message = {
                        "session_id": hit.entity.get("session_id"),
                        "user_id": hit.entity.get("user_id"),
                        "message_type": hit.entity.get("message_type"),
                        "content": hit.entity.get("content"),
                        "intent": hit.entity.get("intent"),
                        "entities": json.loads(hit.entity.get("entities", "{}")),
                        "timestamp": hit.entity.get("timestamp"),
                        "similarity": hit.score
                    }
                    relevant_messages.append(message)
            
            return relevant_messages
            
        except Exception as e:
            logger.error(f"Failed to retrieve chat memory: {e}")
            return []
    
    def store_semantic_record(self, 
                            source: str,
                            record_id: str,
                            content: str,
                            metadata: Dict = None):
        """
        Store a record for semantic search (property listings, etc.).
        
        Args:
            source: Data source identifier
            record_id: Unique record identifier
            content: Text content to be searchable
            metadata: Additional metadata dict
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Prepare data
            metadata_json = json.dumps(metadata or {})
            
            data = [[
                source,
                record_id,
                content,
                metadata_json,
                embedding
            ]]
            
            # Insert into collection
            collection = Collection(self.semantic_collection_name)
            collection.insert(data)
            collection.flush()
            
            logger.info(f"Stored semantic record: {source}/{record_id}")
            
        except Exception as e:
            logger.error(f"Failed to store semantic record: {e}")
            raise
    
    def semantic_search(self, 
                       query_text: str,
                       source_filter: str = None,
                       limit: int = 5,
                       similarity_threshold: float = 0.6) -> List[Dict]:
        """
        Perform semantic search across stored records.
        
        Args:
            query_text: Search query
            source_filter: Optional source filter
            limit: Maximum results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching records
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            # Search parameters
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # Search in collection
            collection = Collection(self.semantic_collection_name)
            
            # Optional source filter
            expr = f"source == '{source_filter}'" if source_filter else None
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["source", "record_id", "content", "metadata"]
            )
            
            # Process results
            matching_records = []
            for hit in results[0]:
                if hit.score >= similarity_threshold:
                    record = {
                        "source": hit.entity.get("source"),
                        "record_id": hit.entity.get("record_id"),
                        "content": hit.entity.get("content"),
                        "metadata": json.loads(hit.entity.get("metadata", "{}")),
                        "similarity": hit.score
                    }
                    matching_records.append(record)
            
            return matching_records
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    def get_recent_chat_history(self, 
                               session_id: str,
                               limit: int = 10) -> List[Dict]:
        """
        Get recent chat history for a session, ordered by timestamp.
        
        Args:
            session_id: Session identifier
            limit: Maximum messages to return
            
        Returns:
            List of recent messages
        """
        try:
            collection = Collection(self.semantic_collection_name)
            
            # Query recent messages
            expr = f"session_id == '{session_id}'"
            
            # Note: Milvus doesn't support ORDER BY in query, so we'll get more than needed
            # and sort in Python
            results = collection.query(
                expr=expr,
                output_fields=["session_id", "user_id", "message_type", "content", 
                              "intent", "entities", "timestamp"],
                limit=limit * 2  # Get more to ensure we have enough after sorting
            )
            
            # Sort by timestamp descending and limit
            sorted_results = sorted(results, key=lambda x: x["timestamp"], reverse=True)[:limit]
            
            # Convert to proper format
            messages = []
            for result in sorted_results:
                message = {
                    "session_id": result["session_id"],
                    "user_id": result["user_id"],
                    "message_type": result["message_type"],
                    "content": result["content"],
                    "intent": result["intent"],
                    "entities": json.loads(result.get("entities", "{}")),
                    "timestamp": result["timestamp"]
                }
                messages.append(message)
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get recent chat history: {e}")
            return []
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """
        Clean up chat sessions older than specified days.
        
        Args:
            days_old: Delete sessions older than this many days
        """
        try:
            cutoff_timestamp = int((datetime.now().timestamp() - (days_old * 24 * 60 * 60)))
            
            collection = Collection(self.chat_collection_name)
            expr = f"timestamp < {cutoff_timestamp}"
            
            # Delete old records
            collection.delete(expr)
            collection.flush()
            
            logger.info(f"Cleaned up chat sessions older than {days_old} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")

# Global instance
_milvus_store = None

def get_milvus_store() -> MilvusVectorStore:
    """Get or create global Milvus store instance."""
    global _milvus_store
    if _milvus_store is None:
        # Get connection details from environment or use defaults
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        user = os.getenv("MILVUS_USER", "")
        password = os.getenv("MILVUS_PASSWORD", "")
        db_name = os.getenv("MILVUS_DB", "default")
        
        _milvus_store = MilvusVectorStore(host, port, user, password, db_name)
    
    return _milvus_store

def migrate_faiss_to_milvus(faiss_index_path: str, csv_data_path: str, source_name: str):
    """
    Utility function to migrate existing FAISS data to Milvus.
    
    Args:
        faiss_index_path: Path to FAISS index file
        csv_data_path: Path to corresponding CSV data
        source_name: Source identifier for Milvus
    """
    try:
        import pandas as pd
        from faiss_utils import load_faiss_index, record_to_text
        
        # Load FAISS data
        df = pd.read_csv(csv_data_path)
        milvus_store = get_milvus_store()
        
        # Migrate each record
        for idx, row in df.iterrows():
            content = record_to_text(row.to_dict())
            metadata = row.to_dict()
            record_id = f"{source_name}_{idx}"
            
            milvus_store.store_semantic_record(
                source=source_name,
                record_id=record_id,
                content=content,
                metadata=metadata
            )
        
        logger.info(f"Migrated {len(df)} records from FAISS to Milvus")
        
    except Exception as e:
        logger.error(f"Failed to migrate FAISS to Milvus: {e}")
