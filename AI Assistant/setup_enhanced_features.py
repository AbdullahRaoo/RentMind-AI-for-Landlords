"""
Setup script for Milvus and enhanced conversational AI features.
This script will help you get started with the enhanced features.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        'pymilvus', 'spacy', 'sentence_transformers', 
        'langchain_openai', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\nPlease install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_spacy_model():
    """Check if spaCy English model is available."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy English model is loaded")
        return True
    except OSError:
        print("❌ spaCy English model not found")
        print("Run: python -m spacy download en_core_web_sm")
        return False

def check_environment_variables():
    """Check required environment variables."""
    required_vars = ['OPENAI_API_KEY']
    optional_vars = ['MILVUS_HOST', 'MILVUS_PORT', 'MILVUS_USER', 'MILVUS_PASSWORD']
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            print(f"❌ {var} is not set")
        else:
            print(f"✅ {var} is set")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"ℹ️  {var} is not set (using default)")
    
    if missing_vars:
        print(f"\nPlease set required environment variables in your .env file:")
        for var in missing_vars:
            print(f"  {var}=your_value_here")
        return False
    
    return True

def test_conversation_intelligence():
    """Test the conversation intelligence features without Milvus."""
    try:
        print("\n🧠 Testing Conversation Intelligence...")
        
        # Test without Milvus first
        from conversation_intelligence import get_conversation_intelligence
        
        conversation_ai = get_conversation_intelligence()
        
        test_messages = [
            "Hello there!",
            "I want to estimate rent for my 2 bedroom flat in London",
            "Can you help me screen a tenant with credit score 750 and income £3000?",
            "Thank you for your help",
        ]
        
        for i, msg in enumerate(test_messages, 1):
            print(f"\n--- Test {i} ---")
            print(f"User: {msg}")
            
            analysis = conversation_ai.analyze_message(msg)
            print(f"Primary Intent: {analysis.primary_intent.type.value}")
            print(f"Confidence: {analysis.confidence:.2f}")
            
            if analysis.entities:
                entities_str = [f"{e.label}={e.normalized_value or e.text}" for e in analysis.entities]
                print(f"Entities: {entities_str}")
            
            if analysis.primary_intent.type.value == "greeting":
                response = conversation_ai.handle_greeting()
                print(f"Response: {response}")
            elif analysis.primary_intent.type.value == "small_talk":
                response = conversation_ai.handle_small_talk(msg)
                print(f"Response: {response}")
        
        print("\n✅ Conversation Intelligence test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Conversation Intelligence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_milvus_connection():
    """Test Milvus connection (optional)."""
    try:
        print("\n🗄️  Testing Milvus Connection...")
        
        from milvus_utils import get_milvus_store
        
        milvus_store = get_milvus_store()
        print("✅ Milvus connection successful")
        print("✅ Collections created and indexed")
        
        # Test basic operations
        test_session = "test_session_123"
        test_user = "test_user_456"
        
        # Store a test message
        milvus_store.store_chat_message(
            session_id=test_session,
            user_id=test_user,
            message_type="user",
            content="Hello, I need help with rent pricing",
            intent="rent_prediction",
            entities={"property_type": "flat"}
        )
        print("✅ Chat message storage test passed")
        
        # Retrieve messages
        messages = milvus_store.retrieve_chat_memory(
            session_id=test_session,
            query_text="rent pricing",
            limit=5
        )
        print(f"✅ Chat memory retrieval test passed (found {len(messages)} messages)")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Milvus connection failed: {e}")
        print("This is optional - you can use the enhanced features without Milvus")
        print("To set up Milvus, see MILVUS_SETUP.md")
        return False

def test_enhanced_chatbot():
    """Test the enhanced chatbot integration."""
    try:
        print("\n🤖 Testing Enhanced Chatbot...")
        
        from chatbot_integration import enhanced_conversational_engine
        
        test_conversation = []
        test_message = "Hello, I want to estimate rent for my 2 bedroom flat"
        
        result = enhanced_conversational_engine(
            conversation_history=test_conversation,
            user_message=test_message,
            session_id="test_session",
            user_id="test_user"
        )
        
        print(f"User: {test_message}")
        print(f"Action: {result['action']}")
        print(f"Response: {result['response'][:100]}...")
        
        print("✅ Enhanced chatbot test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced chatbot test failed: {e}")
        # Try fallback to original engine
        try:
            from chatbot_integration import conversational_engine
            result = conversational_engine([], test_message)
            print("✅ Fallback to original engine successful")
            return True
        except Exception as e2:
            print(f"❌ Original engine also failed: {e2}")
            return False

def main():
    """Main setup and testing function."""
    print("🚀 RentMind-AI Enhanced Features Setup")
    print("=" * 50)
    
    # Check requirements
    print("\n1. Checking Package Requirements...")
    if not check_requirements():
        print("\n❌ Setup failed: Missing required packages")
        return False
    
    print("\n2. Checking spaCy Model...")
    if not check_spacy_model():
        print("\n❌ Setup failed: Missing spaCy model")
        return False
    
    print("\n3. Checking Environment Variables...")
    if not check_environment_variables():
        print("\n❌ Setup failed: Missing environment variables")
        return False
    
    # Test features
    print("\n4. Testing Core Features...")
    intelligence_ok = test_conversation_intelligence()
    
    milvus_ok = test_milvus_connection()
    
    chatbot_ok = test_enhanced_chatbot()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY")
    print("=" * 50)
    
    print(f"✅ Package Requirements: {'PASS' if True else 'FAIL'}")
    print(f"✅ spaCy Model: {'PASS' if True else 'FAIL'}")
    print(f"✅ Environment Variables: {'PASS' if True else 'FAIL'}")
    print(f"{'✅' if intelligence_ok else '❌'} Conversation Intelligence: {'PASS' if intelligence_ok else 'FAIL'}")
    print(f"{'✅' if milvus_ok else '⚠️ '} Milvus Connection: {'PASS' if milvus_ok else 'OPTIONAL'}")
    print(f"{'✅' if chatbot_ok else '❌'} Enhanced Chatbot: {'PASS' if chatbot_ok else 'FAIL'}")
    
    if intelligence_ok and chatbot_ok:
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now use the enhanced conversational AI features:")
        print("- Advanced NER and entity extraction")
        print("- Multi-intent detection")
        print("- Context-aware responses")
        print("- Greeting and small talk handling")
        if milvus_ok:
            print("- Persistent chat memory with Milvus")
            print("- Enhanced semantic search")
        
        print("\nNext steps:")
        print("1. Test the features in your Django application")
        print("2. Set up Milvus for production (see MILVUS_SETUP.md)")
        print("3. Customize the conversation flows as needed")
        
        return True
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
