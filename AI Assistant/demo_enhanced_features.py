"""
Demo script to showcase the enhanced conversational AI features.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

load_dotenv()

def demo_conversation_intelligence():
    """Demonstrate advanced NER and intent detection."""
    print("🧠 ADVANCED CONVERSATION INTELLIGENCE DEMO")
    print("=" * 60)
    
    from conversation_intelligence import get_conversation_intelligence
    
    conversation_ai = get_conversation_intelligence()
    
    # Test cases demonstrating various features
    test_cases = [
        {
            "message": "Hello! I'm new here",
            "description": "Greeting Detection"
        },
        {
            "message": "I want to estimate rent for my 3 bedroom house in Manchester, it's about 1200 sq ft",
            "description": "Complex Entity Extraction"
        },
        {
            "message": "Can you help me screen a tenant? Credit score is 680, monthly income £2500, employment is full-time, no prior evictions",
            "description": "Multi-Entity Tenant Screening"
        },
        {
            "message": "I need both rent pricing and tenant screening for my property",
            "description": "Multi-Intent Detection"
        },
        {
            "message": "The property at 123 Oak Street was built 15 years ago, last serviced 3 years ago, it's winter now",
            "description": "Maintenance Prediction with Temporal Info"
        },
        {
            "message": "Thank you so much for your help!",
            "description": "Small Talk & Gratitude"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")
        print(f"💬 User: {test_case['message']}")
        
        # Analyze the message
        analysis = conversation_ai.analyze_message(test_case['message'])
        
        # Display results
        print(f"🎯 Primary Intent: {analysis.primary_intent.type.value} (confidence: {analysis.confidence:.2f})")
        
        if len(analysis.intents) > 1:
            other_intents = [f"{intent.type.value} ({intent.confidence:.2f})" 
                           for intent in analysis.intents[1:]]
            print(f"🔍 Other Intents: {', '.join(other_intents)}")
        
        if analysis.entities:
            print("📝 Extracted Entities:")
            for entity in analysis.entities:
                value = entity.normalized_value if entity.normalized_value is not None else entity.text
                print(f"   • {entity.label}: {value} (confidence: {entity.confidence:.2f})")
        
        # Show appropriate response
        if analysis.primary_intent.type.value == "greeting":
            response = conversation_ai.handle_greeting()
            print(f"🤖 Response: {response}")
        elif analysis.primary_intent.type.value == "small_talk":
            response = conversation_ai.handle_small_talk(test_case['message'])
            print(f"🤖 Response: {response}")
        elif analysis.requires_clarification:
            print(f"❓ Clarification: {analysis.clarification_message}")

def demo_enhanced_chatbot():
    """Demonstrate the enhanced chatbot integration."""
    print("\n\n🤖 ENHANCED CHATBOT INTEGRATION DEMO")
    print("=" * 60)
    
    from chatbot_integration import enhanced_conversational_engine, conversational_engine
    
    conversation_history = []
    session_id = "demo_session_123"
    user_id = "demo_user_456"
    
    demo_conversation = [
        "Hi there!",
        "I want to estimate rent for my property",
        "It's a 2 bedroom flat in London, about 800 sq ft",
        "Yes, that's correct",
        "Thanks! Now I also need to screen a tenant",
        "Credit score 720, income £3200 monthly, employed, no evictions",
        "Yes, please proceed"
    ]
    
    current_fields = {}
    last_intent = None
    intent_completed = False
    
    for i, user_message in enumerate(demo_conversation, 1):
        print(f"\n--- Conversation Turn {i} ---")
        print(f"👤 User: {user_message}")
        
        try:
            # Try enhanced engine first (will fallback if Milvus not available)
            result = enhanced_conversational_engine(
                conversation_history=conversation_history,
                user_message=user_message,
                last_candidate_fields=current_fields,
                last_intent=last_intent,
                intent_completed=intent_completed,
                session_id=session_id,
                user_id=user_id
            )
        except Exception as e:
            print(f"⚠️  Enhanced engine failed, using fallback: {e}")
            result = conversational_engine(
                conversation_history=conversation_history,
                user_message=user_message,
                last_candidate_fields=current_fields,
                last_intent=last_intent,
                intent_completed=intent_completed
            )
        
        # Display results
        print(f"🎯 Action: {result['action']}")
        print(f"🤖 Assistant: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")
        
        if result.get('fields'):
            print(f"📋 Fields Collected: {list(result['fields'].keys())}")
        
        # Update conversation state
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": result['response']})
        current_fields = result.get('fields', {})
        last_intent = result.get('last_intent')
        intent_completed = result.get('intent_completed', False)
        
        # Limit conversation history to prevent it getting too long
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

def demo_comparison():
    """Compare original vs enhanced features."""
    print("\n\n⚖️  FEATURE COMPARISON DEMO")
    print("=" * 60)
    
    test_message = "I want to estimate rent for my 2 bed flat and also screen a tenant with score 750"
    
    print(f"📝 Test Message: {test_message}")
    print("\n--- Original Engine ---")
    
    try:
        from chatbot_integration import conversational_engine
        
        original_result = conversational_engine([], test_message)
        print(f"🎯 Action: {original_result['action']}")
        print(f"📋 Fields: {list(original_result.get('fields', {}).keys())}")
        print(f"🤖 Response: {original_result['response'][:150]}...")
    except Exception as e:
        print(f"❌ Original engine failed: {e}")
    
    print("\n--- Enhanced Engine (without Milvus) ---")
    
    try:
        from conversation_intelligence import get_conversation_intelligence
        
        conversation_ai = get_conversation_intelligence()
        analysis = conversation_ai.analyze_message(test_message)
        
        print(f"🎯 Primary Intent: {analysis.primary_intent.type.value}")
        print(f"🔍 All Intents: {[intent.type.value for intent in analysis.intents]}")
        print(f"📝 Entities: {[f'{e.label}={e.normalized_value or e.text}' for e in analysis.entities]}")
        
        if analysis.requires_clarification:
            print(f"❓ Clarification: {analysis.clarification_message}")
        
    except Exception as e:
        print(f"❌ Enhanced engine failed: {e}")

def main():
    """Run all demos."""
    print("🚀 RENTMIND-AI ENHANCED FEATURES DEMO")
    print("🏠 Intelligent Conversational AI for Landlords")
    print("=" * 80)
    
    try:
        demo_conversation_intelligence()
        demo_enhanced_chatbot()
        demo_comparison()
        
        print("\n\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("\nKey Enhanced Features:")
        print("✅ Advanced NER with real estate domain knowledge")
        print("✅ Multi-intent detection and handling")
        print("✅ Context-aware entity linking")
        print("✅ Greeting and small talk capabilities")
        print("✅ Improved conversation flow management")
        print("✅ Fallback compatibility with original engine")
        print("✅ Optional Milvus integration for memory and search")
        
        print("\nNext Steps:")
        print("1. 🗄️  Set up Milvus for persistent memory (see MILVUS_SETUP.md)")
        print("2. 🔧 Integrate with your Django backend")
        print("3. 🎨 Customize responses and conversation flows")
        print("4. 📊 Add analytics and monitoring")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
