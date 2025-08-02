#!/usr/bin/env python3
"""
Quick test script for enhanced greeting detection
"""

from chatbot_integration import handle_conversation

def test_greetings():
    print("🚀 Testing Enhanced Greeting Detection")
    print("=" * 50)
    
    test_cases = [
        ("Hi", "Should detect as greeting"),
        ("Hello there!", "Should detect as greeting"),
        ("Good morning", "Should detect as greeting"),
        ("Thanks for your help", "Should detect as greeting"),
        ("I want to estimate rent", "Should detect as business intent"),
        ("Can you screen a tenant?", "Should detect as business intent"),
    ]
    
    for message, description in test_cases:
        try:
            result = handle_conversation([], message)
            action = result.get('action', 'unknown')
            response = result.get('response', 'No response')[:100]
            
            print(f"\n💬 User: \"{message}\"")
            print(f"📝 {description}")
            print(f"🎯 Action: {action}")
            print(f"🤖 Response: {response}...")
            
            if message.lower() in ['hi', 'hello there!', 'good morning', 'thanks for your help']:
                status = "✅ PASS" if action == 'greeting' else "❌ FAIL"
            else:
                status = "✅ PASS" if action != 'greeting' else "❌ FAIL"
            
            print(f"📊 Status: {status}")
            
        except Exception as e:
            print(f"❌ Error testing '{message}': {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Enhanced conversational AI testing complete!")

if __name__ == "__main__":
    test_greetings()
