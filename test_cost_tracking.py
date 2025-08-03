#!/usr/bin/env python
"""
Test script to verify OpenAI API cost tracking functionality
"""

import os
import sys
sys.path.append('/home/abdullah/Deploy/rentmind/AI Assistant')

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from cost_tracker import track_langchain_response, get_session_summary
import time

def test_cost_tracking():
    """Test the cost tracking functionality"""
    
    # Test with a simple API call
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    print("🔍 Testing OpenAI API Cost Tracking...")
    print("=" * 50)
    
    # Create a ChatOpenAI instance
    chat = ChatOpenAI(
        model="gpt-4", 
        temperature=0.3, 
        openai_api_key=openai_api_key
    )
    
    # Make a test API call
    test_message = "What is 2+2? Give a very brief answer."
    start_time = time.time()
    
    try:
        response = chat.invoke([HumanMessage(content=test_message)])
        duration = time.time() - start_time
        
        print(f"✅ API Response: {response.content}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        # Track the response
        track_langchain_response(response, "gpt-4", "Cost Tracking Test", duration)
        
        print("\n📊 Session Summary:")
        print("-" * 30)
        print(get_session_summary())
        
    except Exception as e:
        print(f"❌ Error making API call: {e}")

if __name__ == "__main__":
    test_cost_tracking()
