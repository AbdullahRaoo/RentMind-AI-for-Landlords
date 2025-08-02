from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import importlib.util
import sys
import os

# Dynamically import chatbot_integration from AI Assistant (support both local and Docker paths)
ai_assistant_paths = [
    '/app/AI_Assistant/chatbot_integration.py',  # Docker path
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../AI Assistant/chatbot_integration.py'))   # Local path
]

chatbot_integration = None
for ai_assistant_path in ai_assistant_paths:
    if os.path.exists(ai_assistant_path):
        spec = importlib.util.spec_from_file_location("chatbot_integration", ai_assistant_path)
        chatbot_integration = importlib.util.module_from_spec(spec)
        sys.modules["chatbot_integration"] = chatbot_integration
        spec.loader.exec_module(chatbot_integration)
        break

if chatbot_integration is None:
    raise ImportError("Could not find chatbot_integration.py in any expected location")

@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    """
    HTTP API endpoint for chat functionality.
    This provides compatibility for HTTP-based frontend clients.
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        conversation_history = data.get('conversation_history', [])
        last_candidate_fields = data.get('fields', {})
        last_intent = data.get('last_intent')
        intent_completed = data.get('intent_completed', False)
        
        if not user_message:
            return JsonResponse({
                'error': 'No message provided'
            }, status=400)
        
        print(f"[DEBUG] HTTP API - User message: {user_message}")
        print(f"[DEBUG] HTTP API - Conversation history length: {len(conversation_history)}")
        
        # Call the conversational engine
        result = chatbot_integration.handle_conversation(
            conversation_history=conversation_history,
            user_message=user_message,
            last_candidate_fields=last_candidate_fields,
            last_intent=last_intent,
            intent_completed=intent_completed
        )
        
        print(f"[DEBUG] HTTP API - Result action: {result.get('action')}")
        
        # Return the result
        return JsonResponse(result)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        print(f"[ERROR] HTTP API error: {e}")
        import traceback
        traceback.print_exc()
        
        return JsonResponse({
            'error': 'Internal server error',
            'response': 'I apologize, but I encountered a technical issue. Please try again in a moment.'
        }, status=500)
