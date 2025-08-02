from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import importlib.util
import sys
import os

@csrf_exempt
@require_http_methods(["GET", "POST"])
def chat_api(request):
    """
    HTTP API endpoint for chat functionality.
    This provides compatibility for HTTP-based frontend clients.
    """
    try:
        # Handle both GET and POST requests
        if request.method == "GET":
            # GET request - return API info
            return JsonResponse({
                'message': 'LandlordBuddy Chat API',
                'status': 'active',
                'methods': ['GET', 'POST'],
                'usage': 'POST with JSON: {"message": "your message"}',
                'python_path': sys.path,
                'working_directory': os.getcwd()
            })
        
        # Parse request data for POST
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({
                'error': 'No message provided'
            }, status=400)
        
        # Simple echo response for now
        return JsonResponse({
            'response': f'Echo: {user_message}',
            'action': 'chat',
            'fields': {},
            'last_intent': None,
            'intent_completed': True
        })
        
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
            'response': 'I apologize, but I encountered a technical issue. Please try again in a moment.',
            'debug_info': str(e)
        }, status=500)
        
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
