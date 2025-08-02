from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import sys
import os
import traceback
import importlib.util

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
            # GET request - return API info with debug information
            return JsonResponse({
                'message': 'LandlordBuddy Chat API - Full Debug Version',
                'status': 'active',
                'methods': ['GET', 'POST'],
                'usage': 'POST with JSON: {"message": "your message"}',
                'debug_info': {
                    'python_path': sys.path[:5],  # First 5 paths only
                    'working_directory': os.getcwd(),
                    'ai_assistant_exists': os.path.exists('/app/AI_Assistant/chatbot_integration.py')
                }
            })
        
        # Parse request data for POST
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
        except json.JSONDecodeError as e:
            return JsonResponse({
                'error': f'Invalid JSON in request body: {str(e)}'
            }, status=400)
        
        if not user_message:
            return JsonResponse({
                'error': 'No message provided'
            }, status=400)
        
        # Try to import and use the chatbot_integration module
        try:
            # Dynamic import with detailed error reporting
            ai_assistant_path = '/app/AI_Assistant/chatbot_integration.py'
            if not os.path.exists(ai_assistant_path):
                return JsonResponse({
                    'error': 'Chatbot integration module not found',
                    'debug_info': {
                        'searched_path': ai_assistant_path,
                        'directory_contents': os.listdir('/app') if os.path.exists('/app') else 'N/A'
                    }
                }, status=500)
            
            # Import the module
            spec = importlib.util.spec_from_file_location("chatbot_integration", ai_assistant_path)
            chatbot_integration = importlib.util.module_from_spec(spec)
            sys.modules["chatbot_integration"] = chatbot_integration
            spec.loader.exec_module(chatbot_integration)
            
            # Call the handle_conversation function
            result = chatbot_integration.handle_conversation(
                conversation_history=[],
                user_message=user_message,
                last_candidate_fields={},
                last_intent=None,
                intent_completed=False
            )
            
            return JsonResponse(result)
            
        except ImportError as e:
            return JsonResponse({
                'error': 'Failed to import chatbot module',
                'debug_info': {
                    'import_error': str(e),
                    'traceback': traceback.format_exc()
                }
            }, status=500)
        except AttributeError as e:
            return JsonResponse({
                'error': 'Missing function in chatbot module',
                'debug_info': {
                    'attribute_error': str(e),
                    'traceback': traceback.format_exc()
                }
            }, status=500)
        except Exception as e:
            return JsonResponse({
                'error': 'Chatbot execution error',
                'debug_info': {
                    'execution_error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
            }, status=500)
        
    except Exception as e:
        return JsonResponse({
            'error': 'Unexpected server error',
            'debug_info': {
                'unexpected_error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
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
