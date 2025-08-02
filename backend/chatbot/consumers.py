import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .ai_utils import to_native
import importlib.util
import sys
import os
import traceback

# Bulletproof chatbot_integration import with fallback
chatbot_integration = None
import_error = None

try:
    # Dynamically import chatbot_integration from AI Assistant (support both local and Docker paths)
    ai_assistant_paths = [
        '/app/AI_Assistant/chatbot_integration.py',  # Docker path
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../AI Assistant/chatbot_integration.py'))   # Local path
    ]

    for ai_assistant_path in ai_assistant_paths:
        if os.path.exists(ai_assistant_path):
            print(f"[DEBUG] Attempting to import from: {ai_assistant_path}")
            spec = importlib.util.spec_from_file_location("chatbot_integration", ai_assistant_path)
            chatbot_integration = importlib.util.module_from_spec(spec)
            sys.modules["chatbot_integration"] = chatbot_integration
            spec.loader.exec_module(chatbot_integration)
            print(f"[DEBUG] Successfully imported chatbot_integration from: {ai_assistant_path}")
            break
    
    if chatbot_integration is None:
        import_error = f"Could not find chatbot_integration.py in any expected location: {ai_assistant_paths}"
        print(f"[WARNING] {import_error}")

except Exception as e:
    import_error = f"Failed to import chatbot_integration: {str(e)}"
    print(f"[ERROR] {import_error}")
    traceback.print_exc()

def get_emergency_response(user_message):
    """Emergency fallback responses that never fail"""
    msg = user_message.strip().lower()
    
    if msg in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
        return {
            "response": (
                "Hello! I'm LandlordBuddy, your AI assistant for property management. "
                "I can help you with rent pricing, tenant screening, and maintenance predictions.\n\n"
                "What would you like to do today?"
            ),
            "action": "greeting",
            "fields": {}
        }
    elif any(word in msg for word in ["rent", "price", "estimate"]):
        return {
            "response": (
                "I can help you with rent estimation! Please provide me with:\n"
                "- Property address\n"
                "- Number of bedrooms\n" 
                "- Number of bathrooms\n"
                "- Property size (in sq ft)\n"
                "- Property type (e.g., apartment, house)"
            ),
            "action": "chat",
            "fields": {}
        }
    elif any(word in msg for word in ["tenant", "screen", "applicant"]):
        return {
            "response": (
                "I can help you screen tenants! Please provide:\n"
                "- Credit score (300-850)\n"
                "- Monthly income\n"
                "- Desired rent amount\n"
                "- Employment status\n"
                "- Eviction history (yes/no)"
            ),
            "action": "chat", 
            "fields": {}
        }
    elif any(word in msg for word in ["maintenance", "repair", "fix"]):
        return {
            "response": (
                "I can help predict maintenance needs! Please tell me:\n"
                "- Property address\n"
                "- Property age (in years)\n"
                "- Years since last maintenance\n"
                "- Current season"
            ),
            "action": "chat",
            "fields": {}
        }
    else:
        return {
            "response": (
                f"Thank you for your message: \"{user_message}\"\n\n"
                "I can help you with:\n"
                "• **Rent Estimation** - Get market rent predictions\n"
                "• **Tenant Screening** - Assess applicant suitability\n"
                "• **Maintenance Prediction** - Predict maintenance needs\n\n"
                "Which would you like help with?"
            ),
            "action": "chat",
            "fields": {}
        }

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.conversation_history = []  # Store conversation as a list of {role, content}
        self.candidate_fields = {}      # Persist candidate fields across turns
        self.last_rent_prediction = None  # Store last rent prediction per session
        await self.accept()

    async def disconnect(self, close_code):
        pass

    def extract_fields_from_markdown(self, message):
        """
        Extract fields from a markdown-formatted summary (assistant's message).
        Looks for lines like '- **Field:** value' and returns a dict.
        """
        import re
        fields = {}
        # Match both '- **Field:** value' and '- Field: value'
        pattern = r"- (?:\*\*)?(.+?)(?:\*\*)?: (.+)"
        for line in message.splitlines():
            match = re.match(pattern, line.strip())
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                fields[key] = value
        return fields

    async def receive(self, text_data):
        data = json.loads(text_data)
        print("[DEBUG] Received data:", data)
        print(f"[DEBUG] WebSocket bulletproof version 2.0 active - chatbot_integration status: {'available' if chatbot_integration else 'fallback_mode'}")
        # Handle alert fetch request
        if data.get('type') == 'get_alerts':
            try:
                if chatbot_integration is not None:
                    alerts = chatbot_integration.MaintenancePredictionHandler().batch_alerts(as_json=True)
                else:
                    alerts = []  # Empty alerts if module not available
            except Exception as e:
                print(f"[ERROR] Failed to get alerts: {e}")
                alerts = []
            
            await self.send(text_data=json.dumps({
                'type': 'alerts',
                'alerts': alerts
            }))
            return
        # Handle follow-up actions (save/compare/both)
        if data.get('type') == 'followup' and data.get('action') in ['save', 'compare', 'both']:
            try:
                if chatbot_integration is not None:
                    handler = chatbot_integration.RentPredictionHandler()
                    response = handler.handle_followup(data['action'], self.last_rent_prediction)
                else:
                    response = "Follow-up actions are temporarily unavailable. Please try again later."
            except Exception as e:
                print(f"[ERROR] Failed to handle followup: {e}")
                response = "Sorry, I couldn't process that request. Please try again."
                
            await self.send(text_data=json.dumps({
                'type': 'bot_response',
                'message': response
            }))
            return
        # Accept both 'message' and 'text' keys for user input
        user_message = data.get('message') or data.get('text') or ''
        user_intent = data.get('intent')  # <-- NEW: get intent from frontend
        print("[DEBUG] User message:", user_message)
        print("[DEBUG] User intent:", user_intent)
        if not user_message:
            await self.send(text_data=json.dumps({
                'type': 'bot_response',
                'message': "Sorry, I didn't get your message. Please try again."
            }))
            return
        # Treat 'compare', 'save', or 'compare & save' as follow-up actions even if sent as text
        followup_map = {
            'compare': 'compare',
            'save': 'save',
            'compare & save': 'both',
            'compare and save': 'both',
        }
        if user_message.strip().lower() in followup_map:
            try:
                if chatbot_integration is not None:
                    handler = chatbot_integration.RentPredictionHandler()
                    response = handler.handle_followup(followup_map[user_message.strip().lower()], self.last_rent_prediction)
                else:
                    response = "Follow-up actions are temporarily unavailable. Please try again later."
            except Exception as e:
                print(f"[ERROR] Failed to handle followup: {e}")
                response = "Sorry, I couldn't process that request. Please try again."
                
            await self.send(text_data=json.dumps({
                'type': 'bot_response',
                'message': response
            }))
            return
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        # Try to extract fields from the last assistant message (if any)
        if self.conversation_history:
            # Look for the last assistant message
            for msg in reversed(self.conversation_history[:-1]):
                if msg["role"] == "assistant":
                    extracted = self.extract_fields_from_markdown(msg["content"])
                    if extracted:
                        self.candidate_fields.update(extracted)
                    break
        # Call the conversational engine with candidate fields and intent
        try:
            if chatbot_integration is None:
                # Use emergency fallback if module failed to import
                print(f"[DEBUG] Using emergency fallback due to import error: {import_error}")
                result = get_emergency_response(user_message)
            else:
                print(f"[DEBUG] Calling conversational engine with message: {user_message}")
                result = chatbot_integration.handle_conversation(
                    conversation_history=self.conversation_history,
                    user_message=user_message,
                    last_candidate_fields=self.candidate_fields,
                    last_intent=user_intent,  # Pass user intent if provided
                    intent_completed=False
                )
                print(f"[DEBUG] Conversational engine result: {result.get('action', 'no_action')}")
        except Exception as e:
            print(f"[ERROR] Conversational engine failed: {e}")
            traceback.print_exc()
            
            # Emergency fallback response
            print(f"[DEBUG] Using emergency fallback due to execution error")
            result = get_emergency_response(user_message)
        print("[DEBUG] LLM result:", result)
        # If the model just returned new fields (e.g., after prediction), update candidate_fields
        if result.get('fields'):
            self.candidate_fields.update(result['fields'])
        # If this is a rent prediction, store the prediction for follow-up
        if result.get('action') in ['screen_tenant', 'rent_prediction'] and result.get('fields'):
            self.last_rent_prediction = dict(result['fields'])
        # Add assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": result['response']})
        await self.send(text_data=json.dumps({
            'type': 'bot_response',
            'message': result['response']
        }))
