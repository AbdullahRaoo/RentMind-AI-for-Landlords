import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .ai_utils import to_native
import importlib.util
import sys
import os

# Dynamically import chatbot_integration from AI Assistant
ai_assistant_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../AI Assistant/chatbot_integration.py'))
spec = importlib.util.spec_from_file_location("chatbot_integration", ai_assistant_path)
chatbot_integration = importlib.util.module_from_spec(spec)
sys.modules["chatbot_integration"] = chatbot_integration
spec.loader.exec_module(chatbot_integration)

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
        # Handle alert fetch request
        if data.get('type') == 'get_alerts':
            alerts = chatbot_integration.MaintenancePredictionHandler().batch_alerts(as_json=True)
            await self.send(text_data=json.dumps({
                'type': 'alerts',
                'alerts': alerts
            }))
            return
        # Handle follow-up actions (save/compare/both)
        if data.get('type') == 'followup' and data.get('action') in ['save', 'compare', 'both']:
            handler = chatbot_integration.RentPredictionHandler()
            response = handler.handle_followup(data['action'], self.last_rent_prediction)
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
            handler = chatbot_integration.RentPredictionHandler()
            response = handler.handle_followup(followup_map[user_message.strip().lower()], self.last_rent_prediction)
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
        result = chatbot_integration.conversational_engine(
            self.conversation_history,
            user_message,
            last_candidate_fields=self.candidate_fields,
            last_intent=user_intent,  # <-- Pass user intent if provided
            intent_completed=False
        )
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
