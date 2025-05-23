import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .ai_utils import process_property_message

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        msg_type = data.get('type', 'text')
        if msg_type == 'property':
            # Expecting property data for rent prediction
            input_data = data.get('input_data', {})
            try:
                result = process_property_message(input_data)
                await self.send(text_data=json.dumps({
                    'type': 'property_response',
                    **result
                }))
            except Exception as e:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'error': str(e)
                }))
        else:
            message = data.get('message', '')
            await self.send(text_data=json.dumps({
                'type': 'echo',
                'message': f"Echo: {message}"
            }))
