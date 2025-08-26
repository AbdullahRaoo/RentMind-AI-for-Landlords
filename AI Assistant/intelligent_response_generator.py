"""
Intelligent Response Generation System
=====================================

Features:
- Natural, conversational responses
- Context-aware communication
- Proactive assistance
- Personality and tone consistency
"""

import random
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ResponseTemplate:
    """Template for generating natural responses"""
    template: str
    variants: List[str]
    context_requirements: List[str]
    tone: str  # 'friendly', 'professional', 'helpful', 'encouraging'

class IntelligentResponseGenerator:
    """
    Generates natural, context-aware responses like ChatGPT/Claude
    """
    
    def __init__(self):
        # Response templates organized by intent and situation
        self.templates = {
            'greeting': {
                'first_time': ResponseTemplate(
                    template="Hello! I'm LandlordBuddy, your AI assistant for property management. I can help you with rent pricing, tenant screening, and maintenance predictions. What would you like to explore today?",
                    variants=[
                        "Hi there! I'm LandlordBuddy, here to help with all your property management needs. Whether it's pricing rent, screening tenants, or predicting maintenance - I've got you covered. What can I help you with?",
                        "Welcome! I'm LandlordBuddy, your intelligent property management assistant. I specialize in rent estimation, tenant evaluation, and maintenance planning. How can I assist you today?",
                        "Hello! Great to meet you. I'm LandlordBuddy, and I'm here to make property management easier. I can help estimate rents, screen tenants, and predict maintenance needs. Where shall we start?"
                    ],
                    context_requirements=[],
                    tone='friendly'
                ),
                'returning': ResponseTemplate(
                    template="Welcome back! Ready to tackle another property management task?",
                    variants=[
                        "Great to see you again! What can I help you with today?",
                        "Hello again! How can I assist with your property management needs?",
                        "Welcome back! What would you like to work on today?"
                    ],
                    context_requirements=['previous_interactions'],
                    tone='friendly'
                ),
                'time_aware': ResponseTemplate(
                    template="{time_greeting}! I'm LandlordBuddy, ready to help with your property management needs.",
                    variants=[
                        "{time_greeting}! Hope you're having a great {time_period}. What can I help you with?",
                        "{time_greeting}! Ready to tackle some property management tasks?",
                        "{time_greeting}! I'm here to help make your property management easier."
                    ],
                    context_requirements=['time_context'],
                    tone='friendly'
                )
            },
            
            'rent_prediction_start': {
                'enthusiastic': ResponseTemplate(
                    template="Excellent! I'd love to help you determine the right rent for your property. Let me gather some details to give you an accurate estimate.",
                    variants=[
                        "Perfect! Let's figure out the optimal rent for your property. I'll need a few details to provide you with a precise estimate.",
                        "Great choice! Rent pricing is one of my specialties. Let me ask you a few questions to get you an accurate rental estimate.",
                        "Wonderful! I'll help you price your property competitively. Just need some key details to work with."
                    ],
                    context_requirements=[],
                    tone='enthusiastic'
                ),
                'direct': ResponseTemplate(
                    template="I can help you estimate the rent for your property. What details can you share about the property?",
                    variants=[
                        "I'll help you with rent pricing. Tell me about your property - location, size, type, etc.",
                        "Let's get you a rent estimate. What can you tell me about the property?",
                        "I can provide a rental estimate. Share some property details with me."
                    ],
                    context_requirements=[],
                    tone='professional'
                )
            },
            
            'information_gathering': {
                'encouraging': ResponseTemplate(
                    template="Great start! I have {gathered_info}. {missing_info_request}",
                    variants=[
                        "Perfect! I've got {gathered_info}. Now I just need {missing_info_request}",
                        "Excellent! With {gathered_info}, I can work with that. Could you also share {missing_info_request}?",
                        "That's helpful! I have {gathered_info}. To give you the best estimate, I'd also like to know {missing_info_request}"
                    ],
                    context_requirements=['gathered_info', 'missing_info'],
                    tone='encouraging'
                ),
                'smart_defaults': ResponseTemplate(
                    template="I can work with what you've provided! I'll use smart defaults for the missing details and give you an estimate. You can always provide more specifics for a more precise result.",
                    variants=[
                        "No problem! I can estimate using intelligent defaults for any missing information. Let me run the analysis with what we have.",
                        "That's enough to get started! I'll fill in reasonable assumptions for missing details and provide you with an estimate.",
                        "Perfect! I can work with that information. I'll use typical values for anything missing and give you a solid estimate."
                    ],
                    context_requirements=['partial_info'],
                    tone='helpful'
                )
            },
            
            'results_presentation': {
                'confident': ResponseTemplate(
                    template="Here's your analysis! Based on the information provided, {main_result}. {additional_insights}",
                    variants=[
                        "I've crunched the numbers! {main_result}. {additional_insights}",
                        "Analysis complete! {main_result}. {additional_insights}",
                        "Here are your results! {main_result}. {additional_insights}"
                    ],
                    context_requirements=['main_result', 'insights'],
                    tone='confident'
                ),
                'detailed': ResponseTemplate(
                    template="Here's a comprehensive analysis of your request:\n\n{detailed_breakdown}\n\n{recommendations}",
                    variants=[
                        "I've completed a thorough analysis:\n\n{detailed_breakdown}\n\n{recommendations}",
                        "Here's what my analysis reveals:\n\n{detailed_breakdown}\n\n{recommendations}"
                    ],
                    context_requirements=['detailed_breakdown', 'recommendations'],
                    tone='professional'
                )
            },
            
            'clarification': {
                'gentle': ResponseTemplate(
                    template="I want to make sure I understand correctly. Are you looking to {assumed_intent}?",
                    variants=[
                        "Just to clarify - you'd like me to help you {assumed_intent}, is that right?",
                        "Let me make sure I've got this right. You're interested in {assumed_intent}?",
                        "I want to be sure I'm helping with the right thing. Are you looking for {assumed_intent}?"
                    ],
                    context_requirements=['assumed_intent'],
                    tone='gentle'
                ),
                'options': ResponseTemplate(
                    template="I can help with several things. Are you interested in:\n\n{option_list}",
                    variants=[
                        "Let me offer a few options. Would you like help with:\n\n{option_list}",
                        "I can assist with various tasks. Are you looking for:\n\n{option_list}"
                    ],
                    context_requirements=['option_list'],
                    tone='helpful'
                )
            },
            
            'error_recovery': {
                'understanding': ResponseTemplate(
                    template="I apologize, but I'm having trouble understanding your request. Could you rephrase it or let me know which of these areas you're interested in: {available_options}",
                    variants=[
                        "I'm not quite sure what you're asking for. Could you clarify or choose from: {available_options}",
                        "I want to help, but I need a bit more clarity. Are you interested in: {available_options}",
                        "Let me help you better. Could you specify if you need: {available_options}"
                    ],
                    context_requirements=['available_options'],
                    tone='understanding'
                )
            },
            
            'follow_up': {
                'proactive': ResponseTemplate(
                    template="Would you like me to {suggested_action}? I can also {alternative_action}.",
                    variants=[
                        "I can {suggested_action} if you'd like. Or perhaps {alternative_action}?",
                        "Next, I could {suggested_action}. Would that be helpful? I can also {alternative_action}.",
                        "Would it be useful if I {suggested_action}? Alternatively, I can {alternative_action}."
                    ],
                    context_requirements=['suggested_action', 'alternative_action'],
                    tone='proactive'
                )
            }
        }
        
        # Contextual phrases for natural flow
        self.transition_phrases = {
            'acknowledgment': [
                "I see", "Got it", "Perfect", "Understood", "That makes sense",
                "I understand", "Right", "Okay", "I see what you mean"
            ],
            'enthusiasm': [
                "Excellent!", "Great!", "Perfect!", "Wonderful!", "That's great!",
                "Fantastic!", "Awesome!", "Love it!", "Brilliant!"
            ],
            'encouragement': [
                "You're on the right track", "That's a good approach", "Smart thinking",
                "That's exactly what I need", "Perfect information"
            ],
            'empathy': [
                "I understand that can be challenging", "That's a common concern",
                "I see why that's important to you", "That makes perfect sense"
            ]
        }
        
        # Personality traits for consistent tone
        self.personality = {
            'helpful': True,
            'professional': True,
            'friendly': True,
            'proactive': True,
            'encouraging': True,
            'clear': True
        }
    
    def generate_response(self, intent: str, context: Dict[str, Any], 
                         situation: str = 'default') -> str:
        """
        Generate a natural, context-aware response
        """
        # Get appropriate template
        template_key = f"{intent}_{situation}" if f"{intent}_{situation}" in self.templates else intent
        
        if template_key not in self.templates:
            return self._generate_fallback_response(intent, context)
        
        template_options = self.templates[template_key]
        
        # Select appropriate template variant
        selected_template = self._select_template_variant(template_options, context)
        
        # Generate response with context
        response = self._populate_template(selected_template, context)
        
        # Add natural flow elements
        response = self._enhance_natural_flow(response, context)
        
        return response
    
    def _select_template_variant(self, template_options: Dict[str, ResponseTemplate], 
                                context: Dict[str, Any]) -> ResponseTemplate:
        """Select the most appropriate template variant"""
        
        # Check if we have multiple template styles
        if isinstance(template_options, dict):
            # Select based on context or user preference
            user_tone = context.get('preferred_tone', 'friendly')
            
            # Default selection logic
            if context.get('is_returning_user'):
                preferred_styles = ['enthusiastic', 'friendly', 'encouraging']
            elif context.get('is_complex_request'):
                preferred_styles = ['detailed', 'professional', 'confident']
            else:
                preferred_styles = ['friendly', 'enthusiastic', 'encouraging']
            
            # Find matching style
            for style in preferred_styles:
                if style in template_options:
                    return template_options[style]
            
            # Fallback to first available
            return list(template_options.values())[0]
        
        return template_options
    
    def _populate_template(self, template: ResponseTemplate, context: Dict[str, Any]) -> str:
        """Populate template with context-specific content"""
        
        # Choose variant or use main template
        if template.variants and random.choice([True, False]):  # 50% chance to use variant
            content = random.choice(template.variants)
        else:
            content = template.template
        
        # Replace placeholders with context values
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in content:
                content = content.replace(placeholder, str(value))
        
        # Handle special placeholders
        content = self._handle_special_placeholders(content, context)
        
        return content
    
    def _handle_special_placeholders(self, content: str, context: Dict[str, Any]) -> str:
        """Handle special placeholders like time greetings"""
        
        # Time-based greetings
        if '{time_greeting}' in content:
            hour = datetime.now().hour
            if hour < 12:
                greeting = "Good morning"
                period = "morning"
            elif hour < 17:
                greeting = "Good afternoon"
                period = "afternoon"
            else:
                greeting = "Good evening"
                period = "evening"
            
            content = content.replace('{time_greeting}', greeting)
            content = content.replace('{time_period}', period)
        
        # Dynamic information formatting
        if '{gathered_info}' in content:
            gathered = context.get('gathered_fields', [])
            if gathered:
                info_text = self._format_gathered_info(gathered)
                content = content.replace('{gathered_info}', info_text)
        
        if '{missing_info_request}' in content:
            missing = context.get('missing_fields', [])
            if missing:
                request_text = self._format_missing_info_request(missing)
                content = content.replace('{missing_info_request}', request_text)
        
        return content
    
    def _format_gathered_info(self, gathered_fields: List[str]) -> str:
        """Format gathered information naturally"""
        if len(gathered_fields) == 1:
            return f"the {gathered_fields[0]}"
        elif len(gathered_fields) == 2:
            return f"the {gathered_fields[0]} and {gathered_fields[1]}"
        else:
            return f"the {', '.join(gathered_fields[:-1])}, and {gathered_fields[-1]}"
    
    def _format_missing_info_request(self, missing_fields: List[str]) -> str:
        """Format missing information request naturally"""
        field_names = {
            'address': 'property location',
            'bedrooms': 'number of bedrooms',
            'bathrooms': 'number of bathrooms',
            'size': 'property size',
            'property_type': 'property type',
            'credit_score': 'credit score',
            'income': 'monthly income',
            'employment_status': 'employment situation',
            'age_years': 'property age',
            'last_service_years_ago': 'last maintenance date'
        }
        
        formatted_fields = [field_names.get(field, field.replace('_', ' ')) for field in missing_fields]
        
        if len(formatted_fields) == 1:
            return f"the {formatted_fields[0]}"
        elif len(formatted_fields) == 2:
            return f"the {formatted_fields[0]} and {formatted_fields[1]}"
        else:
            return f"the {', '.join(formatted_fields[:-1])}, and {formatted_fields[-1]}"
    
    def _enhance_natural_flow(self, response: str, context: Dict[str, Any]) -> str:
        """Add natural flow elements to response"""
        
        # Add acknowledgment phrases for follow-up responses
        if context.get('is_follow_up') and not response.startswith(tuple(self.transition_phrases['acknowledgment'])):
            acknowledgment = random.choice(self.transition_phrases['acknowledgment'])
            response = f"{acknowledgment}! {response}"
        
        # Add encouraging elements for partial information
        if context.get('has_partial_info') and 'perfect' not in response.lower():
            encouragement = random.choice(self.transition_phrases['encouragement'])
            # Insert after first sentence
            sentences = response.split('. ')
            if len(sentences) > 1:
                sentences.insert(1, encouragement)
                response = '. '.join(sentences)
        
        return response
    
    def _generate_fallback_response(self, intent: str, context: Dict[str, Any]) -> str:
        """Generate a helpful fallback response"""
        
        base_responses = {
            'rent_prediction': "I'd be happy to help you estimate the rent for your property. Could you share some details about it?",
            'tenant_screening': "I can help you evaluate a tenant application. What information do you have about the applicant?",
            'maintenance_prediction': "I can predict maintenance needs for your property. Tell me about the property and its maintenance history.",
            'help': "I'm here to help! I can assist with rent pricing, tenant screening, and maintenance predictions. What would you like to know more about?",
            'greeting': "Hello! I'm LandlordBuddy, your AI assistant for property management. How can I help you today?"
        }
        
        return base_responses.get(intent, 
            "I'd like to help you with that. Could you provide a bit more detail about what you're looking for?")
    
    def generate_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """Generate contextual suggestions for user"""
        
        current_intent = context.get('current_intent')
        gathered_fields = context.get('gathered_fields', [])
        
        if current_intent == 'rent_prediction':
            if not gathered_fields:
                return [
                    "I have a 2-bedroom flat in Central London",
                    "What's the rent for a house in Manchester?",
                    "Property is 800 sq ft with 1 bedroom"
                ]
            else:
                return [
                    "That's all the details I have",
                    "Let me add more information", 
                    "Can you estimate with what I've provided?"
                ]
        
        elif current_intent == 'tenant_screening':
            if not gathered_fields:
                return [
                    "Credit score is 720, income £3000/month",
                    "Applicant is employed, no evictions",
                    "Need to screen someone for £1200 rent"
                ]
            else:
                return [
                    "Screen with current information",
                    "Add more details about applicant",
                    "What else do you need to know?"
                ]
        
        elif current_intent == 'maintenance_prediction':
            if not gathered_fields:
                return [
                    "Property is 15 years old in London",
                    "Last maintenance was 2 years ago",
                    "Check maintenance for my flat"
                ]
            else:
                return [
                    "Predict with current details",
                    "Add property age information",
                    "Include maintenance history"
                ]
        
        # Default suggestions
        return [
            "Help me price rent for my property",
            "Screen a tenant application",
            "Check maintenance needs"
        ]

# Response quality enhancement
class ResponseEnhancer:
    """Enhance responses with personality and context"""
    
    def __init__(self):
        self.emoji_map = {
            'rent_prediction': '🏠',
            'tenant_screening': '👥', 
            'maintenance_prediction': '🔧',
            'greeting': '👋',
            'help': '💡',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
        
        self.formatting_rules = {
            'highlight_numbers': r'\b(\d+(?:\.\d+)?)\b',
            'highlight_money': r'(£\d+(?:,\d{3})*(?:\.\d{2})?)',
            'highlight_percentages': r'(\d+(?:\.\d+)?%)'
        }
    
    def enhance_response(self, response: str, intent: str, context: Dict[str, Any]) -> str:
        """Enhance response with formatting and personality"""
        
        # Add appropriate emoji if context suggests it
        if context.get('add_emoji', True) and intent in self.emoji_map:
            if not response.startswith(self.emoji_map[intent]):
                response = f"{self.emoji_map[intent]} {response}"
        
        # Format important information
        response = self._apply_formatting(response)
        
        # Add closing suggestions if appropriate
        if context.get('add_suggestions', False):
            suggestions = context.get('suggestions', [])
            if suggestions:
                suggestion_text = "\n\n**What would you like to do next?**\n"
                for i, suggestion in enumerate(suggestions[:3], 1):
                    suggestion_text += f"{i}. {suggestion}\n"
                response += suggestion_text
        
        return response
    
    def _apply_formatting(self, response: str) -> str:
        """Apply formatting rules to response"""
        
        # Highlight important numbers (rent amounts, scores, etc.)
        response = re.sub(
            self.formatting_rules['highlight_money'],
            r'**\1**',
            response
        )
        
        return response

# Export classes
__all__ = [
    'IntelligentResponseGenerator',
    'ResponseEnhancer',
    'ResponseTemplate'
]
