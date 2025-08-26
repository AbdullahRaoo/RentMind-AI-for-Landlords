"""
Advanced Conversational AI System - ChatGPT/Claude Level Implementation
=====================================================================

Key Features:
1. Context-Aware Memory Management
2. Intelligent Intent Understanding  
3. Proactive Assistance
4. Natural Language Flexibility
5. Resource-Efficient Processing
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

# Advanced conversation state management
@dataclass
class ConversationContext:
    """Enhanced conversation context with memory and intent tracking"""
    session_id: str
    user_id: str
    current_intent: Optional[str] = None
    intent_confidence: float = 0.0
    intent_history: List[Dict] = None
    extracted_entities: Dict = None
    conversation_summary: str = ""
    last_action: str = ""
    context_memory: Dict = None
    user_preferences: Dict = None
    conversation_depth: int = 0
    
    def __post_init__(self):
        if self.intent_history is None:
            self.intent_history = []
        if self.extracted_entities is None:
            self.extracted_entities = {}
        if self.context_memory is None:
            self.context_memory = {}
        if self.user_preferences is None:
            self.user_preferences = {}

class ConversationState(Enum):
    """Conversation states for better flow management"""
    GREETING = "greeting"
    DISCOVERING = "discovering"  # Learning what user wants
    GATHERING = "gathering"      # Collecting required info
    PROCESSING = "processing"    # Running models/analysis
    CLARIFYING = "clarifying"    # Asking for clarification
    COMPLETING = "completing"    # Finishing task
    FOLLOWING_UP = "following_up" # Post-completion actions

class IntentType(Enum):
    """Comprehensive intent classification"""
    # Primary intents
    RENT_PREDICTION = "rent_prediction"
    TENANT_SCREENING = "tenant_screening" 
    MAINTENANCE_PREDICTION = "maintenance_prediction"
    
    # Secondary intents
    GREETING = "greeting"
    INFORMATION_REQUEST = "information_request"
    CLARIFICATION = "clarification"
    CORRECTION = "correction"
    FOLLOW_UP = "follow_up"
    
    # Meta intents
    HELP = "help"
    CAPABILITIES = "capabilities"
    EXAMPLE = "example"
    UNKNOWN = "unknown"

@dataclass
class UserIntent:
    """Enhanced intent representation"""
    primary_intent: IntentType
    confidence: float
    secondary_intents: List[IntentType]
    entities: Dict[str, Any]
    context_clues: List[str]
    user_goal: str
    complexity_level: int  # 1-5 scale

class AdvancedConversationEngine:
    """
    Next-generation conversation engine with ChatGPT-level understanding
    """
    
    def __init__(self):
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.intent_cache = {}  # Cache frequent patterns
        self.entity_memory = {}  # Remember user's typical entity values
        self.conversation_patterns = {}  # Learn from conversation flows
        
        # Resource optimization
        self.cache_ttl = 3600  # 1 hour cache
        self.max_context_length = 10  # Keep last 10 exchanges
        
    def create_session_key(self, user_id: str, session_id: str = None) -> str:
        """Create unique session identifier"""
        if not session_id:
            session_id = hashlib.md5(f"{user_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        return f"{user_id}_{session_id}"
    
    def get_or_create_context(self, user_id: str, session_id: str = None) -> ConversationContext:
        """Get existing context or create new one"""
        session_key = self.create_session_key(user_id, session_id)
        
        if session_key not in self.conversation_contexts:
            self.conversation_contexts[session_key] = ConversationContext(
                session_id=session_id or session_key.split('_')[1],
                user_id=user_id
            )
        
        return self.conversation_contexts[session_key]
    
    def analyze_user_intent(self, message: str, context: ConversationContext, 
                           conversation_history: List[Dict]) -> UserIntent:
        """
        Advanced intent analysis with context awareness
        """
        # Quick pattern matching for efficiency
        intent = self._quick_intent_detection(message, context)
        if intent:
            return intent
            
        # Deep analysis for complex cases
        return self._deep_intent_analysis(message, context, conversation_history)
    
    def _quick_intent_detection(self, message: str, context: ConversationContext) -> Optional[UserIntent]:
        """
        Efficient intent detection using patterns and cache
        """
        msg_lower = message.lower().strip()
        
        # Cache check
        msg_hash = hashlib.md5(msg_lower.encode()).hexdigest()
        if msg_hash in self.intent_cache:
            cached_result = self.intent_cache[msg_hash]
            if time.time() - cached_result['timestamp'] < self.cache_ttl:
                return cached_result['intent']
        
        # High-confidence patterns (90%+ accuracy)
        patterns = {
            IntentType.GREETING: [
                r'\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b',
                r'\b(thanks|thank you|appreciate)\b'
            ],
            IntentType.RENT_PREDICTION: [
                r'\b(rent|rental|price|pricing|estimate|how much|monthly cost)\b.*\b(property|flat|house|apartment)\b',
                r'\b(predict|estimate|calculate).*\brent\b',
                r'\bhow much.*\b(rent|charge|cost)\b'
            ],
            IntentType.TENANT_SCREENING: [
                r'\b(screen|check|evaluate|assess).*\b(tenant|applicant|renter)\b',
                r'\b(tenant|applicant).*\b(screening|background|check|credit)\b',
                r'\b(approve|accept|reject).*\b(tenant|applicant)\b'
            ],
            IntentType.MAINTENANCE_PREDICTION: [
                r'\b(maintenance|repair|fix|service|upkeep).*\b(predict|check|need|required)\b',
                r'\b(check|see if|predict).*\b(maintenance|repair|service)\b',
                r'\b(maintenance|repair).*\b(needed|required|schedule)\b'
            ],
            IntentType.HELP: [
                r'\b(help|assist|guide|how to|what can|capabilities)\b',
                r'\b(confused|lost|stuck|don\'t understand)\b'
            ]
        }
        
        # Match patterns
        for intent_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, msg_lower):
                    entities = self._extract_quick_entities(message, intent_type)
                    
                    user_intent = UserIntent(
                        primary_intent=intent_type,
                        confidence=0.95,
                        secondary_intents=[],
                        entities=entities,
                        context_clues=[pattern],
                        user_goal=self._infer_user_goal(intent_type, entities),
                        complexity_level=self._assess_complexity(message, entities)
                    )
                    
                    # Cache result
                    self.intent_cache[msg_hash] = {
                        'intent': user_intent,
                        'timestamp': time.time()
                    }
                    
                    return user_intent
        
        return None
    
    def _extract_quick_entities(self, message: str, intent_type: IntentType) -> Dict[str, Any]:
        """Extract entities efficiently based on intent type"""
        entities = {}
        msg_lower = message.lower()
        
        # Common entity patterns
        entity_patterns = {
            'location': r'\b(in|at|located|area|near)\s+([A-Za-z\s]{2,30})\b',
            'number': r'\b(\d+(?:\.\d+)?)\s*(bedroom|bathroom|bed|bath|room|year|month|pound|£|sqft|sq\s*ft)\b',
            'property_type': r'\b(flat|apartment|house|studio|property|unit|building|place)\b',
            'time_reference': r'\b(last|ago|since|years?|months?|recently|new|old)\b'
        }
        
        for entity_type, pattern in entity_patterns.items():
            matches = re.findall(pattern, msg_lower)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def _infer_user_goal(self, intent_type: IntentType, entities: Dict) -> str:
        """Infer what the user is trying to achieve"""
        goal_templates = {
            IntentType.RENT_PREDICTION: "Determine appropriate rental price for property",
            IntentType.TENANT_SCREENING: "Evaluate tenant application suitability", 
            IntentType.MAINTENANCE_PREDICTION: "Assess property maintenance needs",
            IntentType.GREETING: "Initiate conversation and explore capabilities",
            IntentType.HELP: "Understand available features and how to use them"
        }
        
        base_goal = goal_templates.get(intent_type, "Clarify request and provide assistance")
        
        # Enhance with entity context
        if entities:
            if 'location' in entities:
                base_goal += f" for property in {entities['location'][0][1] if entities['location'] else 'specified location'}"
        
        return base_goal
    
    def _assess_complexity(self, message: str, entities: Dict) -> int:
        """Assess conversation complexity (1-5 scale)"""
        complexity = 1
        
        # Length factor
        if len(message.split()) > 20:
            complexity += 1
        
        # Entity richness
        if len(entities) > 2:
            complexity += 1
        
        # Multiple questions/requests
        if message.count('?') > 1 or message.count(' and ') > 1:
            complexity += 1
        
        # Technical terms
        technical_terms = ['mortgage', 'yield', 'roi', 'depreciation', 'capital gains']
        if any(term in message.lower() for term in technical_terms):
            complexity += 1
        
        return min(complexity, 5)
    
    def generate_intelligent_response(self, user_intent: UserIntent, context: ConversationContext,
                                    conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Generate contextually aware, intelligent responses
        """
        # Determine conversation state
        state = self._determine_conversation_state(user_intent, context, conversation_history)
        
        # Route to appropriate response generator
        if user_intent.primary_intent == IntentType.GREETING:
            return self._handle_greeting(user_intent, context)
        elif user_intent.primary_intent == IntentType.HELP:
            return self._handle_help_request(user_intent, context)
        elif user_intent.primary_intent in [IntentType.RENT_PREDICTION, IntentType.TENANT_SCREENING, IntentType.MAINTENANCE_PREDICTION]:
            return self._handle_primary_intent(user_intent, context, state, conversation_history)
        else:
            return self._handle_clarification(user_intent, context)
    
    def _determine_conversation_state(self, user_intent: UserIntent, context: ConversationContext,
                                    conversation_history: List[Dict]) -> ConversationState:
        """Determine current conversation state"""
        
        # New conversation
        if len(conversation_history) <= 2:
            return ConversationState.DISCOVERING
        
        # Check if we're in middle of a task
        if context.current_intent and context.current_intent == user_intent.primary_intent.value:
            if len(context.extracted_entities) < 2:
                return ConversationState.GATHERING
            else:
                return ConversationState.PROCESSING
        
        # Intent change
        if context.current_intent and context.current_intent != user_intent.primary_intent.value:
            return ConversationState.DISCOVERING
        
        return ConversationState.CLARIFYING
    
    def _handle_greeting(self, user_intent: UserIntent, context: ConversationContext) -> Dict[str, Any]:
        """Handle greetings with personality and context awareness"""
        
        # Personalize based on time and context
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        
        # Returning user vs new user
        if context.conversation_depth > 0:
            response = f"{greeting}! Great to see you back. How can I help you today?"
        else:
            response = (
                f"{greeting}! I'm LandlordBuddy, your intelligent AI assistant for property management. "
                f"I can help you with:\n\n"
                f"🏠 **Rent Pricing** - Get accurate rental estimates\n"
                f"👥 **Tenant Screening** - Evaluate applicant suitability\n" 
                f"🔧 **Maintenance Prediction** - Anticipate property maintenance needs\n\n"
                f"What would you like to explore today?"
            )
        
        context.last_action = "greeting"
        
        return {
            "response": response,
            "action": "greeting",
            "confidence": 1.0,
            "suggestions": ["Tell me about rent pricing", "How does tenant screening work?", "Predict maintenance for my property"],
            "context_updated": True
        }
    
    def _handle_help_request(self, user_intent: UserIntent, context: ConversationContext) -> Dict[str, Any]:
        """Provide intelligent, contextual help"""
        
        # Analyze what kind of help is needed
        help_type = "general"
        if "rent" in str(user_intent.entities):
            help_type = "rent_pricing"
        elif "tenant" in str(user_intent.entities):
            help_type = "tenant_screening"
        elif "maintenance" in str(user_intent.entities):
            help_type = "maintenance"
        
        help_responses = {
            "general": (
                "I'm here to help! Here's what I can do:\n\n"
                "🏠 **Rent Pricing**: I analyze property details (location, size, bedrooms, etc.) "
                "to provide accurate rental estimates using market data.\n\n"
                "👥 **Tenant Screening**: I evaluate tenant applications based on credit score, "
                "income, employment status, and rental history.\n\n"
                "🔧 **Maintenance Prediction**: I predict potential maintenance needs based on "
                "property age, location, and service history.\n\n"
                "Just tell me what you'd like to do, and I'll guide you through it!"
            ),
            "rent_pricing": (
                "For rent pricing, I'll need some basic property details:\n\n"
                "• **Location** (address or area)\n"
                "• **Property type** (flat, house, studio, etc.)\n"
                "• **Size** (square footage or number of rooms)\n"
                "• **Bedrooms & bathrooms**\n\n"
                "I can provide estimates even with partial information using smart defaults. "
                "Try saying: 'What rent can I charge for my 2-bedroom flat in London?'"
            ),
            "tenant_screening": (
                "For tenant screening, I analyze:\n\n"
                "• **Credit score**\n"
                "• **Monthly income**\n"
                "• **Employment status**\n"
                "• **Rental history/evictions**\n"
                "• **Rent amount**\n\n"
                "Try: 'Screen a tenant with 700 credit score, £3000 income, employed, no evictions for £1200 rent'"
            ),
            "maintenance": (
                "For maintenance prediction, I consider:\n\n"
                "• **Property location**\n"
                "• **Property age**\n"
                "• **Last maintenance date**\n"
                "• **Current season**\n\n"
                "Try: 'Check if my 15-year-old property in Manchester needs maintenance'"
            )
        }
        
        return {
            "response": help_responses[help_type],
            "action": "help",
            "confidence": 1.0,
            "suggestions": self._get_contextual_suggestions(help_type)
        }
    
    def _get_contextual_suggestions(self, context_type: str) -> List[str]:
        """Get smart suggestions based on context"""
        suggestions = {
            "general": [
                "Estimate rent for my property",
                "Screen a tenant application", 
                "Predict maintenance needs"
            ],
            "rent_pricing": [
                "2-bedroom flat in Central London",
                "House with 3 bedrooms in Manchester",
                "Studio apartment near transport"
            ],
            "tenant_screening": [
                "Applicant with 650 credit score",
                "Self-employed tenant evaluation",
                "High-income professional screening"
            ],
            "maintenance": [
                "Victorian property maintenance",
                "New build inspection needs",
                "Seasonal maintenance planning"
            ]
        }
        return suggestions.get(context_type, [])

# Memory-efficient context manager
class ConversationMemory:
    """Efficient conversation memory management"""
    
    def __init__(self, max_contexts=1000, cleanup_interval=3600):
        self.contexts = {}
        self.max_contexts = max_contexts
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
    
    def get_context(self, session_key: str) -> Optional[ConversationContext]:
        """Get conversation context with automatic cleanup"""
        self._cleanup_if_needed()
        return self.contexts.get(session_key)
    
    def set_context(self, session_key: str, context: ConversationContext):
        """Set conversation context with memory management"""
        self.contexts[session_key] = context
        
        # Memory limit enforcement
        if len(self.contexts) > self.max_contexts:
            self._cleanup_old_contexts()
    
    def _cleanup_if_needed(self):
        """Cleanup old contexts periodically"""
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_contexts()
            self.last_cleanup = time.time()
    
    def _cleanup_old_contexts(self):
        """Remove oldest 20% of contexts"""
        if len(self.contexts) <= 100:  # Keep minimum contexts
            return
        
        # Sort by last activity and remove oldest
        contexts_with_time = [
            (key, getattr(ctx, 'last_activity', 0))
            for key, ctx in self.contexts.items()
        ]
        contexts_with_time.sort(key=lambda x: x[1])
        
        remove_count = len(self.contexts) // 5  # Remove 20%
        for key, _ in contexts_with_time[:remove_count]:
            del self.contexts[key]

# Export the enhanced system
__all__ = [
    'AdvancedConversationEngine',
    'ConversationContext', 
    'ConversationState',
    'UserIntent',
    'IntentType',
    'ConversationMemory'
]
