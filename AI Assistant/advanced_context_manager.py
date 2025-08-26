"""
Advanced Context Management System
=================================

Features:
- Conversation memory and history
- User preference learning
- Context-aware responses
- Efficient memory management
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib

@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    timestamp: datetime
    user_input: str
    intent: str
    extracted_data: Dict[str, Any]
    system_response: str
    confidence: float
    processing_time: float
    context_used: List[str]

@dataclass
class UserPreferences:
    """User preferences and patterns"""
    preferred_tone: str = 'friendly'
    detail_level: str = 'moderate'  # 'brief', 'moderate', 'detailed'
    response_style: str = 'conversational'  # 'direct', 'conversational', 'professional'
    typical_requests: List[str] = None
    preferred_defaults: Dict[str, Any] = None
    timezone: str = 'UTC'
    
    def __post_init__(self):
        if self.typical_requests is None:
            self.typical_requests = []
        if self.preferred_defaults is None:
            self.preferred_defaults = {}

@dataclass
class SessionContext:
    """Current session context"""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    last_activity: datetime
    current_intent: Optional[str]
    active_data: Dict[str, Any]
    conversation_state: str  # 'greeting', 'gathering', 'processing', 'completed'
    turns: List[ConversationTurn]
    preferences: UserPreferences
    
    def __post_init__(self):
        if not self.turns:
            self.turns = []
        if not self.active_data:
            self.active_data = {}

class AdvancedContextManager:
    """
    Manages conversation context, memory, and user preferences
    """
    
    def __init__(self, max_history_per_session: int = 50, 
                 session_timeout_hours: int = 24,
                 max_sessions_per_user: int = 10):
        
        self.max_history_per_session = max_history_per_session
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.max_sessions_per_user = max_sessions_per_user
        
        # Active sessions
        self.active_sessions: Dict[str, SessionContext] = {}
        
        # User data and preferences (in production, this would be a database)
        self.user_data: Dict[str, Dict[str, Any]] = {}
        self.user_preferences: Dict[str, UserPreferences] = {}
        
        # Pattern recognition
        self.common_patterns: Dict[str, int] = defaultdict(int)
        self.intent_transitions: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Caching for efficiency
        self.context_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(minutes=30)
        
        # Analytics
        self.session_analytics = {
            'total_sessions': 0,
            'avg_turns_per_session': 0,
            'common_intents': defaultdict(int),
            'successful_completions': 0
        }
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new conversation session"""
        
        session_id = self._generate_session_id()
        current_time = datetime.now()
        
        # Get or create user preferences
        preferences = self.get_user_preferences(user_id) if user_id else UserPreferences()
        
        session = SessionContext(
            session_id=session_id,
            user_id=user_id,
            start_time=current_time,
            last_activity=current_time,
            current_intent=None,
            active_data={},
            conversation_state='greeting',
            turns=[],
            preferences=preferences
        )
        
        self.active_sessions[session_id] = session
        self.session_analytics['total_sessions'] += 1
        
        # Clean up old sessions for this user
        if user_id:
            self._cleanup_old_user_sessions(user_id)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session context"""
        
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        # Check if session has expired
        if datetime.now() - session.last_activity > self.session_timeout:
            self.close_session(session_id)
            return None
        
        return session
    
    def update_session(self, session_id: str, 
                      user_input: str = None,
                      intent: str = None,
                      extracted_data: Dict[str, Any] = None,
                      system_response: str = None,
                      confidence: float = None,
                      processing_time: float = None) -> bool:
        """Update session with new turn"""
        
        session = self.get_session(session_id)
        if not session:
            return False
        
        current_time = datetime.now()
        session.last_activity = current_time
        
        # Update current intent and active data
        if intent:
            # Track intent transitions
            if session.current_intent and intent != session.current_intent:
                transition = (session.current_intent, intent)
                self.intent_transitions[transition] += 1
            
            session.current_intent = intent
            self.session_analytics['common_intents'][intent] += 1
        
        if extracted_data:
            session.active_data.update(extracted_data)
        
        # Create conversation turn if we have user input
        if user_input:
            turn = ConversationTurn(
                timestamp=current_time,
                user_input=user_input,
                intent=intent or session.current_intent or 'unknown',
                extracted_data=extracted_data or {},
                system_response=system_response or '',
                confidence=confidence or 0.0,
                processing_time=processing_time or 0.0,
                context_used=list(session.active_data.keys())
            )
            
            session.turns.append(turn)
            
            # Maintain history limit
            if len(session.turns) > self.max_history_per_session:
                session.turns = session.turns[-self.max_history_per_session:]
        
        # Update conversation state
        self._update_conversation_state(session)
        
        # Learn from patterns
        self._learn_from_turn(session, user_input, intent)
        
        return True
    
    def get_context_for_response(self, session_id: str) -> Dict[str, Any]:
        """Get relevant context for generating responses"""
        
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Check cache first
        cache_key = f"context_{session_id}_{len(session.turns)}"
        if cache_key in self.context_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < self.cache_ttl:
                return self.context_cache[cache_key]
        
        context = {
            # Session information
            'session_id': session_id,
            'user_id': session.user_id,
            'current_intent': session.current_intent,
            'conversation_state': session.conversation_state,
            'active_data': session.active_data.copy(),
            
            # User preferences
            'preferred_tone': session.preferences.preferred_tone,
            'detail_level': session.preferences.detail_level,
            'response_style': session.preferences.response_style,
            'preferred_defaults': session.preferences.preferred_defaults,
            
            # Conversation history context
            'is_first_interaction': len(session.turns) == 0,
            'is_returning_user': session.user_id is not None and self._is_returning_user(session.user_id),
            'turn_count': len(session.turns),
            'session_duration': (datetime.now() - session.start_time).total_seconds() / 60,  # minutes
            
            # Recent context
            'recent_intents': self._get_recent_intents(session),
            'gathered_fields': list(session.active_data.keys()),
            'missing_fields': self._identify_missing_fields(session),
            
            # Behavioral context
            'typical_user_patterns': self._get_user_patterns(session.user_id) if session.user_id else {},
            'suggested_next_actions': self._suggest_next_actions(session),
            
            # System context
            'confidence_level': self._calculate_session_confidence(session),
            'processing_efficiency': self._calculate_processing_efficiency(session),
            
            # Time context
            'current_time': datetime.now(),
            'session_start': session.start_time,
            'time_of_day': self._get_time_of_day()
        }
        
        # Add specific context based on intent
        if session.current_intent:
            context.update(self._get_intent_specific_context(session))
        
        # Cache the context
        self.context_cache[cache_key] = context
        self.cache_timestamps[cache_key] = datetime.now()
        
        return context
    
    def learn_user_preferences(self, session_id: str, feedback: Dict[str, Any]):
        """Learn from user feedback to improve preferences"""
        
        session = self.get_session(session_id)
        if not session or not session.user_id:
            return
        
        user_id = session.user_id
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreferences()
        
        preferences = self.user_preferences[user_id]
        
        # Update preferences based on feedback
        if 'tone_feedback' in feedback:
            if feedback['tone_feedback'] == 'too_formal':
                preferences.preferred_tone = 'friendly'
            elif feedback['tone_feedback'] == 'too_casual':
                preferences.preferred_tone = 'professional'
        
        if 'detail_feedback' in feedback:
            if feedback['detail_feedback'] == 'too_brief':
                preferences.detail_level = 'detailed'
            elif feedback['detail_feedback'] == 'too_detailed':
                preferences.detail_level = 'brief'
        
        if 'response_style_feedback' in feedback:
            preferences.response_style = feedback['response_style_feedback']
        
        # Learn default preferences from successful interactions
        if feedback.get('successful_completion'):
            current_data = session.active_data
            for field, value in current_data.items():
                if field not in preferences.preferred_defaults:
                    preferences.preferred_defaults[field] = value
        
        # Update session preferences
        session.preferences = preferences
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation for handoffs or analysis"""
        
        session = self.get_session(session_id)
        if not session:
            return {}
        
        return {
            'session_id': session_id,
            'user_id': session.user_id,
            'duration_minutes': (datetime.now() - session.start_time).total_seconds() / 60,
            'total_turns': len(session.turns),
            'intents_covered': list(set(turn.intent for turn in session.turns)),
            'data_collected': session.active_data,
            'current_state': session.conversation_state,
            'avg_confidence': sum(turn.confidence for turn in session.turns) / len(session.turns) if session.turns else 0,
            'key_moments': self._identify_key_moments(session),
            'completion_status': self._assess_completion_status(session)
        }
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """Close session and return summary"""
        
        session = self.active_sessions.get(session_id)
        if not session:
            return {}
        
        summary = self.get_conversation_summary(session_id)
        
        # Update analytics
        if summary['completion_status'] == 'completed':
            self.session_analytics['successful_completions'] += 1
        
        # Update user data if applicable
        if session.user_id:
            self._update_user_data(session)
        
        # Archive or clean up session
        del self.active_sessions[session_id]
        
        # Clean up related caches
        cache_keys_to_remove = [k for k in self.context_cache.keys() if session_id in k]
        for key in cache_keys_to_remove:
            del self.context_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
        
        return summary
    
    def get_user_preferences(self, user_id: str) -> UserPreferences:
        """Get or create user preferences"""
        
        if user_id not in self.user_preferences:
            # Check if we have historical data for this user
            historical_prefs = self._analyze_user_history(user_id)
            self.user_preferences[user_id] = historical_prefs or UserPreferences()
        
        return self.user_preferences[user_id]
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get system analytics"""
        
        active_session_count = len(self.active_sessions)
        total_turns = sum(len(session.turns) for session in self.active_sessions.values())
        
        if self.session_analytics['total_sessions'] > 0:
            avg_turns = total_turns / self.session_analytics['total_sessions']
        else:
            avg_turns = 0
        
        return {
            'active_sessions': active_session_count,
            'total_sessions': self.session_analytics['total_sessions'],
            'avg_turns_per_session': avg_turns,
            'successful_completion_rate': (
                self.session_analytics['successful_completions'] / 
                max(1, self.session_analytics['total_sessions'])
            ) * 100,
            'common_intents': dict(self.session_analytics['common_intents']),
            'common_intent_transitions': dict(self.intent_transitions),
            'cache_efficiency': len(self.context_cache),
            'memory_usage': self._estimate_memory_usage()
        }
    
    # Private helper methods
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = str(int(time.time() * 1000))
        hash_input = f"{timestamp}_{len(self.active_sessions)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _update_conversation_state(self, session: SessionContext):
        """Update conversation state based on current context"""
        
        if len(session.turns) == 0:
            session.conversation_state = 'greeting'
        elif session.current_intent and len(session.active_data) == 0:
            session.conversation_state = 'gathering'
        elif session.current_intent and len(session.active_data) > 0:
            session.conversation_state = 'processing'
        elif self._has_completed_task(session):
            session.conversation_state = 'completed'
        else:
            session.conversation_state = 'gathering'
    
    def _has_completed_task(self, session: SessionContext) -> bool:
        """Check if current task appears completed"""
        
        required_fields = {
            'rent_prediction': ['address'],
            'tenant_screening': ['credit_score'],
            'maintenance_prediction': ['age_years']
        }
        
        if session.current_intent in required_fields:
            required = required_fields[session.current_intent]
            return any(field in session.active_data for field in required)
        
        return False
    
    def _learn_from_turn(self, session: SessionContext, user_input: str, intent: str):
        """Learn patterns from conversation turn"""
        
        if user_input and intent:
            # Track common input patterns
            input_pattern = self._extract_pattern(user_input)
            self.common_patterns[f"{intent}:{input_pattern}"] += 1
            
            # Learn user-specific patterns
            if session.user_id:
                user_pattern_key = f"{session.user_id}:{intent}"
                self.common_patterns[user_pattern_key] += 1
    
    def _extract_pattern(self, user_input: str) -> str:
        """Extract pattern from user input for learning"""
        
        # Simple pattern extraction (can be enhanced with NLP)
        words = user_input.lower().split()
        
        # Look for key pattern indicators
        if any(word in words for word in ['how', 'what', 'can', 'help']):
            return 'question'
        elif any(word in words for word in ['i', 'my', 'have']):
            return 'statement'
        elif any(word in words for word in ['please', 'want', 'need']):
            return 'request'
        else:
            return 'general'
    
    def _get_recent_intents(self, session: SessionContext) -> List[str]:
        """Get recent intents from conversation"""
        
        recent_turns = session.turns[-5:]  # Last 5 turns
        return [turn.intent for turn in recent_turns if turn.intent != 'unknown']
    
    def _identify_missing_fields(self, session: SessionContext) -> List[str]:
        """Identify missing fields for current intent"""
        
        intent_requirements = {
            'rent_prediction': ['address', 'bedrooms', 'property_type'],
            'tenant_screening': ['credit_score', 'income', 'employment_status'],
            'maintenance_prediction': ['age_years', 'last_service_years_ago', 'location']
        }
        
        if session.current_intent in intent_requirements:
            required = intent_requirements[session.current_intent]
            return [field for field in required if field not in session.active_data]
        
        return []
    
    def _get_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Get learned patterns for user"""
        
        if not user_id:
            return {}
        
        user_patterns = {}
        for pattern, count in self.common_patterns.items():
            if pattern.startswith(f"{user_id}:"):
                intent = pattern.split(':', 1)[1]
                user_patterns[intent] = count
        
        return user_patterns
    
    def _suggest_next_actions(self, session: SessionContext) -> List[str]:
        """Suggest next actions based on context"""
        
        suggestions = []
        
        if session.current_intent == 'rent_prediction':
            if 'address' not in session.active_data:
                suggestions.append('provide_property_location')
            elif len(session.active_data) >= 2:
                suggestions.append('calculate_rent_estimate')
            else:
                suggestions.append('add_property_details')
        
        elif session.current_intent == 'tenant_screening':
            if 'credit_score' not in session.active_data:
                suggestions.append('provide_credit_score')
            elif len(session.active_data) >= 2:
                suggestions.append('screen_tenant')
            else:
                suggestions.append('add_tenant_details')
        
        elif session.current_intent == 'maintenance_prediction':
            if 'age_years' not in session.active_data:
                suggestions.append('provide_property_age')
            elif len(session.active_data) >= 2:
                suggestions.append('predict_maintenance')
            else:
                suggestions.append('add_maintenance_history')
        
        return suggestions
    
    def _calculate_session_confidence(self, session: SessionContext) -> float:
        """Calculate confidence level for session"""
        
        if not session.turns:
            return 0.0
        
        avg_confidence = sum(turn.confidence for turn in session.turns) / len(session.turns)
        data_completeness = len(session.active_data) / max(1, len(self._identify_missing_fields(session)) + len(session.active_data))
        
        return (avg_confidence + data_completeness) / 2
    
    def _calculate_processing_efficiency(self, session: SessionContext) -> float:
        """Calculate processing efficiency"""
        
        if not session.turns:
            return 1.0
        
        avg_processing_time = sum(turn.processing_time for turn in session.turns) / len(session.turns)
        turn_efficiency = min(1.0, 2.0 / max(0.1, avg_processing_time))  # Target: under 2 seconds
        
        return turn_efficiency
    
    def _get_time_of_day(self) -> str:
        """Get time of day context"""
        
        hour = datetime.now().hour
        if hour < 6:
            return 'early_morning'
        elif hour < 12:
            return 'morning'
        elif hour < 17:
            return 'afternoon'
        elif hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def _get_intent_specific_context(self, session: SessionContext) -> Dict[str, Any]:
        """Get context specific to current intent"""
        
        context = {}
        
        if session.current_intent == 'rent_prediction':
            context['has_location'] = 'address' in session.active_data
            context['has_property_details'] = any(k in session.active_data for k in ['bedrooms', 'bathrooms', 'size'])
            context['estimate_readiness'] = len(session.active_data) >= 1
        
        elif session.current_intent == 'tenant_screening':
            context['has_financial_info'] = any(k in session.active_data for k in ['credit_score', 'income'])
            context['has_background_info'] = any(k in session.active_data for k in ['employment_status', 'rental_history'])
            context['screening_readiness'] = len(session.active_data) >= 2
        
        elif session.current_intent == 'maintenance_prediction':
            context['has_property_info'] = any(k in session.active_data for k in ['age_years', 'location'])
            context['has_maintenance_history'] = 'last_service_years_ago' in session.active_data
            context['prediction_readiness'] = len(session.active_data) >= 1
        
        return context
    
    def _is_returning_user(self, user_id: str) -> bool:
        """Check if user has previous sessions"""
        return user_id in self.user_data
    
    def _cleanup_old_user_sessions(self, user_id: str):
        """Clean up old sessions for user"""
        
        user_sessions = [(sid, session) for sid, session in self.active_sessions.items() 
                        if session.user_id == user_id]
        
        if len(user_sessions) > self.max_sessions_per_user:
            # Sort by last activity and remove oldest
            user_sessions.sort(key=lambda x: x[1].last_activity)
            sessions_to_remove = user_sessions[:-self.max_sessions_per_user]
            
            for session_id, _ in sessions_to_remove:
                self.close_session(session_id)
    
    def _analyze_user_history(self, user_id: str) -> Optional[UserPreferences]:
        """Analyze user history to infer preferences"""
        
        # This would analyze historical data in production
        # For now, return None to use defaults
        return None
    
    def _identify_key_moments(self, session: SessionContext) -> List[Dict[str, Any]]:
        """Identify key moments in conversation"""
        
        key_moments = []
        
        for i, turn in enumerate(session.turns):
            # High confidence responses
            if turn.confidence > 0.9:
                key_moments.append({
                    'turn': i + 1,
                    'type': 'high_confidence',
                    'description': 'High confidence response'
                })
            
            # Intent changes
            if i > 0 and turn.intent != session.turns[i-1].intent:
                key_moments.append({
                    'turn': i + 1,
                    'type': 'intent_change',
                    'description': f'Intent changed from {session.turns[i-1].intent} to {turn.intent}'
                })
            
            # Data breakthroughs (when enough data gathered)
            if len(turn.extracted_data) > 0:
                key_moments.append({
                    'turn': i + 1,
                    'type': 'data_gathered',
                    'description': f'Gathered: {list(turn.extracted_data.keys())}'
                })
        
        return key_moments
    
    def _assess_completion_status(self, session: SessionContext) -> str:
        """Assess if conversation completed successfully"""
        
        if session.conversation_state == 'completed':
            return 'completed'
        elif session.current_intent and len(session.active_data) > 0:
            return 'partially_completed'
        elif len(session.turns) > 10:
            return 'extended'
        else:
            return 'in_progress'
    
    def _update_user_data(self, session: SessionContext):
        """Update user data with session information"""
        
        user_id = session.user_id
        if not user_id:
            return
        
        if user_id not in self.user_data:
            self.user_data[user_id] = {
                'first_seen': session.start_time,
                'total_sessions': 0,
                'completed_tasks': [],
                'common_intents': defaultdict(int)
            }
        
        user_data = self.user_data[user_id]
        user_data['total_sessions'] += 1
        user_data['last_seen'] = session.last_activity
        
        # Track completed tasks
        if session.conversation_state == 'completed':
            task_completion = {
                'intent': session.current_intent,
                'date': session.start_time,
                'turns': len(session.turns),
                'data_fields': list(session.active_data.keys())
            }
            user_data['completed_tasks'].append(task_completion)
        
        # Track intent preferences
        for turn in session.turns:
            user_data['common_intents'][turn.intent] += 1
    
    def _estimate_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage for monitoring"""
        
        import sys
        
        sessions_size = sum(sys.getsizeof(session) for session in self.active_sessions.values())
        cache_size = sum(sys.getsizeof(item) for item in self.context_cache.values())
        user_data_size = sum(sys.getsizeof(data) for data in self.user_data.values())
        
        return {
            'active_sessions_mb': sessions_size / (1024 * 1024),
            'cache_mb': cache_size / (1024 * 1024),
            'user_data_mb': user_data_size / (1024 * 1024),
            'total_mb': (sessions_size + cache_size + user_data_size) / (1024 * 1024)
        }

# Export
__all__ = [
    'AdvancedContextManager',
    'SessionContext',
    'UserPreferences',
    'ConversationTurn'
]
