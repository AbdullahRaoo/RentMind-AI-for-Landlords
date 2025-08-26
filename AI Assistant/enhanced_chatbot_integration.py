"""
Seamless Integration Script for Advanced Chatbot Features
========================================================

This script provides a seamless integration of advanced features
into the existing chatbot system without breaking current functionality.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing components  
from chatbot_integration import handle_conversation, extract_information, detect_intent

# Import new advanced components
from advanced_conversation_engine import AdvancedConversationEngine, ConversationContext, ConversationState
from smart_intent_detection import SmartIntentDetector, IntentResult
from intelligent_response_generator import IntelligentResponseGenerator, ResponseEnhancer
from advanced_context_manager import AdvancedContextManager, SessionContext, UserPreferences

class EnhancedChatbotConversation:
    """
    Enhanced chatbot conversation that integrates all advanced features
    while maintaining compatibility with existing system
    """
    
    def __init__(self, debug_mode: bool = False):
        print("🚀 Initializing Enhanced Chatbot System...")
        
        # Store reference to legacy functions
        self.legacy_handle_conversation = handle_conversation
        self.legacy_extract_information = extract_information
        self.legacy_detect_intent = detect_intent
        
        # Initialize advanced components
        self.conversation_engine = AdvancedConversationEngine()
        self.intent_detector = SmartIntentDetector()
        self.response_generator = IntelligentResponseGenerator()
        self.response_enhancer = ResponseEnhancer()
        self.context_manager = AdvancedContextManager()
        
        # Configuration
        self.debug_mode = debug_mode
        self.use_advanced_features = True
        self.fallback_to_legacy = True
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'advanced_success': 0,
            'legacy_fallback': 0,
            'avg_response_time': 0,
            'user_satisfaction': []
        }
        
        print("✅ Enhanced Chatbot System Initialized Successfully!")
        if debug_mode:
            print("🔧 Debug mode enabled - detailed logging active")
    
    def start_conversation(self, user_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        
        try:
            # Create new session
            session_id = self.context_manager.create_session(user_id)
            
            if self.debug_mode:
                print(f"🆕 New session created: {session_id}")
            
            # Generate welcome message
            context = self.context_manager.get_context_for_response(session_id)
            
            if context.get('is_returning_user'):
                response = self.response_generator.generate_response(
                    'greeting', context, 'returning'
                )
            else:
                response = self.response_generator.generate_response(
                    'greeting', context, 'first_time'
                )
            
            # Enhance response
            enhanced_response = self.response_enhancer.enhance_response(
                response, 'greeting', context
            )
            
            # Update session
            self.context_manager.update_session(
                session_id, 
                system_response=enhanced_response,
                confidence=1.0,
                processing_time=0.1
            )
            
            return f"SESSION_ID:{session_id}\n\n{enhanced_response}"
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error in start_conversation: {e}")
            
            # Fallback to legacy system
            return self.legacy_chatbot.process_message("Hello")
    
    def process_message(self, message: str, session_id: str = None) -> str:
        """Process user message with advanced features"""
        
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1
        
        try:
            # Extract session ID if provided in message format
            if session_id is None and message.startswith("SESSION_ID:"):
                lines = message.split('\n', 2)
                if len(lines) >= 3:
                    session_id = lines[0].replace("SESSION_ID:", "")
                    message = lines[2]
            
            # Use advanced processing if session exists
            if session_id and self.use_advanced_features:
                response = self._process_with_advanced_features(message, session_id, start_time)
                if response:
                    self.performance_metrics['advanced_success'] += 1
                    return response
            
            # Fallback to legacy system
            if self.fallback_to_legacy:
                if self.debug_mode:
                    print("🔄 Falling back to legacy system")
                
                self.performance_metrics['legacy_fallback'] += 1
                result = self.legacy_handle_conversation([], message)
                if isinstance(result, dict) and 'response' in result:
                    return result['response']
                return str(result)
            
            return "I'm sorry, I'm having trouble processing your request. Please try again."
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error in process_message: {e}")
            
            # Emergency fallback
            result = self.legacy_handle_conversation([], message)
            if isinstance(result, dict) and 'response' in result:
                return result['response']
            return str(result)
    
    def _process_with_advanced_features(self, message: str, session_id: str, start_time: float) -> Optional[str]:
        """Process message using advanced features"""
        
        try:
            # Get session context
            session = self.context_manager.get_session(session_id)
            if not session:
                if self.debug_mode:
                    print(f"⚠️ Session {session_id} not found")
                return None
            
            # Step 1: Advanced intent detection
            intent_result = self.intent_detector.detect_intent(
                message, 
                session.active_data,
                session.turns[-5:] if session.turns else []
            )
            
            if self.debug_mode:
                print(f"🎯 Intent detected: {intent_result.intent} (confidence: {intent_result.confidence:.2f})")
            
            # Step 2: Extract data using enhanced methods
            extracted_data = self._enhanced_data_extraction(message, intent_result.intent, session)
            
            if self.debug_mode and extracted_data:
                print(f"📊 Data extracted: {extracted_data}")
            
            # Step 3: Update session context
            processing_time = time.time() - start_time
            self.context_manager.update_session(
                session_id,
                user_input=message,
                intent=intent_result.intent,
                extracted_data=extracted_data,
                confidence=intent_result.confidence,
                processing_time=processing_time
            )
            
            # Step 4: Generate intelligent response
            response_context = self.context_manager.get_context_for_response(session_id)
            
            # Handle different conversation stages
            if intent_result.intent in ['rent_prediction', 'tenant_screening', 'maintenance_prediction']:
                response = self._handle_task_intent(intent_result.intent, response_context, session_id)
            elif intent_result.intent == 'help':
                response = self._handle_help_intent(response_context)
            elif intent_result.intent == 'greeting':
                response = self._handle_greeting_intent(response_context)
            else:
                response = self._handle_general_intent(intent_result.intent, response_context)
            
            # Step 5: Enhance response
            enhanced_response = self.response_enhancer.enhance_response(
                response, intent_result.intent, response_context
            )
            
            # Step 6: Add session ID for continuity
            final_response = f"SESSION_ID:{session_id}\n\n{enhanced_response}"
            
            # Update metrics
            total_time = time.time() - start_time
            self._update_performance_metrics(total_time)
            
            if self.debug_mode:
                print(f"⚡ Response generated in {total_time:.2f}s")
            
            return final_response
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error in advanced processing: {e}")
            return None
    
    def _enhanced_data_extraction(self, message: str, intent: str, session: SessionContext) -> Dict[str, Any]:
        """Enhanced data extraction using multiple methods"""
        
        extracted_data = {}
        
        # Use legacy extraction as baseline
        legacy_data = self.legacy_extract_information(message, intent)
        if legacy_data:
            extracted_data.update(legacy_data)
        
        # Add intelligent defaults using context
        if intent == 'rent_prediction':
            if 'address' not in extracted_data and 'address' not in session.active_data:
                # Look for location indicators in message
                location_indicators = ['in', 'at', 'near', 'around']
                words = message.lower().split()
                for i, word in enumerate(words):
                    if word in location_indicators and i + 1 < len(words):
                        potential_location = ' '.join(words[i+1:i+3])
                        if len(potential_location) > 2:
                            extracted_data['address'] = potential_location.title()
                            break
        
        elif intent == 'maintenance_prediction':
            # Smart defaults for maintenance prediction
            if 'age_years' not in extracted_data and 'age_years' not in session.active_data:
                # Look for age indicators
                import re
                age_patterns = [
                    r'(\d+)\s*year[s]?\s*old',
                    r'built\s*(\d+)\s*year[s]?\s*ago',
                    r'(\d+)\s*year[s]?\s*since'
                ]
                for pattern in age_patterns:
                    match = re.search(pattern, message.lower())
                    if match:
                        extracted_data['age_years'] = int(match.group(1))
                        break
            
            if 'last_service_years_ago' not in extracted_data and 'last_service_years_ago' not in session.active_data:
                # Look for maintenance history indicators
                maintenance_patterns = [
                    r'(\d+)\s*year[s]?\s*ago',
                    r'last\s*maintenance\s*.*?(\d+)\s*year[s]?',
                    r'(\d+)\s*year[s]?\s*since\s*maintenance'
                ]
                for pattern in maintenance_patterns:
                    match = re.search(pattern, message.lower())
                    if match:
                        extracted_data['last_service_years_ago'] = int(match.group(1))
                        break
        
        # Add location context if available
        if session.preferences.preferred_defaults:
            for key, value in session.preferences.preferred_defaults.items():
                if key not in extracted_data and key not in session.active_data:
                    extracted_data[key] = value
        
        return extracted_data
    
    def _handle_task_intent(self, intent: str, context: Dict[str, Any], session_id: str) -> str:
        """Handle task-based intents with intelligent flow"""
        
        session = self.context_manager.get_session(session_id)
        missing_fields = context.get('missing_fields', [])
        gathered_fields = context.get('gathered_fields', [])
        
        # Check if we can provide immediate results
        can_proceed = self._can_proceed_with_task(intent, context)
        
        if can_proceed:
            # Generate result using legacy system with smart defaults
            try:
                # Prepare complete data for legacy system
                complete_data = session.active_data.copy()
                
                # Add intelligent defaults for missing required fields
                defaults = self._get_intelligent_defaults(intent, complete_data, context)
                complete_data.update(defaults)
                
                # Call legacy system for actual computation
                result = self._call_legacy_handler(intent, complete_data)
                
                if result:
                    # Generate intelligent response
                    response = self.response_generator.generate_response(
                        'results_presentation', 
                        {
                            **context,
                            'main_result': result,
                            'additional_insights': self._generate_insights(intent, complete_data, result)
                        },
                        'confident'
                    )
                    return response
            
            except Exception as e:
                if self.debug_mode:
                    print(f"❌ Error in task processing: {e}")
        
        # Need more information - generate helpful gathering response
        if missing_fields:
            response = self.response_generator.generate_response(
                'information_gathering',
                {
                    **context,
                    'gathered_info': ', '.join(gathered_fields) if gathered_fields else 'some initial details',
                    'missing_info_request': self._format_missing_info_friendly(missing_fields)
                },
                'encouraging'
            )
        else:
            # Use smart defaults approach
            response = self.response_generator.generate_response(
                'information_gathering',
                context,
                'smart_defaults'
            )
        
        return response
    
    def _can_proceed_with_task(self, intent: str, context: Dict[str, Any]) -> bool:
        """Determine if we can proceed with task using smart defaults"""
        
        gathered_fields = context.get('gathered_fields', [])
        
        # Minimum requirements for each task
        min_requirements = {
            'rent_prediction': 1,  # Just need some property info
            'tenant_screening': 1,  # Just need some applicant info
            'maintenance_prediction': 1  # Just need some property/maintenance info
        }
        
        return len(gathered_fields) >= min_requirements.get(intent, 2)
    
    def _get_intelligent_defaults(self, intent: str, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Get intelligent defaults for missing fields"""
        
        defaults = {}
        
        if intent == 'rent_prediction':
            if 'bedrooms' not in data:
                defaults['bedrooms'] = 2  # UK average
            if 'bathrooms' not in data:
                defaults['bathrooms'] = 1
            if 'property_type' not in data:
                defaults['property_type'] = 'Flat'
            if 'size' not in data:
                defaults['size'] = 70  # Average UK flat size
        
        elif intent == 'tenant_screening':
            if 'employment_status' not in data:
                defaults['employment_status'] = 'Employed'
            if 'rental_history' not in data:
                defaults['rental_history'] = 'Good'
        
        elif intent == 'maintenance_prediction':
            if 'age_years' not in data:
                defaults['age_years'] = 25  # UK average property age
            if 'last_service_years_ago' not in data:
                defaults['last_service_years_ago'] = 1
            if 'location' not in data and 'address' in data:
                defaults['location'] = data['address']
        
        return defaults
    
    def _call_legacy_handler(self, intent: str, data: Dict[str, Any]) -> Optional[str]:
        """Call appropriate legacy handler with complete data"""
        
        try:
            if intent == 'rent_prediction':
                result = self.legacy_handle_conversation([], f"rent prediction {json.dumps(data)}")
                if isinstance(result, dict) and 'response' in result:
                    return result['response']
                return str(result)
            
            elif intent == 'tenant_screening':
                result = self.legacy_handle_conversation([], f"tenant screening {json.dumps(data)}")
                if isinstance(result, dict) and 'response' in result:
                    return result['response']
                return str(result)
            
            elif intent == 'maintenance_prediction':
                result = self.legacy_handle_conversation([], f"maintenance prediction {json.dumps(data)}")
                if isinstance(result, dict) and 'response' in result:
                    return result['response']
                return str(result)
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Legacy handler error: {e}")
            return None
    
    def _generate_insights(self, intent: str, data: Dict[str, Any], result: str) -> str:
        """Generate additional insights based on result"""
        
        insights = []
        
        if intent == 'rent_prediction':
            insights.append("This estimate is based on current market data and property characteristics.")
            if 'address' in data:
                insights.append(f"Location in {data['address']} is a key factor in this pricing.")
        
        elif intent == 'tenant_screening':
            insights.append("This assessment considers multiple risk factors for rental decisions.")
            if 'credit_score' in data:
                score = data['credit_score']
                if isinstance(score, (int, float)) and score > 700:
                    insights.append("Strong credit score indicates good financial reliability.")
        
        elif intent == 'maintenance_prediction':
            insights.append("This prediction is based on property age, location, and maintenance history.")
            if 'age_years' in data:
                age = data['age_years']
                if isinstance(age, (int, float)) and age > 20:
                    insights.append("Older properties typically require more frequent maintenance attention.")
        
        return " ".join(insights)
    
    def _format_missing_info_friendly(self, missing_fields: List[str]) -> str:
        """Format missing information request in friendly way"""
        
        field_names = {
            'address': 'property location',
            'bedrooms': 'number of bedrooms',
            'bathrooms': 'number of bathrooms',
            'size': 'property size',
            'property_type': 'property type (flat, house, etc.)',
            'credit_score': 'credit score',
            'income': 'monthly income',
            'employment_status': 'employment situation',
            'age_years': 'property age',
            'last_service_years_ago': 'when maintenance was last done'
        }
        
        formatted = [field_names.get(field, field.replace('_', ' ')) for field in missing_fields]
        
        if len(formatted) == 1:
            return f"the {formatted[0]}"
        elif len(formatted) == 2:
            return f"the {formatted[0]} and {formatted[1]}"
        else:
            return f"details like {', '.join(formatted[:2])}, or other property information"
    
    def _handle_help_intent(self, context: Dict[str, Any]) -> str:
        """Handle help requests"""
        
        available_options = [
            "🏠 **Rent Pricing** - Get market-based rent estimates for your property",
            "👥 **Tenant Screening** - Evaluate tenant applications and risk assessment", 
            "🔧 **Maintenance Prediction** - Predict when your property might need maintenance"
        ]
        
        return self.response_generator.generate_response(
            'clarification',
            {
                **context,
                'option_list': '\n'.join(available_options)
            },
            'options'
        )
    
    def _handle_greeting_intent(self, context: Dict[str, Any]) -> str:
        """Handle greeting with time awareness"""
        
        situation = 'time_aware' if context.get('current_time') else 'first_time'
        return self.response_generator.generate_response('greeting', context, situation)
    
    def _handle_general_intent(self, intent: str, context: Dict[str, Any]) -> str:
        """Handle general intents"""
        
        if intent == 'clarification':
            return self.response_generator.generate_response(
                'clarification',
                {
                    **context,
                    'assumed_intent': 'help with property management tasks'
                },
                'gentle'
            )
        
        return self.response_generator.generate_response('help', context)
    
    def _update_performance_metrics(self, response_time: float):
        """Update performance tracking"""
        
        # Update average response time
        total_responses = self.performance_metrics['advanced_success'] + self.performance_metrics['legacy_fallback']
        if total_responses > 0:
            current_avg = self.performance_metrics['avg_response_time']
            self.performance_metrics['avg_response_time'] = (
                (current_avg * (total_responses - 1) + response_time) / total_responses
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        
        total_requests = self.performance_metrics['total_requests']
        if total_requests == 0:
            return {'status': 'No requests processed yet'}
        
        advanced_success_rate = (self.performance_metrics['advanced_success'] / total_requests) * 100
        
        return {
            'total_requests': total_requests,
            'advanced_features_success_rate': f"{advanced_success_rate:.1f}%",
            'legacy_fallback_rate': f"{(self.performance_metrics['legacy_fallback'] / total_requests) * 100:.1f}%",
            'avg_response_time': f"{self.performance_metrics['avg_response_time']:.2f}s",
            'context_manager_analytics': self.context_manager.get_analytics(),
            'intent_detector_stats': self.intent_detector.get_statistics()
        }
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """Close conversation session"""
        
        return self.context_manager.close_session(session_id)

# Example usage and testing functions
def test_enhanced_chatbot():
    """Test the enhanced chatbot system"""
    
    print("\n" + "="*60)
    print("🧪 TESTING ENHANCED CHATBOT SYSTEM")
    print("="*60)
    
    # Initialize enhanced chatbot
    chatbot = EnhancedChatbotConversation(debug_mode=True)
    
    # Test conversation flow
    print("\n📝 Test 1: Starting conversation")
    response1 = chatbot.start_conversation("test_user_123")
    print(f"🤖 Response: {response1}")
    
    # Extract session ID
    session_id = response1.split('\n')[0].replace("SESSION_ID:", "")
    
    print(f"\n📝 Test 2: Maintenance prediction request")
    test_message = "see if we need to check this flat for maintenance in Battersea and last maintenance was done 2 years ago"
    response2 = chatbot.process_message(f"SESSION_ID:{session_id}\n\n{test_message}")
    print(f"🤖 Response: {response2}")
    
    print(f"\n📝 Test 3: Additional information")
    response3 = chatbot.process_message(f"SESSION_ID:{session_id}\n\nthe property is 15 years old")
    print(f"🤖 Response: {response3}")
    
    print(f"\n📝 Test 4: Rent prediction request")
    response4 = chatbot.process_message(f"SESSION_ID:{session_id}\n\nWhat's the rent for a 2-bedroom flat in Central London?")
    print(f"🤖 Response: {response4}")
    
    # Performance report
    print(f"\n📊 Performance Report:")
    report = chatbot.get_performance_report()
    for key, value in report.items():
        print(f"   {key}: {value}")
    
    # Close session
    summary = chatbot.close_session(session_id)
    print(f"\n📋 Session Summary: {summary}")
    
    print("\n✅ Enhanced chatbot testing completed!")

if __name__ == "__main__":
    test_enhanced_chatbot()
