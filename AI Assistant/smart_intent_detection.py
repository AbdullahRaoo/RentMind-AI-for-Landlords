"""
Smart Intent Detection System - Resource Efficient & Accurate
============================================================

Features:
- Multi-layer intent detection (fast → accurate)
- Context-aware pattern matching
- Learning from user patterns
- Resource optimization
"""

import re
import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import string

@dataclass
class IntentSignal:
    """Represents an intent detection signal"""
    intent: str
    confidence: float
    source: str  # 'pattern', 'keyword', 'context', 'ml'
    evidence: List[str]

class SmartIntentDetector:
    """
    Highly efficient, multi-layer intent detection system
    """
    
    def __init__(self):
        # Pattern-based detection (fastest)
        self.patterns = {
            'rent_prediction': [
                r'\b(rent|rental|price|pricing|estimate|how much|monthly cost)\b.*\b(property|flat|house|apartment)\b',
                r'\b(predict|estimate|calculate).*\brent\b',
                r'\bhow much.*\b(rent|charge|cost)\b',
                r'\b(property|flat|house).*\b(rent|price|cost)\b',
                r'\brent.*\b(prediction|estimate|calculator)\b'
            ],
            'tenant_screening': [
                r'\b(screen|check|evaluate|assess).*\b(tenant|applicant|renter)\b',
                r'\b(tenant|applicant).*\b(screening|background|check|credit)\b',
                r'\b(approve|accept|reject).*\b(tenant|applicant)\b',
                r'\b(credit score|income|employment).*\b(tenant|applicant)\b',
                r'\b(tenant|applicant).*\b(application|approval)\b'
            ],
            'maintenance_prediction': [
                # Enhanced maintenance patterns with misspellings
                r'\b(maintenance|maintice|maintnance|maintnence|maintanence|maintiance|maintince)\b',
                r'\b(repair|fix|service|upkeep).*\b(predict|check|need|required)\b',
                r'\b(check|see if|predict).*\b(maintenance|repair|service)\b',
                r'\b(maintenance|repair).*\b(needed|required|schedule)\b',
                r'\b(property|flat|house).*\b(maintenance|repair|service)\b',
                r'\b(last|recent).*\b(maintenance|service|repair)\b'
            ],
            'greeting': [
                r'^\s*(hi|hello|hey|good morning|good afternoon|good evening|greetings)\s*[!.]*\s*$',
                r'^\s*(thanks|thank you|appreciate)\b',
                r'^\s*(how are you|how\'s it going)\b'
            ],
            'help': [
                r'\b(help|assist|guide|how to|what can|capabilities)\b',
                r'\b(confused|lost|stuck|don\'t understand)\b',
                r'\b(show me|tell me).*\b(how|what)\b',
                r'^\s*(what|how)\b.*\?'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for intent, patterns in self.patterns.items():
            self.compiled_patterns[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]
        
        # Context keywords (lighter weight detection)
        self.context_keywords = {
            'rent_prediction': {
                'primary': ['rent', 'rental', 'price', 'pricing', 'estimate', 'cost'],
                'secondary': ['property', 'flat', 'house', 'apartment', 'bedroom', 'bathroom', 'size'],
                'negative': ['tenant', 'applicant', 'maintenance', 'repair']
            },
            'tenant_screening': {
                'primary': ['tenant', 'applicant', 'screen', 'screening', 'background'],
                'secondary': ['credit', 'income', 'employment', 'eviction', 'approve', 'reject'],
                'negative': ['rent', 'price', 'maintenance', 'repair']
            },
            'maintenance_prediction': {
                'primary': ['maintenance', 'maintice', 'maintnance', 'repair', 'service', 'fix'],
                'secondary': ['property', 'flat', 'house', 'building', 'check', 'predict'],
                'negative': ['tenant', 'rent', 'price']
            }
        }
        
        # Intent cache for repeated queries
        self.intent_cache = {}
        self.cache_ttl = 1800  # 30 minutes
        
        # User pattern learning
        self.user_patterns = defaultdict(Counter)
        
    def detect_intent(self, message: str, context: Dict = None, user_id: str = None) -> List[IntentSignal]:
        """
        Multi-layer intent detection with confidence scoring
        """
        # Normalize message
        normalized_msg = self._normalize_message(message)
        
        # Check cache first
        cache_key = hashlib.md5(normalized_msg.encode()).hexdigest()
        if cache_key in self.intent_cache:
            cached_result = self.intent_cache[cache_key]
            if time.time() - cached_result['timestamp'] < self.cache_ttl:
                return cached_result['signals']
        
        signals = []
        
        # Layer 1: Fast pattern matching (highest confidence)
        pattern_signals = self._pattern_based_detection(normalized_msg)
        signals.extend(pattern_signals)
        
        # Layer 2: Keyword analysis (medium confidence)
        if not pattern_signals or max(s.confidence for s in pattern_signals) < 0.8:
            keyword_signals = self._keyword_based_detection(normalized_msg)
            signals.extend(keyword_signals)
        
        # Layer 3: Context analysis (when available)
        if context:
            context_signals = self._context_based_detection(normalized_msg, context)
            signals.extend(context_signals)
        
        # Layer 4: User pattern analysis (personalization)
        if user_id and user_id in self.user_patterns:
            pattern_signals = self._user_pattern_detection(normalized_msg, user_id)
            signals.extend(pattern_signals)
        
        # Combine and rank signals
        final_signals = self._combine_signals(signals)
        
        # Cache result
        self.intent_cache[cache_key] = {
            'signals': final_signals,
            'timestamp': time.time()
        }
        
        # Learn from user patterns
        if user_id and final_signals:
            self._update_user_patterns(user_id, normalized_msg, final_signals[0].intent)
        
        return final_signals
    
    def _normalize_message(self, message: str) -> str:
        """Normalize message for consistent processing"""
        # Convert to lowercase
        msg = message.lower().strip()
        
        # Remove extra whitespace
        msg = re.sub(r'\s+', ' ', msg)
        
        # Handle contractions
        contractions = {
            "don't": "do not",
            "won't": "will not", 
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "i'm": "i am",
            "you're": "you are",
            "it's": "it is",
            "that's": "that is"
        }
        
        for contraction, expansion in contractions.items():
            msg = msg.replace(contraction, expansion)
        
        return msg
    
    def _pattern_based_detection(self, message: str) -> List[IntentSignal]:
        """Fast regex-based pattern detection"""
        signals = []
        
        for intent, patterns in self.compiled_patterns.items():
            for i, pattern in enumerate(patterns):
                match = pattern.search(message)
                if match:
                    # Higher confidence for more specific patterns (later in list)
                    confidence = 0.7 + (i * 0.05)  # 0.7 to 0.95
                    
                    # Boost confidence for exact matches
                    if match.group().strip() == message.strip():
                        confidence = min(0.98, confidence + 0.2)
                    
                    signals.append(IntentSignal(
                        intent=intent,
                        confidence=confidence,
                        source='pattern',
                        evidence=[match.group()]
                    ))
                    break  # Use first matching pattern only
        
        return signals
    
    def _keyword_based_detection(self, message: str) -> List[IntentSignal]:
        """Keyword-based detection with scoring"""
        signals = []
        words = set(message.split())
        
        for intent, keywords in self.context_keywords.items():
            score = 0
            evidence = []
            
            # Primary keywords (high weight)
            primary_matches = words.intersection(keywords['primary'])
            score += len(primary_matches) * 3
            evidence.extend(primary_matches)
            
            # Secondary keywords (medium weight)
            secondary_matches = words.intersection(keywords['secondary'])
            score += len(secondary_matches) * 1
            evidence.extend(secondary_matches)
            
            # Negative keywords (penalty)
            negative_matches = words.intersection(keywords.get('negative', []))
            score -= len(negative_matches) * 2
            
            # Convert score to confidence
            if score > 0:
                # Normalize confidence based on message length and keyword density
                max_possible_score = len(keywords['primary']) * 3 + len(keywords['secondary'])
                confidence = min(0.85, score / max_possible_score * 0.85)
                
                # Minimum threshold
                if confidence > 0.3:
                    signals.append(IntentSignal(
                        intent=intent,
                        confidence=confidence,
                        source='keyword',
                        evidence=evidence
                    ))
        
        return signals
    
    def _context_based_detection(self, message: str, context: Dict) -> List[IntentSignal]:
        """Context-aware intent detection"""
        signals = []
        
        # Current conversation intent
        current_intent = context.get('current_intent')
        if current_intent:
            # Check if user is continuing the same intent
            continuation_indicators = [
                'yes', 'correct', 'right', 'exactly', 'that\'s it',
                'no', 'wrong', 'not right', 'incorrect',
                'also', 'and', 'additionally', 'furthermore'
            ]
            
            if any(indicator in message for indicator in continuation_indicators):
                signals.append(IntentSignal(
                    intent=current_intent,
                    confidence=0.6,
                    source='context',
                    evidence=['conversation_continuation']
                ))
        
        # Recent entities suggest intent
        recent_entities = context.get('recent_entities', {})
        if recent_entities:
            if 'property_details' in recent_entities:
                signals.append(IntentSignal(
                    intent='rent_prediction',
                    confidence=0.5,
                    source='context',
                    evidence=['property_entities_present']
                ))
            
            if 'tenant_details' in recent_entities:
                signals.append(IntentSignal(
                    intent='tenant_screening',
                    confidence=0.5,
                    source='context',
                    evidence=['tenant_entities_present']
                ))
        
        return signals
    
    def _user_pattern_detection(self, message: str, user_id: str) -> List[IntentSignal]:
        """Personalized intent detection based on user patterns"""
        signals = []
        user_pattern = self.user_patterns[user_id]
        
        if not user_pattern:
            return signals
        
        # Find most common intent for this user
        most_common_intent = user_pattern.most_common(1)[0][0]
        total_interactions = sum(user_pattern.values())
        
        # If user heavily uses one intent, boost its confidence
        intent_frequency = user_pattern[most_common_intent] / total_interactions
        if intent_frequency > 0.6:  # User prefers this intent
            signals.append(IntentSignal(
                intent=most_common_intent,
                confidence=0.3 * intent_frequency,
                source='user_pattern',
                evidence=[f'user_preference_{intent_frequency:.2f}']
            ))
        
        return signals
    
    def _combine_signals(self, signals: List[IntentSignal]) -> List[IntentSignal]:
        """Combine and rank intent signals"""
        if not signals:
            return []
        
        # Group by intent
        intent_groups = defaultdict(list)
        for signal in signals:
            intent_groups[signal.intent].append(signal)
        
        # Combine signals for each intent
        combined_signals = []
        for intent, group_signals in intent_groups.items():
            # Take highest confidence from each source type
            source_confidences = {}
            all_evidence = []
            
            for signal in group_signals:
                if signal.source not in source_confidences or signal.confidence > source_confidences[signal.source]:
                    source_confidences[signal.source] = signal.confidence
                all_evidence.extend(signal.evidence)
            
            # Combine confidences (weighted average with boost for multiple sources)
            if len(source_confidences) == 1:
                final_confidence = list(source_confidences.values())[0]
            else:
                # Multiple sources boost confidence
                final_confidence = sum(source_confidences.values()) / len(source_confidences)
                final_confidence = min(0.95, final_confidence * 1.2)  # Boost but cap at 95%
            
            combined_signals.append(IntentSignal(
                intent=intent,
                confidence=final_confidence,
                source='combined',
                evidence=list(set(all_evidence))  # Remove duplicates
            ))
        
        # Sort by confidence
        combined_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return combined_signals
    
    def _update_user_patterns(self, user_id: str, message: str, detected_intent: str):
        """Learn from user patterns for personalization"""
        self.user_patterns[user_id][detected_intent] += 1
        
        # Limit memory usage - keep only last 100 interactions per user
        if sum(self.user_patterns[user_id].values()) > 100:
            # Remove least common intent
            least_common = self.user_patterns[user_id].most_common()[-1][0]
            del self.user_patterns[user_id][least_common]
    
    def get_intent_suggestions(self, partial_message: str, context: Dict = None) -> List[str]:
        """Get intent suggestions for autocomplete/suggestions"""
        suggestions = []
        
        # Analyze partial message
        signals = self.detect_intent(partial_message, context)
        
        if signals:
            top_intent = signals[0].intent
            
            # Provide completion suggestions based on top intent
            intent_completions = {
                'rent_prediction': [
                    "What rent can I charge for my property?",
                    "Estimate rent for a 2-bedroom flat in London",
                    "How much should I charge for rent?"
                ],
                'tenant_screening': [
                    "Screen this tenant application",
                    "Evaluate tenant with 700 credit score",
                    "Check if this applicant is suitable"
                ],
                'maintenance_prediction': [
                    "Check if my property needs maintenance",
                    "Predict maintenance for my flat",
                    "When should I schedule maintenance?"
                ]
            }
            
            suggestions = intent_completions.get(top_intent, [])
        
        return suggestions[:3]  # Return top 3 suggestions

# Lightweight entity extraction
class QuickEntityExtractor:
    """Fast, pattern-based entity extraction"""
    
    def __init__(self):
        self.patterns = {
            'numbers': r'\b\d+(?:\.\d+)?\b',
            'money': r'[£$]\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:pounds?|dollars?|£|$)\b',
            'bedrooms': r'\b(\d+)\s*(?:bed|bedroom|br)\b',
            'bathrooms': r'\b(\d+)\s*(?:bath|bathroom|ba)\b',
            'property_types': r'\b(flat|apartment|house|studio|property|unit|building|place)\b',
            'locations': r'\b(?:in|at|located|near)\s+([A-Z][a-zA-Z\s]{2,30})\b',
            'time_periods': r'\b(\d+)\s*(year|month|week|day)s?\s*ago\b',
            'credit_scores': r'\b([3-8]\d{2})\s*(?:credit\s*score|score)?\b'
        }
        
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE) 
            for name, pattern in self.patterns.items()
        }
    
    def extract(self, message: str) -> Dict[str, List[str]]:
        """Extract entities from message"""
        entities = {}
        
        for entity_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(message)
            if matches:
                entities[entity_type] = matches
        
        return entities

# Export components
__all__ = [
    'SmartIntentDetector',
    'QuickEntityExtractor', 
    'IntentSignal'
]
