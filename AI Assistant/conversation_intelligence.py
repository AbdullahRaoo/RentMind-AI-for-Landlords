"""
Advanced NER, multi-intent detection, and entity linking for conversational AI.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import spacy
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Supported intent types."""
    RENT_PREDICTION = "rent_prediction"
    TENANT_SCREENING = "tenant_screening"
    MAINTENANCE_PREDICTION = "maintenance_prediction"
    GREETING = "greeting"
    SMALL_TALK = "small_talk"
    CLARIFICATION = "clarification"
    UNKNOWN = "unknown"

@dataclass
class Entity:
    """Extracted entity with metadata."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized_value: Any = None
    context: str = ""

@dataclass
class Intent:
    """Detected intent with confidence."""
    type: IntentType
    confidence: float
    entities: List[Entity]
    context: str = ""

@dataclass
class ConversationResult:
    """Complete conversation analysis result."""
    intents: List[Intent]
    entities: List[Entity]
    primary_intent: Intent
    confidence: float
    requires_clarification: bool = False
    clarification_message: str = ""

class AdvancedNERProcessor:
    """
    Advanced Named Entity Recognition using spaCy and custom rules.
    """
    
    def __init__(self):
        """Initialize NER processor with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic regex fallback.")
            self.nlp = None
        
        # Custom entity patterns for real estate domain
        self.custom_patterns = {
            'ADDRESS': [
                r'\b\d+\s+[A-Za-z\s]+(?:Road|Rd|Street|St|Lane|Ln|Avenue|Ave|Drive|Dr|Place|Pl|Court|Ct|Way|Close|Crescent|Cres)\b',
                r'\b[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2,3}\s*\d{3,5}\b'
            ],
            'POSTCODE': [
                r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b',
                r'\bUB\d+\b'
            ],
            'BEDROOMS': [
                r'\b(\d+)\s*(?:bed|bedroom|br)s?\b',
                r'\b(\d+)\s*b\b'
            ],
            'BATHROOMS': [
                r'\b(\d+)\s*(?:bath|bathroom|br)s?\b',
                r'\b(\d+)\s*ba\b'
            ],
            'SIZE': [
                r'\b(\d+(?:\.\d+)?)\s*(?:sq\s*ft|sqft|square\s*feet|ft²)\b',
                r'\b(\d+(?:\.\d+)?)\s*(?:m²|square\s*meters?)\b'
            ],
            'PROPERTY_TYPE': [
                r'\b(apartment|flat|house|villa|condo|studio|maisonette|penthouse|bungalow)\b'
            ],
            'CREDIT_SCORE': [
                r'\bcredit\s*score\s*(?:is\s*|of\s*)?(\d{2,4})\b',
                r'\bscore\s*(?:is\s*|of\s*)?(\d{2,4})\b'
            ],
            'INCOME': [
                r'\b£?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:per\s*month|monthly|/month)\b',
                r'\bincome\s*(?:is\s*|of\s*)?£?(\d+(?:,\d{3})*(?:\.\d{2})?)\b'
            ],
            'RENT': [
                r'\brent\s*(?:is\s*|of\s*)?£?(\d+(?:,\d{3})*(?:\.\d{2})?)\b',
                r'\b£(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:per\s*month|monthly|/month|rent)\b'
            ]
        }
        
        # Field synonyms for entity linking
        self.field_synonyms = {
            'address': ['address', 'location', 'property address', 'where', 'place'],
            'postcode': ['postcode', 'postal code', 'zip code', 'code', 'area code'],
            'bedrooms': ['bedrooms', 'beds', 'bed', 'bedroom', 'br'],
            'bathrooms': ['bathrooms', 'baths', 'bath', 'bathroom', 'washroom'],
            'size': ['size', 'area', 'square feet', 'sq ft', 'sqft', 'footage'],
            'property_type': ['type', 'property type', 'kind', 'category'],
            'credit_score': ['credit score', 'score', 'credit rating', 'rating'],
            'income': ['income', 'salary', 'earnings', 'monthly income', 'annual income'],
            'rent': ['rent', 'rental', 'monthly rent', 'asking rent', 'price'],
            'employment_status': ['employment', 'job', 'work', 'occupation', 'employed'],
            'eviction_record': ['eviction', 'evicted', 'prior eviction', 'eviction history']
        }
    
    def extract_entities(self, text: str, context: str = "") -> List[Entity]:
        """
        Extract entities from text using spaCy and custom patterns.
        
        Args:
            text: Input text to analyze
            context: Additional context for entity resolution
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Map spaCy labels to our domain
                mapped_label = self._map_spacy_label(ent.label_)
                if mapped_label:
                    entity = Entity(
                        text=ent.text,
                        label=mapped_label,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8,  # spaCy confidence approximation
                        context=context
                    )
                    entities.append(entity)
        
        # Extract using custom patterns
        custom_entities = self._extract_custom_patterns(text, context)
        entities.extend(custom_entities)
        
        # Deduplicate and resolve conflicts
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _map_spacy_label(self, label: str) -> Optional[str]:
        """Map spaCy entity labels to our domain-specific labels."""
        mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'GPE': 'ADDRESS',  # Geopolitical entity -> address
            'LOC': 'ADDRESS',  # Location -> address
            'MONEY': 'MONEY',
            'CARDINAL': 'NUMBER',
            'ORDINAL': 'NUMBER'
        }
        return mapping.get(label)
    
    def _extract_custom_patterns(self, text: str, context: str) -> List[Entity]:
        """Extract entities using custom regex patterns."""
        entities = []
        text_lower = text.lower()
        
        for label, patterns in self.custom_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract the captured group if present, otherwise full match
                    if match.groups():
                        entity_text = match.group(1)
                        # Adjust start/end positions for captured group
                        full_match = match.group(0)
                        start_offset = full_match.find(entity_text)
                        start = match.start() + start_offset
                        end = start + len(entity_text)
                    else:
                        entity_text = match.group(0)
                        start = match.start()
                        end = match.end()
                    
                    # Normalize value based on label
                    normalized_value = self._normalize_entity_value(entity_text, label)
                    
                    entity = Entity(
                        text=entity_text,
                        label=label,
                        start=start,
                        end=end,
                        confidence=0.9,  # High confidence for pattern matches
                        normalized_value=normalized_value,
                        context=context
                    )
                    entities.append(entity)
        
        return entities
    
    def _normalize_entity_value(self, text: str, label: str) -> Any:
        """Normalize entity values to standard formats."""
        text_clean = text.strip()
        
        if label in ['BEDROOMS', 'BATHROOMS', 'CREDIT_SCORE']:
            # Extract numeric value
            num_match = re.search(r'(\d+)', text_clean)
            if num_match:
                return int(num_match.group(1))
        
        elif label in ['SIZE', 'INCOME', 'RENT']:
            # Extract numeric value with decimals
            num_match = re.search(r'(\d+(?:\.\d+)?)', text_clean.replace(',', ''))
            if num_match:
                return float(num_match.group(1))
        
        elif label == 'POSTCODE':
            # Standardize postcode format
            return text_clean.upper().replace(' ', '')
        
        elif label == 'PROPERTY_TYPE':
            # Standardize property type
            return text_clean.lower()
        
        elif label == 'ADDRESS':
            # Clean up address
            return text_clean.title()
        
        return text_clean
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities and resolve conflicts."""
        # Sort by start position
        entities.sort(key=lambda e: e.start)
        
        deduplicated = []
        for entity in entities:
            # Check for overlaps with existing entities
            overlap = False
            for existing in deduplicated:
                if (entity.start < existing.end and entity.end > existing.start):
                    # Overlapping entities - keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlap = True
                    break
            
            if not overlap:
                deduplicated.append(entity)
        
        return deduplicated

class MultiIntentDetector:
    """
    Detects multiple intents in a single user message.
    """
    
    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI client for LLM-based intent detection."""
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        # Keyword patterns for quick intent detection
        self.intent_keywords = {
            IntentType.RENT_PREDICTION: [
                'rent', 'price', 'estimate', 'cost', 'how much', 'pricing', 'value'
            ],
            IntentType.TENANT_SCREENING: [
                'screen', 'tenant', 'applicant', 'background', 'check', 'approve', 'credit'
            ],
            IntentType.MAINTENANCE_PREDICTION: [
                'maintenance', 'repair', 'fix', 'upkeep', 'service', 'broken', 'issue'
            ],
            IntentType.GREETING: [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'
            ],
            IntentType.SMALL_TALK: [
                'how are you', 'thank you', 'thanks', 'goodbye', 'bye', 'help'
            ]
        }
    
    def detect_intents(self, 
                      text: str, 
                      conversation_history: List[Dict] = None,
                      entities: List[Entity] = None) -> List[Intent]:
        """
        Detect multiple intents in the given text.
        
        Args:
            text: User input text
            conversation_history: Previous conversation context
            entities: Extracted entities
            
        Returns:
            List of detected intents
        """
        intents = []
        
        # Quick keyword-based detection
        keyword_intents = self._detect_keyword_intents(text)
        intents.extend(keyword_intents)
        
        # LLM-based detection for complex cases
        if not intents or len(text.split()) > 15:  # Use LLM for longer/complex texts
            llm_intents = self._detect_llm_intents(text, conversation_history, entities)
            
            # Merge with keyword intents, avoiding duplicates
            for llm_intent in llm_intents:
                if not any(intent.type == llm_intent.type for intent in intents):
                    intents.append(llm_intent)
        
        # Sort by confidence
        intents.sort(key=lambda i: i.confidence, reverse=True)
        
        return intents
    
    def _detect_keyword_intents(self, text: str) -> List[Intent]:
        """Detect intents using keyword matching."""
        intents = []
        text_lower = text.lower()
        
        for intent_type, keywords in self.intent_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                confidence = min(0.9, matches * 0.3)  # Scale confidence based on keyword matches
                intent = Intent(
                    type=intent_type,
                    confidence=confidence,
                    entities=[],
                    context=f"Keyword matches: {matches}"
                )
                intents.append(intent)
        
        return intents
    
    def _detect_llm_intents(self, 
                           text: str,
                           conversation_history: List[Dict] = None,
                           entities: List[Entity] = None) -> List[Intent]:
        """Use LLM for sophisticated intent detection."""
        try:
            # Prepare context
            context = ""
            if conversation_history:
                recent_context = conversation_history[-3:]  # Last 3 messages
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_context])
            
            entity_context = ""
            if entities:
                entity_context = "Entities found: " + ", ".join([f"{e.label}={e.text}" for e in entities])
            
            # LLM prompt for intent detection
            prompt = f"""
You are an expert intent classifier for a landlord AI assistant. Analyze the user message and detect ALL possible intents.

Available intents:
- rent_prediction: User wants to estimate property rent
- tenant_screening: User wants to screen a tenant applicant  
- maintenance_prediction: User wants to predict maintenance needs
- greeting: User is greeting or starting conversation
- small_talk: User is making small talk, thanking, or ending conversation
- clarification: User is asking for clarification or help
- unknown: Intent is unclear or not supported

Context from conversation:
{context}

{entity_context}

User message: "{text}"

Return your analysis as JSON with this format:
{{
    "intents": [
        {{
            "type": "intent_name",
            "confidence": 0.85,
            "reasoning": "why this intent was detected"
        }}
    ]
}}

Detect ALL relevant intents, not just the primary one. Be conservative with confidence scores.
"""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()
            
            # Parse JSON response
            try:
                data = json.loads(result)
                intents = []
                
                for intent_data in data.get("intents", []):
                    intent_type_str = intent_data.get("type", "unknown")
                    # Map string to enum
                    try:
                        intent_type = IntentType(intent_type_str)
                    except ValueError:
                        intent_type = IntentType.UNKNOWN
                    
                    intent = Intent(
                        type=intent_type,
                        confidence=float(intent_data.get("confidence", 0.5)),
                        entities=entities or [],
                        context=intent_data.get("reasoning", "")
                    )
                    intents.append(intent)
                
                return intents
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM intent response as JSON")
                return []
            
        except Exception as e:
            logger.error(f"LLM intent detection failed: {e}")
            return []

class ContextAwareEntityLinker:
    """
    Links entities to conversation context and resolves ambiguities.
    """
    
    def __init__(self):
        """Initialize entity linker."""
        # Field priority for ambiguous entities
        self.field_priority = {
            'address': 1,
            'rent': 2,
            'bedrooms': 3,
            'bathrooms': 4,
            'size': 5,
            'property_type': 6,
            'credit_score': 7,
            'income': 8
        }
    
    def link_entities(self, 
                     entities: List[Entity],
                     intents: List[Intent],
                     conversation_history: List[Dict] = None,
                     current_fields: Dict = None) -> Dict[str, Any]:
        """
        Link entities to specific fields based on context.
        
        Args:
            entities: Extracted entities
            intents: Detected intents
            conversation_history: Previous conversation
            current_fields: Currently collected field values
            
        Returns:
            Dictionary of linked field values
        """
        linked_fields = current_fields.copy() if current_fields else {}
        
        # Determine primary intent
        primary_intent = intents[0] if intents else None
        
        for entity in entities:
            field_name = self._resolve_entity_to_field(entity, primary_intent, conversation_history)
            
            if field_name:
                # Use normalized value if available
                value = entity.normalized_value if entity.normalized_value is not None else entity.text
                
                # Only update if not already set or new value has higher confidence
                if (field_name not in linked_fields or 
                    entity.confidence > linked_fields.get(f"{field_name}_confidence", 0)):
                    
                    linked_fields[field_name] = value
                    linked_fields[f"{field_name}_confidence"] = entity.confidence
        
        # Clean up confidence tracking fields for return
        clean_fields = {k: v for k, v in linked_fields.items() if not k.endswith('_confidence')}
        
        return clean_fields
    
    def _resolve_entity_to_field(self, 
                                entity: Entity,
                                primary_intent: Intent = None,
                                conversation_history: List[Dict] = None) -> Optional[str]:
        """Resolve entity to a specific field name."""
        
        # Direct label mapping
        label_mapping = {
            'ADDRESS': 'address',
            'POSTCODE': 'subdistrict_code',
            'BEDROOMS': 'BEDROOMS',
            'BATHROOMS': 'BATHROOMS',
            'SIZE': 'SIZE',
            'PROPERTY_TYPE': 'PROPERTY TYPE',
            'CREDIT_SCORE': 'credit_score',
            'INCOME': 'income',
            'RENT': 'rent'
        }
        
        direct_field = label_mapping.get(entity.label)
        if direct_field:
            return direct_field
        
        # Context-based resolution for ambiguous entities
        if entity.label in ['NUMBER', 'MONEY']:
            return self._resolve_ambiguous_number(entity, primary_intent, conversation_history)
        
        # Use entity text to infer field
        entity_text_lower = entity.text.lower()
        
        # Check against field synonyms
        for field, synonyms in self.field_synonyms.items():
            if any(synonym in entity_text_lower for synonym in synonyms):
                return self._standardize_field_name(field)
        
        return None
    
    def _resolve_ambiguous_number(self, 
                                 entity: Entity,
                                 primary_intent: Intent = None,
                                 conversation_history: List[Dict] = None) -> Optional[str]:
        """Resolve ambiguous numeric entities based on context."""
        
        # Look for context clues in surrounding text
        context_window = 50  # Characters before and after
        start = max(0, entity.start - context_window)
        end = entity.end + context_window
        context = entity.context[start:end].lower() if entity.context else ""
        
        # Check for field indicators in context
        if any(word in context for word in ['bed', 'bedroom', 'br']):
            return 'BEDROOMS'
        elif any(word in context for word in ['bath', 'bathroom']):
            return 'BATHROOMS'
        elif any(word in context for word in ['sq ft', 'sqft', 'square feet', 'size']):
            return 'SIZE'
        elif any(word in context for word in ['credit', 'score']):
            return 'credit_score'
        elif any(word in context for word in ['income', 'salary', 'earn']):
            return 'income'
        elif any(word in context for word in ['rent', 'monthly', '£']):
            return 'rent'
        
        # Use intent to guess field
        if primary_intent:
            if primary_intent.type == IntentType.RENT_PREDICTION:
                # For rent prediction, numbers likely refer to property features
                value = entity.normalized_value or entity.text
                try:
                    num_value = float(str(value).replace(',', ''))
                    if num_value < 10:
                        return 'BEDROOMS'  # Small numbers likely bedrooms
                    elif num_value < 20:
                        return 'BATHROOMS'  # Medium numbers likely bathrooms
                    elif num_value > 100:
                        return 'SIZE'  # Large numbers likely size
                except (ValueError, TypeError):
                    pass
            
            elif primary_intent.type == IntentType.TENANT_SCREENING:
                # For tenant screening, numbers likely refer to financial info
                value = entity.normalized_value or entity.text
                try:
                    num_value = float(str(value).replace(',', ''))
                    if 300 <= num_value <= 850:
                        return 'credit_score'  # Typical credit score range
                    elif num_value > 1000:
                        return 'income'  # Large numbers likely income
                    else:
                        return 'rent'  # Medium numbers likely rent
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def _standardize_field_name(self, field: str) -> str:
        """Standardize field names to match handler requirements."""
        field_mapping = {
            'postcode': 'subdistrict_code',
            'bedrooms': 'BEDROOMS',
            'bathrooms': 'BATHROOMS',
            'size': 'SIZE',
            'property_type': 'PROPERTY TYPE'
        }
        return field_mapping.get(field, field)

class ConversationalIntelligence:
    """
    Main class that combines NER, intent detection, and entity linking.
    """
    
    def __init__(self, openai_api_key: str):
        """Initialize all components."""
        self.ner_processor = AdvancedNERProcessor()
        self.intent_detector = MultiIntentDetector(openai_api_key)
        self.entity_linker = ContextAwareEntityLinker()
        
        # Greeting responses
        self.greeting_responses = [
            "Hello! I'm LandlordBuddy, your AI assistant for property management. I can help you with rent pricing, tenant screening, and maintenance predictions. What would you like to do today?",
            "Hi there! I'm here to help you with all your landlord needs. Whether you want to estimate rent, screen a tenant, or predict maintenance issues, just let me know!",
            "Good day! Welcome to LandlordBuddy. How can I assist you with your property management today?"
        ]
        
        # Small talk responses
        self.small_talk_responses = {
            'thank': "You're welcome! I'm happy to help with your property management needs.",
            'help': "I can help you with three main tasks: 1) Rent pricing estimates, 2) Tenant screening, and 3) Maintenance predictions. Which would you like to try?",
            'goodbye': "Goodbye! Feel free to come back anytime you need help with your properties.",
            'how_are_you': "I'm doing great and ready to help you with your landlord tasks! What can I assist you with today?"
        }
    
    def analyze_message(self, 
                       user_message: str,
                       conversation_history: List[Dict] = None,
                       current_fields: Dict = None,
                       session_id: str = None) -> ConversationResult:
        """
        Perform complete conversation analysis.
        
        Args:
            user_message: User's input message
            conversation_history: Previous conversation context
            current_fields: Currently collected field values
            session_id: Session identifier for context
            
        Returns:
            Complete analysis result
        """
        # Extract entities
        entities = self.ner_processor.extract_entities(user_message, context=user_message)
        
        # Detect intents
        intents = self.intent_detector.detect_intents(user_message, conversation_history, entities)
        
        # Determine primary intent
        primary_intent = intents[0] if intents else Intent(
            type=IntentType.UNKNOWN,
            confidence=0.0,
            entities=entities
        )
        
        # Link entities to fields
        linked_fields = self.entity_linker.link_entities(
            entities, intents, conversation_history, current_fields
        )
        
        # Check if clarification is needed
        requires_clarification = self._requires_clarification(intents, entities)
        clarification_message = self._generate_clarification(intents, entities) if requires_clarification else ""
        
        return ConversationResult(
            intents=intents,
            entities=entities,
            primary_intent=primary_intent,
            confidence=primary_intent.confidence,
            requires_clarification=requires_clarification,
            clarification_message=clarification_message
        )
    
    def handle_greeting(self) -> str:
        """Handle greeting intent."""
        import random
        return random.choice(self.greeting_responses)
    
    def handle_small_talk(self, user_message: str) -> str:
        """Handle small talk intent."""
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['thank', 'thanks']):
            return self.small_talk_responses['thank']
        elif any(word in message_lower for word in ['help', 'what can you do']):
            return self.small_talk_responses['help']
        elif any(word in message_lower for word in ['bye', 'goodbye', 'see you']):
            return self.small_talk_responses['goodbye']
        elif any(phrase in message_lower for phrase in ['how are you', 'how do you do']):
            return self.small_talk_responses['how_are_you']
        else:
            return "I'm here to help you with property management. What would you like to do today?"
    
    def _requires_clarification(self, intents: List[Intent], entities: List[Entity]) -> bool:
        """Check if the message requires clarification."""
        # No clear intent detected
        if not intents or all(intent.confidence < 0.5 for intent in intents):
            return True
        
        # Multiple high-confidence intents
        high_confidence_intents = [i for i in intents if i.confidence > 0.7]
        if len(high_confidence_intents) > 1:
            return True
        
        # Intent detected but no relevant entities
        primary_intent = intents[0]
        if primary_intent.type in [IntentType.RENT_PREDICTION, IntentType.TENANT_SCREENING, IntentType.MAINTENANCE_PREDICTION]:
            if not entities:
                return True
        
        return False
    
    def _generate_clarification(self, intents: List[Intent], entities: List[Entity]) -> str:
        """Generate clarification message."""
        if not intents or all(intent.confidence < 0.5 for intent in intents):
            return ("I'm not sure what you'd like me to help with. I can assist you with:\n"
                   "- **Rent Pricing**: Estimate the rent for your property\n"
                   "- **Tenant Screening**: Assess a tenant applicant\n"
                   "- **Maintenance Prediction**: Predict maintenance needs\n\n"
                   "Which of these would you like to do?")
        
        high_confidence_intents = [i for i in intents if i.confidence > 0.7]
        if len(high_confidence_intents) > 1:
            intent_names = [intent.type.value.replace('_', ' ').title() for intent in high_confidence_intents]
            return (f"I detected multiple requests: {', '.join(intent_names)}. "
                   f"Which one would you like to start with?")
        
        # Intent detected but missing entities
        primary_intent = intents[0]
        if primary_intent.type == IntentType.RENT_PREDICTION:
            return ("I understand you want a rent estimate. To help you, I'll need some property details. "
                   "Can you tell me about the address, number of bedrooms and bathrooms, size, and property type?")
        elif primary_intent.type == IntentType.TENANT_SCREENING:
            return ("I can help screen a tenant. I'll need the applicant's credit score, income, "
                   "the monthly rent amount, employment status, and whether they have any prior evictions.")
        elif primary_intent.type == IntentType.MAINTENANCE_PREDICTION:
            return ("I can predict maintenance needs. Please provide the property address, "
                   "how old the property is, when it was last serviced, and the current season.")
        
        return "Could you provide more details about what you'd like me to help with?"

# Global instance
_conversation_intelligence = None

def get_conversation_intelligence() -> ConversationalIntelligence:
    """Get or create global conversation intelligence instance."""
    global _conversation_intelligence
    if _conversation_intelligence is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _conversation_intelligence = ConversationalIntelligence(openai_api_key)
    return _conversation_intelligence
