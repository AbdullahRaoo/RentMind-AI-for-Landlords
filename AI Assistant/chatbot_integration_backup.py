import os
import re
import json
import csv
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import time
import random
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# Import our new modules
from milvus_utils import get_milvus_store
from conversation_intelligence import get_conversation_intelligence, IntentType
from cost_tracker import track_langchain_response, print_session_summary, get_session_summary

load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Wrapper function to track API calls
def invoke_with_tracking(chat_instance, messages, context="API Call"):
    """
    Wrapper function to invoke LangChain ChatOpenAI with cost tracking
    """
    start_time = time.time()
    response = chat_instance.invoke(messages)
    duration = time.time() - start_time
    
    # Extract model name from the chat instance
    model_name = getattr(chat_instance, 'model_name', 'gpt-4')
    
    # Track the response
    track_langchain_response(response, model_name, context, duration)
    
    return response

# --- Modular Conversational Engine ---

class BaseModuleHandler:
    """
    Base class for all module handlers (rent, tenant, maintenance).
    """
    def extract_fields(self, user_message, conversation_history, last_candidate_fields=None):
        raise NotImplementedError
    def summarize_fields(self, fields):
        raise NotImplementedError
    def needs_confirmation(self, user_message):
        # Only treat as confirmation if the user is confirming the information, not to trigger the model
        confirmation_phrases = ["yes", "correct", "that's right", "yep", "confirmed", "go ahead", "proceed"]
        return user_message.strip().lower() in confirmation_phrases
    def run_model(self, fields):
        raise NotImplementedError
    def format_result(self, result):
        raise NotImplementedError

class RentPredictionHandler(BaseModuleHandler):
    required_fields = [
        "address", "subdistrict_code", "BEDROOMS", "BATHROOMS", "SIZE", "PROPERTY TYPE"
    ]
    
    FIELD_SYNONYMS = {
        "address": ["address", "location", "property address"],
        "subdistrict_code": ["subdistrict code", "code", "postcode", "postal code", "ub8"],
        "BEDROOMS": ["bedrooms", "number of bedrooms", "bed room", "beds", "bed"],
        "BATHROOMS": ["bathrooms", "number of bathrooms", "bathroom", "washroom", "baths", "bath"],
        "SIZE": ["size", "area", "square feet", "sq ft", "sqft", "foot", "feet"],
        "PROPERTY TYPE": ["property type", "type", "apartment", "house", "flat"]
    }
    _model = None
    _model_path = os.path.join(os.path.dirname(__file__), "../Rent Pricing AI/rent_xgboost_model.json")

    @classmethod
    def get_model(cls):
        if cls._model is None:
            import xgboost as xgb
            cls._model = xgb.Booster()
            cls._model.load_model(cls._model_path)
        return cls._model

    def __init__(self):
        self.system_prompt = (
            "You are LandlordBuddy, an expert and friendly AI assistant for landlords. "
            "You support rent pricing, tenant screening, and maintenance prediction. "
            "If the user asks for a feature you support, guide them to provide all required info. "
            "If info is missing, you can now provide estimates using intelligent defaults when at least 2 key details are provided. "
            "You must be flexible in understanding user input: users may provide information in any format, not just JSON or structured lists. "
            "You should do your best to interpret and extract the required details for rent prediction even if the user uses casual language, synonyms, or different phrasings. "
            "For example, if the user says 'avg distance', 'average station distance', 'station distance', or any similar phrase, you should map it to 'avg_distance_to_nearest_station'. "
            "Similarly, for all required fields, try to match and extract the information even if the user does not use the exact field names. "
            "The required information for rent prediction is: address, subdistrict_code, BEDROOMS, BATHROOMS, SIZE (in sq ft), PROPERTY TYPE. "
            "When you have at least 2 key details, you can provide an estimate using smart defaults for missing information. "
            "When providing estimates with defaults, be conversational and mention what assumptions you're making. "
            "Always offer to provide a more precise estimate if they provide the missing details. "
            "If the request is not supported, politely say so. "
            "Never mention OpenAI or your own limitations. "
            "Always keep the conversation natural and helpful. "
            "Respond in markdown."
        )
        self.chat = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

    def get_smart_defaults(self, provided_fields):
        """Generate contextual defaults based on provided fields"""
        defaults = {}
        
        # Get provided values
        property_type = provided_fields.get('PROPERTY TYPE', '').lower()
        bedrooms = provided_fields.get('BEDROOMS')
        bathrooms = provided_fields.get('BATHROOMS')
        size = provided_fields.get('SIZE')
        
        try:
            bedrooms = int(float(bedrooms)) if bedrooms and bedrooms not in (None, '', 0) else None
        except:
            bedrooms = None
            
        try:
            bathrooms = int(float(bathrooms)) if bathrooms and bathrooms not in (None, '', 0) else None
        except:
            bathrooms = None
            
        try:
            size = float(size) if size and size not in (None, '', 0) else None
        except:
            size = None

        # Default BATHROOMS based on context
        if not bathrooms:
            if bedrooms:
                if bedrooms == 1:
                    defaults['BATHROOMS'] = 1
                elif bedrooms == 2:
                    defaults['BATHROOMS'] = 1 if 'studio' in property_type or 'flat' in property_type else 2
                elif bedrooms >= 3:
                    defaults['BATHROOMS'] = 2
                else:
                    defaults['BATHROOMS'] = 1
            elif 'studio' in property_type:
                defaults['BATHROOMS'] = 1
            elif size:
                # Size-based bathroom estimation
                if size < 400:
                    defaults['BATHROOMS'] = 1
                elif size < 1000:
                    defaults['BATHROOMS'] = 1
                else:
                    defaults['BATHROOMS'] = 2
            else:
                defaults['BATHROOMS'] = 1  # Conservative default

        # Default BEDROOMS based on context
        if not bedrooms:
            if 'studio' in property_type:
                defaults['BEDROOMS'] = 0
            elif bathrooms:
                if bathrooms == 1:
                    defaults['BEDROOMS'] = 1 if size and size < 600 else 2
                elif bathrooms >= 2:
                    defaults['BEDROOMS'] = 2 if size and size < 1200 else 3
                else:
                    defaults['BEDROOMS'] = 1
            elif size:
                # Size-based bedroom estimation
                if size < 400:
                    defaults['BEDROOMS'] = 0 if 'studio' in property_type else 1
                elif size < 700:
                    defaults['BEDROOMS'] = 1
                elif size < 1200:
                    defaults['BEDROOMS'] = 2
                else:
                    defaults['BEDROOMS'] = 3
            else:
                defaults['BEDROOMS'] = 2  # Common default

        # Default SIZE based on context
        if not size:
            if bedrooms == 0 or 'studio' in property_type:
                defaults['SIZE'] = 350
            elif bedrooms == 1:
                defaults['SIZE'] = 550
            elif bedrooms == 2:
                defaults['SIZE'] = 850 if 'apartment' in property_type or 'flat' in property_type else 1100
            elif bedrooms == 3:
                defaults['SIZE'] = 1200
            elif bedrooms >= 4:
                defaults['SIZE'] = 1600
            else:
                # Fallback based on bathrooms
                if bathrooms == 1:
                    defaults['SIZE'] = 600
                elif bathrooms >= 2:
                    defaults['SIZE'] = 1000
                else:
                    defaults['SIZE'] = 750

        # Default PROPERTY TYPE
        if not provided_fields.get('PROPERTY TYPE'):
            defaults['PROPERTY TYPE'] = 'Apartment'

        # Default address/subdistrict_code (use common London area as fallback)
        if not provided_fields.get('address'):
            defaults['address'] = 'London Area'
        if not provided_fields.get('subdistrict_code'):
            defaults['subdistrict_code'] = 'SW1'

        return defaults

    def create_estimate_with_defaults(self, provided_fields):
        """Create an estimate using smart defaults and return a conversational response"""
        defaults = self.get_smart_defaults(provided_fields)
        
        # Merge provided fields with defaults
        complete_fields = {}
        for field in self.required_fields:
            if field in provided_fields and provided_fields[field] not in (None, '', 0, 0.0):
                complete_fields[field] = provided_fields[field]
            else:
                complete_fields[field] = defaults.get(field, '')

        # Generate assumptions text
        assumptions = []
        for field, default_value in defaults.items():
            if field in ['BATHROOMS', 'BEDROOMS', 'SIZE']:
                field_display = field.lower().replace('_', ' ')
                if field == 'BATHROOMS':
                    assumptions.append(f"{default_value} bathroom{'s' if default_value != 1 else ''}")
                elif field == 'BEDROOMS':
                    if default_value == 0:
                        assumptions.append("studio (0 bedrooms)")
                    else:
                        assumptions.append(f"{default_value} bedroom{'s' if default_value != 1 else ''}")
                elif field == 'SIZE':
                    assumptions.append(f"approximately {int(default_value)} sq ft")

        # Get the rent prediction
        try:
            result = self.run_model(complete_fields)
            
            # Extract prediction values for conversational response
            import re
            rent_match = re.search(r'£(\d+)', result)
            range_match = re.search(r'£(\d+)–£(\d+)', result)
            confidence_match = re.search(r'(\d+\.?\d*)%', result)
            
            predicted_rent = rent_match.group(1) if rent_match else "N/A"
            rent_range = f"£{range_match.group(1)}-£{range_match.group(2)}" if range_match else "N/A"
            confidence = confidence_match.group(1) if confidence_match else "N/A"

            # Create conversational response
            property_desc = self.create_property_description(provided_fields, complete_fields)
            
            response = f"Based on what you've told me, {property_desc}"
            
            if assumptions:
                response += f" As you didn't provide the {self.format_assumptions(assumptions)}, I'm assuming {', '.join(assumptions[:-1])}"
                if len(assumptions) > 1:
                    response += f" and {assumptions[-1]}"
                else:
                    response += assumptions[0] if assumptions else ""
                response += "."
            
            response += f"\n\n**Estimated Monthly Rent: £{predicted_rent}**\n"
            response += f"**Range: {rent_range}**\n"
            response += f"**Confidence: {confidence}%**\n\n"
            
            response += "If you'd like a more precise estimation, just let me know the missing details! "
            
            # Ask for missing information
            missing_original = [f for f in self.required_fields if f not in provided_fields or provided_fields[f] in (None, '', 0, 0.0)]
            if missing_original:
                missing_friendly = []
                for field in missing_original:
                    if field == 'BATHROOMS':
                        missing_friendly.append('number of bathrooms')
                    elif field == 'BEDROOMS':
                        missing_friendly.append('number of bedrooms')
                    elif field == 'SIZE':
                        missing_friendly.append('property size (in sq ft)')
                    elif field == 'PROPERTY TYPE':
                        missing_friendly.append('property type (flat, house, etc.)')
                    elif field == 'address':
                        missing_friendly.append('specific address')
                    elif field == 'subdistrict_code':
                        missing_friendly.append('postcode')
                
                if missing_friendly:
                    response += f"I'd particularly love to know the {', '.join(missing_friendly[:-1])}"
                    if len(missing_friendly) > 1:
                        response += f" and {missing_friendly[-1]}"
                    else:
                        response += missing_friendly[0] if missing_friendly else ""
                    response += " for a more accurate estimate."

            return response, complete_fields
            
        except Exception as e:
            return f"I can provide an estimate, but encountered an issue: {str(e)}. Please provide more details for a better prediction.", complete_fields

    def create_property_description(self, provided_fields, complete_fields):
        """Create a natural description of the property"""
        property_type = provided_fields.get('PROPERTY TYPE', complete_fields.get('PROPERTY TYPE', 'property')).lower()
        bedrooms = provided_fields.get('BEDROOMS') or complete_fields.get('BEDROOMS')
        
        try:
            bedrooms = int(float(bedrooms)) if bedrooms else 0
        except:
            bedrooms = 0
            
        if bedrooms == 0 or 'studio' in property_type:
            return f"a studio {property_type}"
        else:
            return f"a {bedrooms}-bedroom {property_type}"

    def format_assumptions(self, assumptions):
        """Format the list of missing information naturally"""
        if len(assumptions) == 1:
            if 'bathroom' in assumptions[0]:
                return 'number of bathrooms'
            elif 'bedroom' in assumptions[0]:
                return 'number of bedrooms'
            elif 'sq ft' in assumptions[0]:
                return 'property size'
        return 'exact details'

    def extract_fields(self, user_message, conversation_history, last_candidate_fields=None):
        # Use LangChain's PydanticOutputParser for robust extraction
        from langchain.output_parsers import PydanticOutputParser
        from pydantic import BaseModel, Field, ValidationError
        from langchain_core.prompts import ChatPromptTemplate
        import re
        
        class RentFields(BaseModel):
            address: str = Field("", description="The property address or location")
            subdistrict_code: str = Field("", description="The subdistrict code or postcode")
            BEDROOMS: int = Field(0, description="Number of bedrooms")
            BATHROOMS: int = Field(0, description="Number of bathrooms")
            SIZE: float = Field(0.0, description="Size in square feet")
            PROPERTY_TYPE: str = Field("", description="Property type (e.g. flat, house, apartment)")

        parser = PydanticOutputParser(pydantic_object=RentFields)
        
        # Filter conversation to only rent prediction related messages (last 6 messages max)
        rent_messages = []
        recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        
        for msg in recent_messages:
            content_lower = msg["content"].lower()
            if any(keyword in content_lower for keyword in ["rent", "prediction", "property", "address", "bedroom", "bathroom", "size", "flat", "house", "apartment", "subdistrict", "postcode"]):
                rent_messages.append(msg)
        
        # Add current message
        filtered_text = "\n".join([f"{m['role']}: {m['content']}" for m in rent_messages])
        filtered_text += f"\nuser: {user_message}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant for rent prediction ONLY. Extract ONLY rent prediction fields from the conversation: address, subdistrict_code, BEDROOMS, BATHROOMS, SIZE, PROPERTY_TYPE. DO NOT extract any other fields like credit score, income, employment, maintenance info, etc. If a field is missing, use an empty string or 0. Output only the JSON object as specified by the schema: {format_instructions}"),
            ("user", "Rent prediction conversation:\n{conversation}\nCurrent message:\n{user_message}")
        ])
        format_instructions = parser.get_format_instructions()
        prompt_value = prompt.format_prompt(
            conversation=filtered_text,
            user_message=user_message,
            format_instructions=format_instructions
        )
        response = invoke_with_tracking(
            self.chat,
            [HumanMessage(content=prompt_value.to_string())],
            "Rent Prediction - Field Extraction"
        )
        content = response.content.strip()
        try:
            parsed = parser.parse(content)
            fields = parsed.dict()
        except (ValidationError, Exception) as e:
            print(f"[DEBUG] Pydantic parsing failed for rent fields: {e}. Using fallback extraction.")
            # Fallback: try regex extraction as before
            fields = dict(last_candidate_fields) if last_candidate_fields else {}
            all_text = filtered_text
            markdown_field_pattern = re.compile(r"(?:^|\n)[\-\d\.\*\s]*\*?\*?([A-Za-z0-9_\s]+?)\*?\*?\s*[:：]\s*([\w\-,.\/()''\s]+)", re.IGNORECASE)
            for match in markdown_field_pattern.finditer(all_text):
                raw_field, value = match.group(1).strip(), match.group(2).strip()
                for canonical, synonyms in self.FIELD_SYNONYMS.items():
                    for syn in synonyms:
                        if syn.lower() in raw_field.lower():
                            if canonical in ["BEDROOMS", "BATHROOMS", "SIZE"]:
                                try:
                                    value_num = float(value)
                                    if value_num.is_integer():
                                        value = int(value_num)
                                    else:
                                        value = value_num
                                except Exception:
                                    pass
                            fields[canonical] = value
                            break
            # Fallback: extract from natural language
            for field, synonyms in self.FIELD_SYNONYMS.items():
                if field in fields:
                    continue
                for syn in synonyms:
                    pattern = rf"(?:{syn})\s*[:=\-]?\s*(\d+\.?\d*|[\w\s,.'']+)"
                    match = re.search(pattern, all_text, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        if field in ["BEDROOMS", "BATHROOMS", "SIZE"]:
                            try:
                                value_num = float(value)
                                if value_num.is_integer():
                                    value = int(value_num)
                                else:
                                    value = value_num
                            except Exception:
                                pass
                        fields[field] = value
                        break
        # Map Pydantic/JSON keys to canonical field names if needed
        if "PROPERTY_TYPE" in fields:
            fields["PROPERTY TYPE"] = fields.pop("PROPERTY_TYPE")
        return fields

    def count_provided_fields(self, fields):
        """Count how many meaningful fields are provided"""
        meaningful_fields = 0
        for field in self.required_fields:
            value = fields.get(field)
            if value and value not in (None, '', 0, 0.0):
                meaningful_fields += 1
        return meaningful_fields

    def summarize_fields(self, fields):
        # Summarize in markdown with a professional heading
        summary = "**Property Information for Rent Estimation:**\n\n"
        for k, v in fields.items():
            summary += f"- **{k}**: {v}\n"
        summary += "\nIs this information correct? Please confirm to proceed with the rent estimation."
        return summary

    def encode_fields_for_model(self, fields):
        """
        Map user-friendly fields to encoded values using the mapping files.
        """
        import json
        import os
        # Load mapping files (cache for performance if needed)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI'))
        with open(os.path.join(base_dir, 'address_map.json'), 'r', encoding='utf-8') as f:
            address_map = json.load(f)
        with open(os.path.join(base_dir, 'property_type_map.json'), 'r', encoding='utf-8') as f:
            property_type_map = json.load(f)
        with open(os.path.join(base_dir, 'subdistrict_code_map.json'), 'r', encoding='utf-8') as f:
            subdistrict_code_map = json.load(f)
        encoded = dict(fields)
        # Address: try to match full or partial string
        addr = str(fields.get('address', '')).strip()
        encoded['address'] = address_map.get(addr, address_map.get(addr.upper(), next(iter(address_map.values()))))
        # Subdistrict code: try direct, then upper
        subc = str(fields.get('subdistrict_code', '')).strip()
        encoded['subdistrict_code'] = subdistrict_code_map.get(subc, subdistrict_code_map.get(subc.upper(), next(iter(subdistrict_code_map.values()))))
        # Property type: try direct, then title-case
        ptype = str(fields.get('PROPERTY TYPE', '')).strip()
        encoded['PROPERTY TYPE'] = property_type_map.get(ptype, property_type_map.get(ptype.title(), next(iter(property_type_map.values()))))
        # Ensure numeric fields are correct type
        for k in ['BEDROOMS', 'BATHROOMS', 'SIZE']:
            if k in encoded:
                try:
                    encoded[k] = float(encoded[k])
                    if encoded[k].is_integer():
                        encoded[k] = int(encoded[k])
                except Exception:
                    encoded[k] = 0
        return encoded

    def run_model(self, fields):
        import xgboost as xgb
        import csv
        import os
        print("[DEBUG] User fields (extracted from conversation):", fields)
        encoded_fields = self.encode_fields_for_model(fields)
        print("[DEBUG] Encoded fields for model:", encoded_fields)
        MODEL_FIELDS = [
            "address", "subdistrict_code", "BEDROOMS", "BATHROOMS", "SIZE",
            "PROPERTY TYPE"
        ]
        model_input = {k: encoded_fields[k] for k in MODEL_FIELDS if k in encoded_fields}
        model = self.get_model()
        import pandas as pd
        import numpy as np
        dinput = xgb.DMatrix(pd.DataFrame([model_input]), missing=np.nan)
        predicted_log_rent = model.predict(dinput)
        predicted_rent = np.expm1(predicted_log_rent)
        predicted_rent = predicted_rent[0]
        lower_rent = int(predicted_rent - 0.10 * predicted_rent)
        upper_rent = int(predicted_rent + 0.10 * predicted_rent)
        rmse = 1039.64
        confidence = max(0, 1 - (rmse / predicted_rent))
        confidence_percentage = round(confidence * 100, 2)
        summary = (
            f"- **Estimated Monthly Rent:** £{int(predicted_rent)}\n"
            f"- **Suggested Range:** £{int(lower_rent)}–£{int(upper_rent)}\n"
            f"- **Confidence Level:** {round(float(confidence_percentage), 2)}%\n"
        )
        one_liner = "\n_This estimate is based on your property's size, features, and location._\n"
        explanation = ("\n**How this was calculated:**\n"
            "The suggested rent is determined by analyzing your property's size, number of bedrooms and bathrooms, type, and how close it is to public transport. "
            "Properties with more space, more rooms, and better access to stations generally command higher rents. The confidence score reflects how closely your property matches similar listings in the area.\n"
        )
        follow_ups = (
            "\n---\nWould you like to:\n"
            "- Compare this to similar listings nearby? (Reply or click: compare)\n"
            "- Save this property? (Reply or click: save)\n"
        )
        # Store last prediction for save/compare actions
        self.last_prediction = {
            **fields,
            "predicted_rent": int(predicted_rent),
            "lower_rent": int(lower_rent),
            "upper_rent": int(upper_rent),
            "confidence": confidence_percentage
        }
        return summary + one_liner + explanation + follow_ups

    def handle_followup(self, action, last_prediction=None):
        """
        Handle follow-up actions: 'save', 'compare', or 'both'.
        last_prediction: dict of last prediction fields (from session)
        """
        if not last_prediction:
            return "No property prediction found to process this action. Please estimate rent first."
        import os, csv, pandas as pd, json, re
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI'))
        import json
        with open(os.path.join(base_dir, 'property_type_map.json'), 'r', encoding='utf-8') as f:
            property_type_map = json.load(f)
        # Invert property_type_map for code-to-label lookup
        property_type_code_to_label = {v: k for k, v in property_type_map.items()}
        with open(os.path.join(base_dir, 'subdistrict_code_map.json'), 'r', encoding='utf-8') as f:
            subdistrict_code_map = json.load(f)
        pred = dict(last_prediction)
        subc = str(pred.get('subdistrict_code', '')).strip()
        ptype = str(pred.get('PROPERTY TYPE', '')).strip()
        pred['subdistrict_code'] = subdistrict_code_map.get(subc, subdistrict_code_map.get(subc.upper(), next(iter(subdistrict_code_map.values()))))
        pred['PROPERTY TYPE'] = property_type_map.get(ptype, property_type_map.get(ptype.title(), next(iter(property_type_map.values()))))
        def encode_col(col, mapping):
            return col.map(lambda x: mapping.get(str(x).strip(), mapping.get(str(x).strip().upper(), mapping.get(str(x).strip().title(), None))))
        def try_parse_size(val):
            if pd.isna(val): return None
            if isinstance(val, (int, float)): return float(val)
            m = re.search(r"(\d+(?:\.\d+)?)", str(val))
            return float(m.group(1)) if m else None
        # ...existing code...
        def find_similar(df):
            # Encode columns if not already encoded (robust to raw or encoded data)
            if not pd.api.types.is_integer_dtype(df['subdistrict_code']):
                df['subdistrict_code'] = encode_col(df['subdistrict_code'], subdistrict_code_map)
            if not pd.api.types.is_integer_dtype(df['PROPERTY TYPE']):
                df['PROPERTY TYPE'] = encode_col(df['PROPERTY TYPE'], property_type_map)
            # Parse/clean numeric fields
            df['BEDROOMS'] = pd.to_numeric(df['BEDROOMS'], errors='coerce')
            df['SIZE'] = df['SIZE'].apply(try_parse_size)
            # Filter
            return df[
                (df['subdistrict_code'] == pred['subdistrict_code']) &
                (df['PROPERTY TYPE'] == pred['PROPERTY TYPE']) &
                (df['BEDROOMS'].between(int(pred['BEDROOMS']) - 1, int(pred['BEDROOMS']) + 1)) &
                (df['SIZE'].between(float(pred['SIZE']) * 0.8, float(pred['SIZE']) * 1.2))
            ]
        if action == "save":
            # ...existing code...
            return "✅ Property and prediction saved!"
        elif action == "compare":
            data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/data/cleaned_rent_data.csv'))
            df = pd.read_csv(data_path)
            similar = find_similar(df)
            if similar.empty:
                raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/data/rent_ads_rightmove_extended.csv'))
                if os.path.exists(raw_path):
                    raw_df = pd.read_csv(raw_path)
                    for col in ['subdistrict_code', 'PROPERTY TYPE', 'BEDROOMS', 'SIZE', 'address', 'rent']:
                        if col not in raw_df.columns:
                            raw_df[col] = None
                    similar = find_similar(raw_df)
            if similar.empty:
                # Use FAISS semantic search over both cleaned and raw data
                try:
                    from faiss_utils import semantic_search, load_faiss_index, record_to_text
                    # Load human-readable address map for decoding
                    address_map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/address_map_human.json'))
                    with open(address_map_path, 'r', encoding='utf-8') as f:
                        address_map = json.load(f)
                    # Invert the address map: code (as int, str, float) -> address string
                    inv_address_map = {}
                    for addr, code in address_map.items():
                        inv_address_map[code] = addr
                        try:
                            inv_address_map[int(code)] = addr
                        except Exception:
                            pass
                        try:
                            inv_address_map[str(code)] = addr
                        except Exception:
                            pass
                    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/data/cleaned_rent_data.faiss'))
                    summary = ""  # Ensure summary is always defined
                    if os.path.exists(index_path):
                        print("[DEBUG] Using FAISS index: cleaned data")
                        index = load_faiss_index(index_path)
                        df = pd.read_csv(data_path)
                        query_text = record_to_text(pred)
                        faiss_results = semantic_search(query_text, index, df, top_k=5)
                        # Compose a brief summary using LLM
                        similar_listings = []
                        rents = []
                        for _, row in faiss_results.iterrows() if hasattr(faiss_results, 'iterrows') else [(None, faiss_results)]:
                            addr_code = row.get('address','')
                            addr_str = inv_address_map.get(addr_code)
                            if addr_str is None:
                                try:
                                    addr_str = inv_address_map.get(int(float(addr_code)))
                                except Exception:
                                    addr_str = None
                            if addr_str is None:
                                addr_str = str(addr_code)
                            # Map property type code to label, handle -1 and missing
                            ptype_code = row.get('PROPERTY TYPE','')
                            try:
                                ptype_code_int = int(float(ptype_code))
                            except Exception:
                                ptype_code_int = -1
                            ptype_label = property_type_code_to_label.get(ptype_code_int, 'Unknown')
                            similar_listings.append({
                                'address': addr_str,
                                'BEDROOMS': row.get('BEDROOMS',''),
                                'BATHROOMS': row.get('BATHROOMS',''),
                                'SIZE': row.get('SIZE',''),
                                'PROPERTY TYPE': ptype_label,
                                'rent': row.get('rent','')
                            })
                            try:
                                rents.append(float(row.get('rent','')))
                            except Exception:
                                pass
                        user_rent = float(pred.get('predicted_rent', 0))
                        # LLM summary
                        llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=openai_api_key)
                        summary_prompt = f"""
                            You are a real estate assistant. Compare the user's property (rent: £{user_rent}) to these similar listings (rents: {[l['rent'] for l in similar_listings]}). In 1-2 sentences, summarize if the user's price is above, below, or in line with the local market, and mention any notable differences in features if possible. Be concise and helpful.
                            """
                        summary_response = invoke_with_tracking(
                            llm,
                            [HumanMessage(content=summary_prompt)],
                            "Rent Comparison - Market Summary (Save)"
                        )
                        summary = summary_response.content.strip()
                        out = "✅ Property and prediction saved!\n"  # Separate line
                        out += summary + "\n\n"  # LLM summary replaces section title
                        for l in similar_listings:
                            out += (
                                f"- Address: {l['address']}, Bedrooms: {l['BEDROOMS']}, Bathrooms: {l['BATHROOMS']}, Size: {l['SIZE']} sq ft, Property Type: {l['PROPERTY TYPE']}, Rent: £{l['rent']}\n"
                            )
                        return out
                    raw_index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/data/rent_ads_rightmove_extended.faiss'))
                    if os.path.exists(raw_index_path):
                        print("[DEBUG] Using FAISS index: raw data")
                        index = load_faiss_index(raw_index_path)
                        raw_df = pd.read_csv(raw_path)
                        query_text = record_to_text(pred)
                        faiss_results = semantic_search(query_text, index, raw_df, top_k=5)
                        out = "✅ Property and prediction saved!\n"  # Separate line
                        out += summary + "\n\n"  # LLM summary replaces section title (may be empty)
                        for l in similar_listings if 'similar_listings' in locals() else []:
                            out += (
                                f"- Address: {l['address']}, Bedrooms: {l['BEDROOMS']}, Bathrooms: {l['BATHROOMS']}, Size: {l['SIZE']} sq ft, Property Type: {l['PROPERTY TYPE']}, Rent: £{l['rent']}\n"
                            )
                        return out
                    return "✅ Property and prediction saved!\n\nNo similar listings found in local data. (No FAISS index available.)"
                except Exception as e:
                    return f"✅ Property and prediction saved!\n\nNo similar listings found and semantic search failed: {e}"
            out = "✅ Property and prediction saved!\n\n**Similar Listings Nearby:**\n\n"
            for _, row in similar.head(5).iterrows():
                out += (
                    f"- Address: {row.get('address','')}, Bedrooms: {row.get('BEDROOMS','')}, Bathrooms: {row.get('BATHROOMS','')}, Size: {row.get('SIZE','')} sq ft, Property Type: {row.get('PROPERTY TYPE','')}, Rent: £{row.get('rent','')}\n"
                )
            return out
        elif action == "both":
            save_path = os.path.join(os.path.dirname(__file__), "saved_properties.csv")
            file_exists = os.path.isfile(save_path)
            with open(save_path, mode="a", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(last_prediction.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(last_prediction)
            data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/data/cleaned_rent_data.csv'))
            df = pd.read_csv(data_path)
            similar = find_similar(df)
            if similar.empty:
                raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/data/rent_ads_rightmove_extended.csv'))
                if os.path.exists(raw_path):
                    raw_df = pd.read_csv(raw_path)
                    for col in ['subdistrict_code', 'PROPERTY TYPE', 'BEDROOMS', 'SIZE', 'address', 'rent']:
                        if col not in raw_df.columns:
                            raw_df[col] = None
                    similar = find_similar(raw_df)
            if similar.empty:
                try:
                    from faiss_utils import semantic_search, load_faiss_index, record_to_text
                    address_map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/address_map_human.json'))
                    with open(address_map_path, 'r', encoding='utf-8') as f:
                        address_map = json.load(f)
                    inv_address_map = {}
                    for addr, code in address_map.items():
                        inv_address_map[code] = addr
                        try:
                            inv_address_map[int(code)] = addr
                        except Exception:
                            pass
                        try:
                            inv_address_map[str(code)] = addr
                        except Exception:
                            pass
                    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/data/cleaned_rent_data.faiss'))
                    summary = ""  # Ensure summary is always defined
                    if os.path.exists(index_path):
                        print("[DEBUG] Using FAISS index: cleaned data")
                        index = load_faiss_index(index_path)
                        df = pd.read_csv(data_path)
                        query_text = record_to_text(pred)
                        faiss_results = semantic_search(query_text, index, df, top_k=5)
                        out = "✅ Property and prediction saved!\n"  # Separate line
                        # Compose a brief summary using LLM
                        similar_listings = []
                        rents = []
                        for _, row in faiss_results.iterrows() if hasattr(faiss_results, 'iterrows') else [(None, faiss_results)]:
                            addr_code = row.get('address','')
                            addr_str = inv_address_map.get(addr_code)
                            if addr_str is None:
                                try:
                                    addr_str = inv_address_map.get(int(float(addr_code)))
                                except Exception:
                                    addr_str = None
                            if addr_str is None:
                                addr_str = str(addr_code)
                            # Map property type code to label, handle -1 and missing
                            ptype_code = row.get('PROPERTY TYPE','')
                            try:
                                ptype_code_int = int(float(ptype_code))
                            except Exception:
                                ptype_code_int = -1
                            ptype_label = property_type_code_to_label.get(ptype_code_int, 'Unknown')
                            similar_listings.append({
                                'address': addr_str,
                                'BEDROOMS': row.get('BEDROOMS',''),
                                'BATHROOMS': row.get('BATHROOMS',''),
                                'SIZE': row.get('SIZE',''),
                                'PROPERTY TYPE': ptype_label,
                                'rent': row.get('rent','')
                            })
                            try:
                                rents.append(float(row.get('rent','')))
                            except Exception:
                                pass
                        user_rent = float(pred.get('predicted_rent', 0))
                        # LLM summary
                        llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=openai_api_key)
                        summary_prompt = f"""
                            You are a real estate assistant. Compare the user's property (rent: £{user_rent}) to these similar listings (rents: {[l['rent'] for l in similar_listings]}). In 1-2 sentences, summarize if the user's price is above, below, or in line with the local market, and mention any notable differences in features if possible. Be concise and helpful.
                            """
                        summary_response = invoke_with_tracking(
                            llm,
                            [HumanMessage(content=summary_prompt)],
                            "Rent Comparison - Market Summary (Compare)"
                        )
                        summary = summary_response.content.strip()
                        out += summary + "\n\n"  # LLM summary replaces section title
                        for l in similar_listings:
                            out += (
                                f"- Address: {l['address']}, Bedrooms: {l['BEDROOMS']}, Bathrooms: {l['BATHROOMS']}, Size: {l['SIZE']} sq ft, Property Type: {l['PROPERTY TYPE']}, Rent: £{l['rent']}\n"
                            )
                        return out
                    raw_index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/data/rent_ads_rightmove_extended.faiss'))
                    if os.path.exists(raw_index_path):
                        print("[DEBUG] Using FAISS index: raw data")
                        index = load_faiss_index(raw_index_path)
                        raw_df = pd.read_csv(raw_path)
                        query_text = record_to_text(pred)
                        faiss_results = semantic_search(query_text, index, raw_df, top_k=5)
                        out = "✅ Property and prediction saved!\n"  # Separate line
                        out += summary + "\n\n"  # LLM summary replaces section title (may be empty)
                        for l in similar_listings if 'similar_listings' in locals() else []:
                            out += (
                                f"- Address: {l['address']}, Bedrooms: {l['BEDROOMS']}, Bathrooms: {l['BATHROOMS']}, Size: {l['SIZE']} sq ft, Property Type: {l['PROPERTY TYPE']}, Rent: £{l['rent']}\n"
                            )
                        return out
                    return "✅ Property and prediction saved!\n\nNo similar listings found in local data. (No FAISS index available.)"
                except Exception as e:
                    return f"✅ Property and prediction saved!\n\nNo similar listings found and semantic search failed: {e}"
            out = "✅ Property and prediction saved!\n\n**Similar Listings Nearby:**\n\n"
            for _, row in similar.head(5).iterrows():
                out += (
                    f"- Address: {row.get('address','')}, Bedrooms: {row.get('BEDROOMS','')}, Bathrooms: {row.get('BATHROOMS','')}, Size: {row.get('SIZE','')} sq ft, Property Type: {row.get('PROPERTY TYPE','')}, Rent: £{row.get('rent','')}\n"
                )
            return out
        else:
            return "Unknown follow-up action."

    def needs_confirmation(self, user_message):
        # Only treat as confirmation if the user is confirming the information, not to trigger the model
        confirmation_phrases = ["yes", "correct", "that's right", "yep", "confirmed", "go ahead", "proceed"]
        return user_message.strip().lower() in confirmation_phrases

    def should_run_model(self, conversation_history, candidate_fields):
        # Only run the model if all required fields are present and the last assistant message explicitly asks for confirmation
        if not candidate_fields or not all(f in candidate_fields and candidate_fields[f] not in (None, '', 0, 0.0) for f in self.required_fields):
            return False
        if not conversation_history:
            return False
        last_assistant = next((m for m in reversed(conversation_history) if m["role"] == "assistant"), None)
        if not last_assistant:
            return False
        confirmation_keywords = [
            "please confirm", "is this information correct", "is this correct", "can you confirm", "are these details correct"
        ]
        return any(kw in last_assistant["content"].lower() for kw in confirmation_keywords)

    def handle(self, conversation_history, user_message, last_candidate_fields=None):
        candidate_fields = self.extract_fields(user_message, conversation_history, last_candidate_fields)
        # Only keep rent fields
        rent_fields = {k: v for k, v in (last_candidate_fields or candidate_fields).items() if k in self.required_fields}
        
        # Count meaningful provided fields
        provided_count = self.count_provided_fields(rent_fields)
        
        # Run model if user confirms and all required fields are present (ignore last assistant message)
        if self.needs_confirmation(user_message):
            fields = rent_fields
            if all(f in fields and fields[f] not in (None, '', 0, 0.0) for f in self.required_fields):
                result = self.run_model(fields)
                return {"response": result, "action": "rent_prediction", "fields": fields}
            else:
                missing = [f for f in self.required_fields if f not in fields or fields[f] in (None, '', 0, 0.0)]
                return {"response": f"I need the following details to estimate rent: {', '.join(missing)}. Please provide them.", "action": "ask_for_info", "fields": fields}
        
        # NEW FEATURE: Provide estimate with defaults if at least 2 fields are provided
        if provided_count >= 2:
            try:
                estimate_response, complete_fields = self.create_estimate_with_defaults(rent_fields)
                return {"response": estimate_response, "action": "estimate_with_defaults", "fields": complete_fields}
            except Exception as e:
                print(f"[DEBUG] Estimate with defaults failed: {e}")
                # Fall back to normal LLM flow
                pass
        
        # Otherwise, continue the LLM-driven flow
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        response = invoke_with_tracking(
            self.chat,
            [HumanMessage(content=m["content"]) for m in messages],
            "Rent Prediction - Conversation"
        )
        reply = response.content.strip()
        extracted = self.extract_fields(reply, conversation_history, candidate_fields)
        return {"response": reply, "action": "chat", "fields": extracted}

class TenantScreeningHandler(BaseModuleHandler):
    required_fields = ["credit_score", "income", "rent", "employment_status", "eviction_record"]
    FIELD_SYNONYMS = {
        "credit_score": ["credit score", "credit rating", "score"],
        "income": ["income", "annual income", "salary", "yearly income", "monthly income"],
        "rent": ["rent", "monthly rent", "expected rent", "property rent", "asking rent"],
        "employment_status": ["employment status", "job", "occupation", "employed", "unemployed", "self-employed", "work status"],
        "eviction_record": ["eviction record", "prior eviction", "evicted", "has eviction", "any eviction", "eviction", "has prior eviction", "previous eviction", "eviction history"]
    }
    
    def __init__(self):
        self.system_prompt = (
            "You are LandlordBuddy, a friendly and knowledgeable AI companion for landlords! "
            "Think of me as your trusted sidekick who's here to help with rent pricing, tenant screening, and maintenance predictions. "
            "When it comes to tenant screening, I'm like your personal detective - I gather all the essential clues (credit score, income, rent amount, employment situation, and any eviction history), "
            "chat with you about what we've found, and then run our comprehensive tenant evaluation process. "
            "Here's the thing - I never jump to my own conclusions or give you my personal take on things. That's not my style! "
            "If our evaluation can't run because we're missing some key pieces of the puzzle, I'll simply let you know: "
            "'Looks like we're missing some crucial details to run a proper tenant screening. Mind filling in the gaps?' "
            "When our evaluation is complete, I'll present the findings in a clean, easy-to-read format and help clarify what it all means - "
            "but I stick to explaining the results, not adding my own spin on things. "
            "I keep things natural and conversational - no robotic 'processing' talk or mentions of scripts and models. "
            "I'm your professional yet approachable partner in this! "
            "I love being efficient and interactive, so I won't keep you waiting with unnecessary delays. "
            "I always check with you before diving into the evaluation - your confirmation is my green light! "
            "I focus on what really matters - those five key areas I mentioned - and keep things streamlined. "
            "When we start a new screening, I'll lay out everything we need in one go, nice and organized. "
            "No back-and-forth hunting for details one piece at a time - that's just inefficient! "
        )
        self.chat = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

    def extract_fields(self, user_message, conversation_history, last_candidate_fields=None):
        # Use LLM to extract fields in a structured way, similar to rent prediction
        from langchain.output_parsers import PydanticOutputParser
        from pydantic import BaseModel, Field, ValidationError
        from langchain_core.prompts import ChatPromptTemplate
        import re
        
        class TenantFields(BaseModel):
            credit_score: int = Field(0, description="Applicant's credit score")
            income: float = Field(0, description="Applicant's monthly income")
            rent: float = Field(0, description="Monthly rent for the property")
            employment_status: str = Field("", description="Employment status (e.g., employed, unemployed)")
            eviction_record: bool = Field(False, description="True if applicant has prior eviction, else False")
            tenant_name: str = Field("", description="Tenant's name if mentioned")
        
        parser = PydanticOutputParser(pydantic_object=TenantFields)
        
        # Filter conversation to only tenant screening related messages (last 6 messages max)
        tenant_messages = []
        recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        
        for msg in recent_messages:
            content_lower = msg["content"].lower()
            if any(keyword in content_lower for keyword in ["tenant", "screening", "credit", "income", "rent", "employment", "eviction", "unemployed", "employed"]):
                tenant_messages.append(msg)
        
        # Always include current message and add some context to ensure proper extraction
        all_text = "\n".join([f"{m['role']}: {m['content']}" for m in tenant_messages])
        all_text += f"\nuser: {user_message}"
        
        # Add explicit context that this is for tenant screening to help LLM focus
        context_text = "Context: This is a tenant screening conversation. Extract tenant screening fields (credit_score, income, rent, employment_status, eviction_record).\n" + all_text
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant for tenant screening ONLY. Extract ONLY tenant screening fields from the conversation: credit_score, income, rent, employment_status, eviction_record, and tenant_name if mentioned. DO NOT extract any other fields like address, property age, maintenance, etc. If a field is missing, use 0, empty string, or False. If a tenant name is mentioned, extract it. Output only the JSON object as specified by the schema: {format_instructions}"),
            ("user", "Tenant screening conversation:\n{conversation}\nCurrent message:\n{user_message}")
        ])
        format_instructions = parser.get_format_instructions()
        prompt_value = prompt.format_prompt(
            conversation=context_text,
            user_message=user_message,
            format_instructions=format_instructions
        )
        response = invoke_with_tracking(
            self.chat,
            [HumanMessage(content=prompt_value.to_string())],
            "Tenant Screening - Field Extraction"
        )
        content = response.content.strip()
        try:
            parsed = parser.parse(content)
            fields = parsed.dict()
        except (ValidationError, Exception) as e:
            print(f"[DEBUG] Pydantic parsing failed for tenant fields: {e}. Using fallback extraction.")
            # Fallback: regex extraction as before
            fields = dict(last_candidate_fields) if last_candidate_fields else {}
            all_text = context_text + f"\n{user_message}"
            markdown_field_pattern = re.compile(r"(?:^|\n)[\-\d\.\*\s]*\*?\*?([A-Za-z0-9_\s]+?)\*?\*?\s*[:：]\s*([\w\-,.\/()''\s]+)", re.IGNORECASE)
            for match in markdown_field_pattern.finditer(all_text):
                raw_field, value = match.group(1).strip(), match.group(2).strip()
                for canonical, synonyms in self.FIELD_SYNONYMS.items():
                    for syn in synonyms:
                        if syn.lower() in raw_field.lower():
                            if canonical in ["credit_score", "income", "rent"]:
                                try:
                                    value = float(re.sub(r"[^\d.]", "", value))
                                except Exception:
                                    pass
                            if canonical == "eviction_record":
                                value = value.lower()
                                value = any(word in value for word in ["yes", "true", "prior", "evict", "bad", "negative"])
                            fields[canonical] = value
                            break
            # Fallback: extract from natural language
            for field, synonyms in self.FIELD_SYNONYMS.items():
                if field in fields:
                    continue
                for syn in synonyms:
                    pattern = rf"(?:{syn})\s*[:=\-]?\s*([\w\-,.\/()''\s]+)"
                    match = re.search(pattern, all_text, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        if field in ["credit_score", "income", "rent"]:
                            try:
                                value = float(re.sub(r"[^\d.]", "", value))
                            except Exception:
                                pass
                        if field == "eviction_record":
                            value = value.lower()
                            value = any(word in value for word in ["yes", "true", "prior", "evict", "bad", "negative"])
                        fields[field] = value
                        break
        
        # Only keep required fields and ensure correct types, plus tenant_name
        clean_fields = {}
        for k in self.required_fields + ["tenant_name"]:
            if k == "tenant_name":
                v = fields.get(k, "")
                clean_fields[k] = str(v).strip()
                continue
                
            v = fields.get(k, 0 if k in ["credit_score", "income", "rent"] else (False if k == "eviction_record" else ""))
            # Special handling for income: if string like '50 a month', extract number
            if k == "income" and isinstance(v, str):
                import re
                match = re.search(r"(\d+(?:\.\d+)?)", v)
                if match:
                    v = float(match.group(1))
                else:
                    v = 0.0
            # Ensure correct types
            if k in ["credit_score", "rent"]:
                try:
                    v = int(float(v))
                except Exception:
                    v = 0
            if k == "income":
                try:
                    v = float(v)
                except Exception:
                    v = 0.0
            if k == "eviction_record":
                v = bool(v)
            if k == "employment_status":
                v = str(v)
            clean_fields[k] = v
        return clean_fields

    def summarize_fields(self, fields, user_provided=None, show_result=False):
        """
        Summarize fields for display with more conversational language
        user_provided: list of fields actually provided by user (not assumed)
        show_result: if True, don't add confirmation text (we're showing final result)
        """
        if user_provided is None:
            user_provided = []
        assumed_fields = [k for k in self.required_fields if k not in user_provided]
        
        if show_result:
            summary = "**Here's what we're working with:**\n\n"
        elif assumed_fields:
            summary = "**Here's the picture so far (I've filled in some gaps with typical values):**\n\n"
        else:
            summary = "**Perfect! Here's everything we've got:**\n\n"
        
        # Format each field with more conversational language
        for k in self.required_fields:
            v = fields.get(k, None)
            assumed = k in assumed_fields
            if k == "credit_score":
                if v not in (None, '', 0, 0.0):
                    if assumed and not show_result:
                        summary += f"- **Credit Score:** {v} _(I'm using a typical score since we don't have this one)_\n"
                    else:
                        summary += f"- **Credit Score:** {v}\n"
                else:
                    summary += f"- **Credit Score:** _(we'll need this one)_\n"
            elif k == "income":
                if v not in (None, '', 0, 0.0):
                    if assumed and not show_result:
                        summary += f"- **Monthly Income:** £{float(v):,.2f} _(estimated typical income)_\n"
                    else:
                        summary += f"- **Monthly Income:** £{float(v):,.2f}\n"
                else:
                    summary += f"- **Monthly Income:** _(still need this detail)_\n"
            elif k == "rent":
                if v not in (None, '', 0, 0.0):
                    if assumed and not show_result:
                        summary += f"- **Monthly Rent:** £{float(v):,.2f} _(using average rent estimate)_\n"
                    else:
                        summary += f"- **Monthly Rent:** £{float(v):,.2f}\n"
                else:
                    summary += f"- **Monthly Rent:** _(missing this piece)_\n"
            elif k == "employment_status":
                if v not in (None, '', False, 'unknown', ''):
                    if assumed and not show_result:
                        summary += f"- **Employment:** {v} _(my best guess based on what we know)_\n"
                    else:
                        summary += f"- **Employment:** {v}\n"
                else:
                    summary += f"- **Employment:** _(would love to know this)_\n"
            elif k == "eviction_record":
                if v is False:
                    if assumed and not show_result:
                        summary += f"- **Eviction History:** Clean record _(assuming no issues unless told otherwise)_\n"
                    else:
                        summary += f"- **Eviction History:** Clean record\n"
                elif v is True:
                    summary += f"- **Eviction History:** Has prior evictions\n"
                else:
                    summary += f"- **Eviction History:** _(need to check this)_\n"
        
        # More conversational confirmation text
        if not show_result:
            missing_to_show = [k for k in self.required_fields if k not in user_provided]
            if missing_to_show:
                summary += "\n💡 _Want a spot-on screening? Just share these missing bits with me:_ "
                field_names = {
                    "credit_score": "**Credit Score**",
                    "income": "**Income**", 
                    "rent": "**Rent Amount**",
                    "employment_status": "**Job Status**",
                    "eviction_record": "**Eviction History**"
                }
                summary += ", ".join([f"_{field_names.get(k, k.replace('_', ' ').title())}_" for k in missing_to_show])
                summary += "\n\n_Or we can roll with what we have - just give me the thumbs up!_ 👍"
            else:
                summary += "\n\nLooks good to me! Ready to dive into the screening? Just say the word! 🚀"
        
        return summary

    def run_model(self, fields):
        from tenant_screening import screen_tenant  # Absolute import for script/module compatibility
        credit_score = int(fields.get("credit_score", 0) or 0)
        income = float(fields.get("income", 0) or 0)
        rent = float(fields.get("rent", 0) or 0)
        employment_status = str(fields.get("employment_status", "")).strip() or "unknown"
        eviction_record = bool(fields.get("eviction_record", False))
        tenant_name = str(fields.get("tenant_name", "")).strip()
        
        print("Starting script with fields:", fields)
        print(f"Credit Score: {credit_score}, Income: {income}, Rent: {rent}, Employment Status: {employment_status}, Eviction Record: {eviction_record}")
        result = screen_tenant(credit_score, income, rent, employment_status, eviction_record)
        print("Done with script")
        
        # More engaging and conversational result summaries
        tenant_reference = tenant_name if tenant_name else "your applicant"
        summary = ""
        if result['recommendation'].lower() in ['approve', 'accept', 'approved', 'accepted']:
            summary = f"🎉 **Great news!** {tenant_reference} gets the green light! Here's the scoop:"
        elif result['recommendation'].lower() == 'review':
            summary = f"🤔 **Hmm, this one needs a closer look...** {tenant_reference} has some yellow flags we should chat about:"
        else:
            summary = f"😬 **Not looking promising...** {tenant_reference} doesn't quite make the cut. Here's what's concerning:"
        
        md = f"{summary}\n\n**📋 Screening Results:**\n\n"
        md += f"**🎯 Final Call:** {result['recommendation'].title()}\n\n"
        md += f"**⚠️ Risk Level:** {result['risk_score']}\n\n"
        md += f"**🔍 Here's the breakdown:**\n\n"
        for line in result['explanation'].split('\n'):
            if line.strip():  # Only add non-empty lines
                md += f"• {line.strip()}\n"
        md += "\n"  # Add extra space after breakdown
        return md

    def handle(self, conversation_history, user_message, last_candidate_fields=None):
        print(f"[DEBUG] TenantScreeningHandler.handle called with: {user_message}")
        print(f"[DEBUG] Last candidate fields: {last_candidate_fields}")
        
        # Only use tenant screening fields for logic (preserve tenant_name in merged_fields)
        new_fields = self.extract_fields(user_message, conversation_history, last_candidate_fields)
        print(f"[DEBUG] Extracted new fields: {new_fields}")
        
        merged_fields = {k: v for k, v in (last_candidate_fields or {}).items() if k in self.required_fields + ["tenant_name"] and v not in (None, '', 0, 0.0, False)}
        for k in self.required_fields + ["tenant_name"]:
            v = new_fields.get(k, None)
            if k == "tenant_name":
                if v and str(v).strip():
                    merged_fields[k] = str(v).strip()
            elif v not in (None, '', 0, 0.0, False):
                merged_fields[k] = v
        
        print(f"[DEBUG] Merged fields: {merged_fields}")

        # Robust eviction record extraction: check user message for negative/false/no eviction record
        eviction_phrases = [
            "no eviction", "no prior eviction", "never evicted", "no eviction record", 
            "eviction record is false", "no enviction", "no inviction", "no eviction history", 
            "false", "none", "never"
        ]
        user_message_lower = user_message.lower()
        if any(phrase in user_message_lower for phrase in eviction_phrases):
            merged_fields["eviction_record"] = False

        # Infer employment status from income
        if merged_fields.get("income", 0) not in (None, '', 0, 0.0):
            if merged_fields.get("employment_status", "") in (None, '', 0, 0.0):
                merged_fields["employment_status"] = "employed"

        # Track which fields were actually provided by the user vs. inferred/assumed
        user_provided = []
        
        # Check if this is a direct screening request (contains action words + data)
        action_words = ["screen", "screening", "check", "evaluate", "assess", "analyze", "review"]
        is_direct_request = any(word in user_message.lower() for word in action_words)
        
        for k in self.required_fields:
            v = merged_fields.get(k, None)
            # Check if this field appears to be user-provided in current or previous messages
            field_keywords = self.FIELD_SYNONYMS.get(k, [k])
            
            # Check current message for field mentions
            current_has_field = any(keyword.lower() in user_message.lower() for keyword in field_keywords)
            
            # Check recent conversation for field mentions
            recent_has_field = False
            if conversation_history:
                recent_messages = conversation_history[-5:]  # Check last 5 messages
                for msg in recent_messages:
                    if msg["role"] == "user":
                        if any(keyword.lower() in msg["content"].lower() for keyword in field_keywords):
                            recent_has_field = True
                            break
            
            # Enhanced detection for direct requests with numeric values
            if k in ["income", "rent"] and v not in (None, '', 0, 0.0):
                # If we extracted a numeric value and this is a direct request, consider it provided
                if is_direct_request:
                    user_provided.append(k)
                elif current_has_field or recent_has_field:
                    user_provided.append(k)
            elif k == "credit_score" and v not in (None, '', 0, 0.0):
                if current_has_field or recent_has_field:
                    user_provided.append(k)
            elif k == "eviction_record":
                # Check for eviction-related phrases in current message
                eviction_mentions = ["eviction", "evicted", "prior eviction", "previous eviction", "eviction record", "eviction history"]
                current_mentions_eviction = any(phrase in user_message.lower() for phrase in eviction_mentions)
                
                if current_mentions_eviction or current_has_field or recent_has_field:
                    user_provided.append(k)
                elif any(phrase in user_message_lower for phrase in eviction_phrases):
                    user_provided.append(k)
            elif k == "employment_status":
                # Only consider provided if explicitly mentioned (not just inferred from income)
                employment_mentions = ["employed", "unemployed", "job", "work", "employment", "self-employed", "occupation"]
                current_mentions_employment = any(phrase in user_message.lower() for phrase in employment_mentions)
                
                if current_mentions_employment or current_has_field or recent_has_field:
                    user_provided.append(k)
        
        print(f"[DEBUG] User provided fields: {user_provided}")

        # Find missing fields
        missing = []
        for k in self.required_fields:
            v = merged_fields.get(k, None)
            if k == "eviction_record":
                if v is None:
                    missing.append(k)
            elif k == "employment_status":
                if v in (None, '', False, 'unknown', ''):
                    missing.append(k)
            else:
                if v in (None, '', 0, 0.0, False):
                    missing.append(k)

        # If not enough info, ask for more with conversational flair
        if len(user_provided) < 2:
            conversation_starters = [
                "Alright, let's get this tenant screening party started! 🎉",
                "Time to put on my detective hat and screen this tenant! 🕵️",
                "Let's dive into some tenant screening magic! ✨",
                "Ready to uncover the mystery of this potential tenant? 🔍"
            ]
            import random
            starter = random.choice(conversation_starters)
            
            prompt = (
                f"{starter}\n\n"
                "I'll need some key details about your applicant to work my screening magic. "
                "Just share at least **two** of these with me and I can get started:\n\n"
            )
            
            field_descriptions = {
                "credit_score": "**Credit Score** - What's their credit situation like?",
                "income": "**Monthly Income** - How much are they bringing in each month?",
                "rent": "**Rent Amount** - What's the monthly rent for this property?",
                "employment_status": "**Job Status** - Are they employed, self-employed, or between jobs?",
                "eviction_record": "**Eviction History** - Any past evictions we should know about?"
            }
            
            for k in self.required_fields:
                prompt += f"• {field_descriptions[k]}\n"
            
            prompt += "\nOnce you drop a couple of these details on me, I'll whip up a preliminary screening for you! 🚀"
            
            return {
                "response": prompt,
                "action": "ask_for_info",
                "fields": merged_fields
            }

        # If we have at least 2 fields, show estimated screening with results
        if len(user_provided) >= 2:
            # Fill in missing fields with defaults for estimation
            avg_defaults = {
                "credit_score": 650,
                "income": 2500.0,
                "rent": 1200.0,
                "employment_status": "employed",
                "eviction_record": False
            }
            estimate_fields = dict(merged_fields)
            for k in missing:
                estimate_fields[k] = avg_defaults[k]

            # Show the details used for screening + result with more personality
            result_md = self.run_model(estimate_fields)
            
            # Add missing fields note at the bottom with conversational touch
            missing_to_show = [k for k in missing if k != "employment_status" or merged_fields.get("income", 0) in (None, '', 0, 0.0)]
            
            response = result_md
            
            if missing_to_show:
                response += "\n\n---\n\n"
                response += "💭 **Want an even sharper analysis?** Drop me these details and I'll give you the full picture: "
                field_names = {
                    "credit_score": "**credit score**",
                    "income": "**income**", 
                    "rent": "**rent amount**",
                    "employment_status": "**employment status**",
                    "eviction_record": "**eviction history**"
                }
                response += ", ".join([f"{field_names.get(k, k.replace('_', ' '))}" for k in missing_to_show])
                response += " and I'll re-run everything! 🎯"
            
            return {
                "response": response,
                "action": "show_estimate_with_result",
                "fields": estimate_fields
            }

        # Fallback to LLM conversation
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        response = invoke_with_tracking(
            self.chat,
            [HumanMessage(content=m["content"]) for m in messages],
            "Tenant Screening - Conversation"
        )
        reply = response.content.strip()

        # Remove any problematic phrases and replace with more conversational alternatives
        replacements = {
            "please wait": "hang tight",
            "processing": "working on it",
            "hold on": "give me a sec",
            "one moment": "just a moment",
            "I'll process": "I'll work on",
            "wait a moment": "bear with me",
            "it's important to exercise caution": "worth being careful here",
            "you may wish to consider": "you might want to think about",
            "you may want to explore": "might be worth looking into",
            "consider requesting a guarantor": "maybe ask for a guarantor",
            "By considering these factors": "Looking at all this",
            "Tips for Landlord": "Quick thoughts",
            "Based on the information provided": "From what you've told me",
            "Summary:": "Here's the deal:"
        }
        
        for old_phrase, new_phrase in replacements.items():
            reply = reply.replace(old_phrase, new_phrase)
            reply = reply.replace(old_phrase.capitalize(), new_phrase.capitalize())

        # Remove lines with extra fields we don't want
        lines = reply.split('\n')
        filtered_lines = []
        for line in lines:
            if any(field in line.lower() for field in ["full name", "rental history", "name:", "history:", "annual income", "tenant's name"]):
                continue
            filtered_lines.append(line)
        reply = '\n'.join(filtered_lines)

        return {"response": reply, "action": "chat", "fields": merged_fields}

    def _is_field_filled(self, key, value):
        if key in ["credit_score", "income", "rent"]:
            return value not in (None, '', 0, 0.0)
        if key == "eviction_record":
            return value is not None  # Boolean, so False is a valid value
        if key == "employment_status":
            return bool(value and str(value).strip())
        return value not in (None, '', False)

    def needs_confirmation(self, user_message):
        confirmation_keywords = [
            "yes", "correct", "confirm", "ok", "okay", "proceed", "continue", "that's right",
            "sounds good", "let's go", "do it", "sure", "absolutely", "yep", "yeah", 
            "thumbs up", "green light", "go ahead", "let's roll", "perfect"
        ]
        return any(kw in user_message.lower().strip() for kw in confirmation_keywords)

    def should_run_model(self, conversation_history, candidate_fields):
        # Only run the model if all required fields are present and the last assistant message explicitly asks for confirmation
        if not candidate_fields or not all(f in candidate_fields and candidate_fields[f] not in (None, '', 0, 0.0, False) for f in self.required_fields):
            return False
        if not conversation_history:
            return False
        last_assistant = next((m for m in reversed(conversation_history) if m["role"] == "assistant"), None)
        if not last_assistant:
            return False
        confirmation_keywords = [
            "please confirm", "is this information correct", "is this correct", "can you confirm", 
            "are these details correct", "ready to dive into the screening", "just say the word",
            "give me the thumbs up", "just give me the thumbs up"
        ]
        return any(kw in last_assistant["content"].lower() for kw in confirmation_keywords)


class MaintenancePredictionHandler(BaseModuleHandler):
    def batch_alerts(self, as_json=False):
        """
        Compatibility stub: returns no alerts. Replace with real logic if batch maintenance alerts are needed.
        """
        if as_json:
            return []
        return "No urgent maintenance alerts at this time."
    
    required_fields = [
        'address', 'age_years', 'last_service_years_ago', 'seasonality'
    ]
    
    FIELD_SYNONYMS = {
        'address': ['address', 'property address', 'location', 'property', 'place', 'where'],
        'age_years': ['age', 'property age', 'years old', 'age_years', 'old', 'built', 'constructed'],
        'last_service_years_ago': ['last service', 'last serviced', 'last maintenance', 'last_service_years_ago', 'time since last service', 'years since', 'maintenance'],
        'seasonality': ['seasonality', 'season', 'current season', 'winter', 'spring', 'summer', 'autumn', 'fall']
    }
    
    _model = None
    _address_map = None
    _model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../predictive_maintenance_ai/models/maintenance_rf_model.pkl'))
    _address_map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/address_map.json'))
    
    def __init__(self):
        """Initialize maintenance prediction handler."""
        self.system_prompt = (
            "You are LandlordBuddy, a friendly and knowledgeable AI companion for landlords! "
            "Think of me as your trusted maintenance advisor who's here to help predict potential property issues before they become expensive problems. "
            "When it comes to maintenance prediction, I'm like your crystal ball - I gather the essential clues (property location, age, last service date, and current season), "
            "chat with you about what we've found, and then run our predictive analysis to spot potential maintenance needs. "
            "Here's the thing - I never jump to my own conclusions or give you my personal take on things. That's not my style! "
            "If our prediction can't run because we're missing some key pieces of the puzzle, I'll simply let you know: "
            "'Looks like we're missing some crucial details to run a proper maintenance prediction. Mind filling in the gaps?' "
            "When our analysis is complete, I'll present the findings in a clean, easy-to-read format and help clarify what it all means - "
            "but I stick to explaining the results, not adding my own spin on things. "
            "I keep things natural and conversational - no robotic 'processing' talk or mentions of scripts and models. "
            "I'm your professional yet approachable partner in this! "
            "I love being efficient and interactive, so I won't keep you waiting with unnecessary delays. "
            "I always check with you before diving into the prediction - your confirmation is my green light! "
            "I focus on what really matters - those four key areas I mentioned - and keep things streamlined. "
            "When we start a new prediction, I'll lay out everything we need in one go, nice and organized. "
            "No back-and-forth hunting for details one piece at a time - that's just inefficient! "
            "If you can't provide all the details, don't worry! I can still give an estimate with at least 2 key details using smart defaults for the missing information. "
            "The more information you provide, the more accurate the prediction will be. "
        )
        self.chat = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

    @classmethod
    def get_model(cls):
        if cls._model is None:
            import joblib
            cls._model = joblib.load(cls._model_path)
        return cls._model

    @classmethod
    def get_address_map(cls):
        if cls._address_map is None:
            import json
            with open(cls._address_map_path, 'r', encoding='utf-8') as f:
                cls._address_map = json.load(f)
        return cls._address_map

    def get_smart_defaults(self, provided_fields):
        """Generate contextual defaults based on provided fields and current season"""
        defaults = {}
        
        # Get current season from provided fields or default to current season
        import datetime
        month = datetime.datetime.now().month
        if month in [12, 1, 2]:
            current_season = "Winter"
        elif month in [3, 4, 5]:
            current_season = "Spring"
        elif month in [6, 7, 8]:
            current_season = "Summer"
        else:
            current_season = "Autumn"
        
        # Default seasonality
        if not provided_fields.get('seasonality'):
            defaults['seasonality'] = current_season
        
        # Default property age (typical UK property age)
        if not provided_fields.get('age_years'):
            defaults['age_years'] = 25  # Average age for UK properties
        
        # Default last service based on property age
        age = provided_fields.get('age_years', defaults.get('age_years', 25))
        if not provided_fields.get('last_service_years_ago'):
            if age < 5:
                defaults['last_service_years_ago'] = 1  # New properties, recent service
            elif age < 15:
                defaults['last_service_years_ago'] = 2  # Newer properties
            else:
                defaults['last_service_years_ago'] = 3  # Older properties need more frequent service
        
        # Default address (use common London area as fallback)
        if not provided_fields.get('address'):
            defaults['address'] = 'London Area'
        
        return defaults

    def create_estimate_with_defaults(self, provided_fields):
        """Create a maintenance prediction using smart defaults and return a conversational response"""
        defaults = self.get_smart_defaults(provided_fields)
        
        # Merge provided fields with defaults
        complete_fields = {}
        for field in self.required_fields:
            if field in provided_fields and provided_fields[field] not in (None, '', 0):
                complete_fields[field] = provided_fields[field]
            else:
                complete_fields[field] = defaults[field]
        
        # Generate assumptions text
        assumptions = []
        for field, default_value in defaults.items():
            if field == 'age_years':
                assumptions.append(f"estimated the property to be around {default_value} years old")
            elif field == 'last_service_years_ago':
                assumptions.append(f"assumed it's been about {default_value} years since last maintenance")
            elif field == 'seasonality':
                assumptions.append(f"considering it's currently {default_value}")
            elif field == 'address':
                assumptions.append(f"using typical London area characteristics")
        
        # Get the maintenance prediction
        try:
            result = self.run_model(complete_fields)
            
            # Extract risk score for conversational response
            risk_match = re.search(r'Risk Score:\*\* (\d+\.?\d*)', result)
            action_match = re.search(r'Recommended Action:\*\* (\w+)', result)
            
            risk_score = float(risk_match.group(1)) if risk_match else 0
            action = action_match.group(1) if action_match else "Monitor"
            
            # Create conversational response
            property_desc = self.create_property_description(provided_fields, complete_fields)
            
            response = f"Based on what you've told me, {property_desc}"
            
            if assumptions:
                if len(assumptions) == 1:
                    response += f" (I've {assumptions[0]})"
                else:
                    response += f" (I've {', '.join(assumptions[:-1])} and {assumptions[-1]})"
            
            # Add emoji based on risk level
            if risk_score > 7:
                emoji = "🚨"
                urgency = "needs immediate attention"
            elif risk_score > 4:
                emoji = "⚠️" 
                urgency = "should be monitored closely"
            else:
                emoji = "✅"
                urgency = "is in good shape"
            
            response += f"\n\n{emoji} **Your property {urgency}!**\n\n"
            response += result
            response += "\n\nIf you'd like a more precise prediction, just let me know the missing details! "
            
            # Ask for missing information
            missing_original = [f for f in self.required_fields if f not in provided_fields or provided_fields[f] in (None, '', 0)]
            if missing_original:
                field_names = {
                    "age_years": "**property age**",
                    "last_service_years_ago": "**last maintenance date**",
                    "seasonality": "**current season**",
                    "address": "**exact location**"
                }
                missing_names = [field_names.get(f, f.replace('_', ' ')) for f in missing_original]
                if len(missing_names) == 1:
                    response += f"I'd particularly love to know the {missing_names[0]} for a more accurate prediction."
                else:
                    response += f"I'd particularly love to know the {', '.join(missing_names[:-1])} and {missing_names[-1]} for a more accurate prediction."
            
            return response, complete_fields
            
        except Exception as e:
            return f"I ran into a snag while predicting maintenance needs: {e}. Let me know if you'd like to try again!", complete_fields

    def create_property_description(self, provided_fields, complete_fields):
        """Create a natural description of the property"""
        age = provided_fields.get('age_years') or complete_fields.get('age_years', 0)
        address = provided_fields.get('address') or complete_fields.get('address', 'your property')
        
        try:
            age = int(age)
        except:
            age = 0
            
        if age < 5:
            age_desc = "a relatively new property"
        elif age < 15:
            age_desc = "a modern property"
        elif age < 30:
            age_desc = "a well-established property"
        else:
            age_desc = "a mature property"
        
        if address and address != 'London Area' and address != 'your property':
            return f"{age_desc} at {address}"
        else:
            return age_desc

    def count_provided_fields(self, fields):
        """Count how many meaningful fields are provided"""
        meaningful_fields = 0
        for field in self.required_fields:
            value = fields.get(field)
            if value and value not in (None, '', 0):
                meaningful_fields += 1
        return meaningful_fields

    def encode_fields_for_model(self, fields):
        # Map user-friendly address to coded address using address_map
        address_map = self.get_address_map()
        addr = str(fields.get('address', '')).strip()
        # Try direct, upper, or fallback to first value
        coded_addr = address_map.get(addr, address_map.get(addr.upper(), next(iter(address_map.values()))))
        encoded = dict(fields)
        encoded['address'] = coded_addr
        # Ensure correct types
        for k in ['age_years', 'last_service_years_ago']:
            if k in encoded:
                try:
                    encoded[k] = int(float(encoded[k]))
                except Exception:
                    encoded[k] = 0
        if 'seasonality' in encoded:
            encoded['seasonality'] = str(encoded['seasonality'])
        return encoded

    def extract_fields(self, user_message, conversation_history, last_candidate_fields=None):
        # Use LangChain's PydanticOutputParser for robust extraction
        from langchain.output_parsers import PydanticOutputParser
        from pydantic import BaseModel, Field, ValidationError
        from langchain_core.prompts import ChatPromptTemplate
        import re
        
        class MaintenanceFields(BaseModel):
            address: str = Field("", description="The property address or location")
            age_years: int = Field(0, description="Property age in years")
            last_service_years_ago: int = Field(0, description="Years since last maintenance service")
            seasonality: str = Field("", description="Current season (Winter, Spring, Summer, Autumn)")

        parser = PydanticOutputParser(pydantic_object=MaintenanceFields)
        
        # Filter conversation to only maintenance prediction related messages (last 6 messages max)
        maintenance_messages = []
        recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        
        for msg in recent_messages:
            content_lower = msg["content"].lower()
            if any(keyword in content_lower for keyword in ["maintenance", "property", "address", "age", "service", "season", "winter", "spring", "summer", "autumn", "years", "old"]):
                maintenance_messages.append(msg)
        
        # Add current message
        filtered_text = "\n".join([f"{m['role']}: {m['content']}" for m in maintenance_messages])
        filtered_text += f"\nuser: {user_message}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant for maintenance prediction ONLY. Extract ONLY maintenance prediction fields from the conversation: address, age_years, last_service_years_ago, seasonality. DO NOT extract any other fields like credit score, income, rent, bedrooms, etc. If a field is missing, use an empty string or 0. Output only the JSON object as specified by the schema: {format_instructions}"),
            ("user", "Maintenance prediction conversation:\n{conversation}\nCurrent message:\n{user_message}")
        ])
        format_instructions = parser.get_format_instructions()
        prompt_value = prompt.format_prompt(
            conversation=filtered_text,
            user_message=user_message,
            format_instructions=format_instructions
        )
        response = invoke_with_tracking(
            self.chat,
            [HumanMessage(content=prompt_value.to_string())],
            "Maintenance Prediction - Field Extraction"
        )
        content = response.content.strip()
        try:
            parsed = parser.parse(content)
            fields = parsed.dict()
        except (ValidationError, Exception) as e:
            print(f"[DEBUG] Pydantic parsing failed for maintenance fields: {e}. Using fallback extraction.")
            # Fallback: try regex extraction as before
            fields = dict(last_candidate_fields) if last_candidate_fields else {}
            all_text = filtered_text
            
            # Enhanced regex patterns for maintenance-specific extraction
            
            # Address: look for location indicators
            if not fields.get('address'):
                addr_patterns = [
                    r"(?:property (?:at|on|in)|located (?:at|on|in)|address (?:is|at)?)\s*([A-Za-z0-9,\-\s]+?)(?:,| and| that| which| \d+|\.|$)",
                    r"at ([A-Za-z0-9,\-\s]+?)(?:,| and| that| which| \d+|\.|$)"
                ]
                for pattern in addr_patterns:
                    addr_match = re.search(pattern, all_text, re.IGNORECASE)
                    if addr_match:
                        fields['address'] = addr_match.group(1).strip()
                        break

            # Age: look for age indicators
            if not fields.get('age_years'):
                age_patterns = [
                    r"(?:property|building|house|flat|apartment).*?(?:is|was|age|aged|built|constructed).*?(\d{1,3})\s*years?\s*old",
                    r"(\d{1,3})\s*years?\s*old",
                    r"built.*?(\d{1,3})\s*years?\s*ago",
                    r"constructed.*?(\d{1,3})\s*years?\s*ago",
                    r"age.*?(\d{1,3})",
                    r"it.s\s*(\d{1,3})\s*years"
                ]
                for pattern in age_patterns:
                    age_match = re.search(pattern, all_text, re.IGNORECASE)
                    if age_match:
                        try:
                            fields['age_years'] = int(age_match.group(1))
                            break
                        except Exception:
                            pass

            # Last service: look for maintenance indicators
            if not fields.get('last_service_years_ago'):
                service_patterns = [
                    r"last\s*(?:service|serviced|maintenance|maintained).*?(\d{1,2})\s*years?\s*ago",
                    r"(?:service|maintenance).*?(\d{1,2})\s*years?\s*ago",
                    r"been\s*(\d{1,2})\s*years?\s*since.*?(?:service|maintenance)",
                    r"(\d{1,2})\s*years?\s*since.*?(?:service|maintenance)"
                ]
                for pattern in service_patterns:
                    svc_match = re.search(pattern, all_text, re.IGNORECASE)
                    if svc_match:
                        try:
                            fields['last_service_years_ago'] = int(svc_match.group(1))
                            break
                        except Exception:
                            pass

            # Seasonality: look for season indicators
            if not fields.get('seasonality'):
                season_patterns = [
                    r"(?:it.s|this|current|now)\s*(winter|spring|summer|autumn|fall)",
                    r"(winter|spring|summer|autumn|fall)\s*(?:now|season|time)",
                    r"in\s*(winter|spring|summer|autumn|fall)"
                ]
                for pattern in season_patterns:
                    season_match = re.search(pattern, all_text, re.IGNORECASE)
                    if season_match:
                        season = season_match.group(1).capitalize()
                        if season == "Fall":
                            season = "Autumn"
                        fields['seasonality'] = season
                        break

        # Only keep required fields and ensure correct types
        clean_fields = {}
        for k in self.required_fields:
            v = fields.get(k, 0 if k in ["age_years", "last_service_years_ago"] else "")
            # Ensure correct types
            if k in ["age_years", "last_service_years_ago"]:
                try:
                    v = int(float(v)) if v not in (None, '', 0) else 0
                except Exception:
                    v = 0
            else:
                v = str(v) if v is not None else ""
            clean_fields[k] = v
        return clean_fields

    def summarize_fields(self, fields, user_provided=None, show_result=False):
        """
        Summarize fields for display with more conversational language
        user_provided: list of fields actually provided by user (not assumed)
        show_result: if True, don't add confirmation text (we're showing final result)
        """
        if user_provided is None:
            user_provided = []
        assumed_fields = [k for k in self.required_fields if k not in user_provided]
        
        if show_result:
            summary = "**Here's what we're working with for the maintenance prediction:**\n\n"
        elif assumed_fields:
            summary = "**Here's the picture so far (I've filled in some gaps with typical values):**\n\n"
        else:
            summary = "**Perfect! Here's everything we've got:**\n\n"
        
        # Format each field with more conversational language
        for k in self.required_fields:
            v = fields.get(k, None)
            assumed = k in assumed_fields
            if k == "address":
                if v and v not in (None, '', 'London Area'):
                    if assumed and not show_result:
                        summary += f"- **Property Location:** {v} _(using typical area characteristics)_\n"
                    else:
                        summary += f"- **Property Location:** {v}\n"
                else:
                    summary += f"- **Property Location:** _(we'll need this one)_\n"
            elif k == "age_years":
                if v not in (None, '', 0):
                    if assumed and not show_result:
                        summary += f"- **Property Age:** {v} years old _(estimated typical age)_\n"
                    else:
                        summary += f"- **Property Age:** {v} years old\n"
                else:
                    summary += f"- **Property Age:** _(still need this detail)_\n"
            elif k == "last_service_years_ago":
                if v not in (None, '', 0):
                    if assumed and not show_result:
                        summary += f"- **Last Maintenance:** {v} years ago _(my best guess based on property age)_\n"
                    else:
                        summary += f"- **Last Maintenance:** {v} years ago\n"
                else:
                    summary += f"- **Last Maintenance:** _(missing this piece)_\n"
            elif k == "seasonality":
                if v not in (None, '', False, 'unknown', ''):
                    if assumed and not show_result:
                        summary += f"- **Current Season:** {v} _(assuming current season)_\n"
                    else:
                        summary += f"- **Current Season:** {v}\n"
                else:
                    summary += f"- **Current Season:** _(would love to know this)_\n"
        
        # More conversational confirmation text
        if not show_result:
            missing_to_show = [k for k in self.required_fields if k not in user_provided]
            if missing_to_show:
                summary += "\n💡 _Want a spot-on prediction? Just share these missing bits with me:_ "
                field_names = {
                    "address": "**exact location**",
                    "age_years": "**property age**", 
                    "last_service_years_ago": "**last maintenance date**",
                    "seasonality": "**current season**"
                }
                summary += ", ".join([f"_{field_names.get(k, k.replace('_', ' ').title())}_" for k in missing_to_show])
                summary += "\n\n_Or we can roll with what we have - just give me the thumbs up!_ 👍"
            else:
                summary += "\n\nLooks good to me! Ready to dive into the maintenance prediction? Just say the word! 🚀"
        
        return summary

    def needs_confirmation(self, user_message):
        confirmation_phrases = ["yes", "correct", "that's right", "yep", "confirmed", "go ahead", "proceed"]
        return user_message.strip().lower() in confirmation_phrases

    def should_run_model(self, conversation_history, candidate_fields):
        if not candidate_fields or not all(f in candidate_fields and candidate_fields[f] not in (None, '', 0, 0.0) for f in self.required_fields):
            return False
        if not conversation_history:
            return False
        last_assistant = next((m for m in reversed(conversation_history) if m["role"] == "assistant"), None)
        if not last_assistant:
            return False
        confirmation_keywords = [
            "please confirm", "is this information correct", "is this correct", "can you confirm", "are these details correct"
        ]
        return any(kw in last_assistant["content"].lower() for kw in confirmation_keywords)

    def run_model(self, fields):
        model = self.get_model()
        encoded_fields = self.encode_fields_for_model(fields)
        input_df = pd.DataFrame([{
            'address': encoded_fields['address'],
            'age_years': encoded_fields['age_years'],
            'last_service_years_ago': encoded_fields['last_service_years_ago'],
            'seasonality': encoded_fields['seasonality']
        }])
        risk_score = model.predict(input_df)[0]
        
        # More engaging and conversational result summaries
        property_desc = self.create_property_description(fields, encoded_fields)
        summary = ""
        if risk_score > 7:
            summary = f"🚨 **Urgent attention needed!** {property_desc} is showing some serious red flags:"
            action = 'Immediate Action'
            recommendations = [
                "Schedule emergency maintenance inspection within 1-2 days",
                "Check HVAC, plumbing, and electrical systems immediately", 
                "Consider temporary tenant relocation if safety concerns arise",
                "Contact professional maintenance contractors urgently"
            ]
        elif risk_score > 4:
            summary = f"⚠️ **Keep a close eye on this one...** {property_desc} has some maintenance concerns brewing:"
            action = 'Monitor'
            recommendations = [
                "Schedule comprehensive maintenance inspection within 2-4 weeks",
                "Perform preventive maintenance on aging systems",
                "Monitor tenant reports of any issues closely",
                "Plan budget for potential repairs in the next 3-6 months"
            ]
        else:
            summary = f"✅ **Looking good!** {property_desc} seems to be in solid shape:"
            action = 'Routine'
            recommendations = [
                "Continue with regular maintenance schedule",
                "Perform annual safety checks and inspections", 
                "Keep maintenance reserves for unexpected issues",
                "Monitor seasonal maintenance needs (heating/cooling systems)"
            ]
        
        md = f"{summary}\n\n**🔧 Maintenance Prediction Results:**\n\n"
        md += f"**📊 Risk Score:** {risk_score:.1f}/10\n\n"
        md += f"**🎯 Action Needed:** {action}\n\n"
        md += f"**📋 Here's what you should do:**\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            md += f"{i}. {rec}\n"
        
        md += f"\n**🔍 The breakdown:**\n"
        md += f"• Property age: {fields.get('age_years', 0)} years\n"
        md += f"• Last maintenance: {fields.get('last_service_years_ago', 0)} years ago\n"
        md += f"• Current season: {fields.get('seasonality', 'Unknown')}\n"
        md += f"• Location factors: {fields.get('address', 'Generic area')}\n\n"
        
        explanation = (
            "**How this was calculated:**\n"
            "The risk score considers property age, time since last service, seasonal factors, and location-specific maintenance patterns. "
            "Higher scores indicate more urgent maintenance needs based on predictive analysis of similar properties.\n"
        )
        
        return md + explanation

    def handle(self, conversation_history, user_message, last_candidate_fields=None):
        print(f"[DEBUG] MaintenancePredictionHandler.handle called with: {user_message}")
        print(f"[DEBUG] Last candidate fields: {last_candidate_fields}")
        
        # Extract fields from the current message
        new_fields = self.extract_fields(user_message, conversation_history, last_candidate_fields)
        print(f"[DEBUG] Extracted new fields: {new_fields}")
        
        # Merge with last candidate fields to preserve previously collected info, only keep meaningful values
        merged_fields = {k: v for k, v in (last_candidate_fields or {}).items() if k in self.required_fields and v not in (None, '', 0)}
        for k in self.required_fields:
            v = new_fields.get(k, None)
            if k in ["age_years", "last_service_years_ago"]:
                if v not in (None, '', 0):
                    merged_fields[k] = v
            else:  # address, seasonality
                if v not in (None, '', False, 'unknown', ''):
                    merged_fields[k] = v
        
        print(f"[DEBUG] Merged fields: {merged_fields}")

        # Detect if this is a DIRECT REQUEST for maintenance assessment (like ChatGPT would understand)
        direct_request_patterns = [
            # Question patterns
            "see if", "check if", "do we need", "should we", "is it time", "do i need",
            # Command patterns  
            "check this", "assess this", "analyze this", "predict", "evaluate",
            # Natural language patterns
            "need to check", "time to check", "maintenance needed", "check for maintenance",
            "flat for maintenance", "property for maintenance", "house for maintenance"
        ]
        
        action_context_patterns = [
            "maintenance", "maintice", "maintnance", "service", "inspect", "repair", "upkeep"
        ]
        
        is_direct_request = (
            any(pattern in user_message.lower() for pattern in direct_request_patterns) or
            (any(pattern in user_message.lower() for pattern in action_context_patterns) and
             any(word in user_message.lower() for word in ["this", "the", "property", "flat", "house", "building"]))
        )
        
        print(f"[DEBUG] Is direct request: {is_direct_request}")

        # Track which fields were actually provided by the user vs. inferred/assumed
        user_provided = []
        
        # Enhanced logic: if we have meaningful field values, check if they were mentioned
        for k in self.required_fields:
            v = merged_fields.get(k, None)
            # Check if this field appears to be user-provided in current or previous messages
            field_keywords = self.FIELD_SYNONYMS.get(k, [k])
            
            # Check current message for field mentions
            current_has_field = any(keyword.lower() in user_message.lower() for keyword in field_keywords)
            
            # Check recent conversation for field mentions (last 3 user messages)
            recent_has_field = False
            if conversation_history:
                recent_user_messages = [msg for msg in conversation_history[-6:] if msg["role"] == "user"][-3:]
                for msg in recent_user_messages:
                    if any(keyword.lower() in msg["content"].lower() for keyword in field_keywords):
                        recent_has_field = True
                        break
            
            # Enhanced detection for direct requests
            if k == "address" and v not in (None, '', False, 'unknown', ''):
                # Check for location mentions like "Battersea", "in", "located"
                location_phrases = ["battersea", "london", "in ", "located", "property", "flat", "house"]
                mentions_location = any(phrase in user_message.lower() for phrase in location_phrases)
                
                if mentions_location or current_has_field or recent_has_field:
                    user_provided.append(k)
                    
            elif k == "last_service_years_ago" and v not in (None, '', 0):
                # Check for maintenance timing phrases (be more inclusive)
                timing_phrases = ["last", "ago", "since", "years", "months", "serviced", "maintained", "maintenance", "maintice", "done", "was done"]
                mentions_timing = any(phrase in user_message.lower() for phrase in timing_phrases)
                
                if mentions_timing or current_has_field or recent_has_field:
                    user_provided.append(k)
                    
            elif k == "age_years" and v not in (None, '', 0):
                # Check for age-related phrases
                age_phrases = ["old", "age", "built", "years", "new", "vintage", "constructed"]
                mentions_age = any(phrase in user_message.lower() for phrase in age_phrases)
                
                if mentions_age or current_has_field or recent_has_field:
                    user_provided.append(k)
                    
            elif k == "seasonality" and v not in (None, '', False, 'unknown', ''):
                # Check for season-related phrases in current message
                season_mentions = ["winter", "spring", "summer", "autumn", "fall", "season"]
                current_mentions_season = any(phrase in user_message.lower() for phrase in season_mentions)
                
                if current_mentions_season or current_has_field or recent_has_field:
                    user_provided.append(k)
        
        print(f"[DEBUG] User provided fields: {user_provided}")

        # IMPORTANT: If this is a direct request with ANY meaningful data, provide immediate results
        if is_direct_request and len(user_provided) >= 1:
            # Use smart defaults to fill in missing information
            avg_defaults = {
                "address": "London Area",
                "age_years": 25,
                "last_service_years_ago": 3,
                "seasonality": "Winter"  # Will be auto-detected in get_smart_defaults
            }
            estimate_fields = dict(merged_fields)
            
            # Use smart defaults that consider provided context
            smart_defaults = self.get_smart_defaults(merged_fields)
            for k in self.required_fields:
                if k not in estimate_fields or estimate_fields[k] in (None, '', 0):
                    estimate_fields[k] = smart_defaults.get(k, avg_defaults[k])

            # Show immediate prediction result (like ChatGPT would do)
            result_md = self.run_model(estimate_fields)
            
            # Find missing fields for optional enhancement offer
            missing_to_show = [k for k in self.required_fields if k not in user_provided]
            
            response = result_md
            
            if missing_to_show:
                response += "\n\n---\n\n"
                response += "💭 **Want an even sharper analysis?** Drop me these details and I'll give you the full picture: "
                field_names = {
                    "address": "**exact location**",
                    "age_years": "**property age**", 
                    "last_service_years_ago": "**last maintenance date**",
                    "seasonality": "**current season**"
                }
                response += ", ".join([f"{field_names.get(k, k.replace('_', ' '))}" for k in missing_to_show])
                response += " and I'll re-run everything! 🎯"
            
            return {
                "response": response,
                "action": "show_estimate_with_result",
                "fields": estimate_fields
            }

        # Find missing fields
        missing = []
        for k in self.required_fields:
            v = merged_fields.get(k, None)
            if k in ["age_years", "last_service_years_ago"]:
                if v in (None, '', 0):
                    missing.append(k)
            else:  # address, seasonality
                if v in (None, '', False, 'unknown', ''):
                    missing.append(k)

        # If not enough info and not a direct request, ask for more with conversational flair
        if len(user_provided) < 2:
            conversation_starters = [
                "Alright, let's get this maintenance prediction party started! 🔧",
                "Time to put on my detective hat and predict maintenance needs! 🕵️‍♂️",
                "Let's dive into some maintenance prediction magic! ✨",
                "Ready to uncover potential property issues before they become expensive problems? 🔍"
            ]
            import random
            starter = random.choice(conversation_starters)
            
            prompt = (
                f"{starter}\n\n"
                "I'll need some key details about your property to work my predictive magic. "
                "Just share at least **two** of these with me and I can get started:\n\n"
            )
            
            field_descriptions = {
                "address": "**Property Location** - Where is this property located?",
                "age_years": "**Property Age** - How old is the building (in years)?",
                "last_service_years_ago": "**Last Maintenance** - When was it last serviced or maintained?",
                "seasonality": "**Current Season** - What season are we in right now?"
            }
            
            for k in self.required_fields:
                prompt += f"• {field_descriptions[k]}\n"
            
            prompt += "\nOnce you drop a couple of these details on me, I'll whip up a maintenance prediction for you! 🚀"
            
            return {
                "response": prompt,
                "action": "ask_for_info",
                "fields": merged_fields
            }

        # If we have at least 2 fields, show estimated prediction with results
        if len(user_provided) >= 2:
            # Fill in missing fields with defaults for estimation
            avg_defaults = {
                "address": "London Area",
                "age_years": 25,
                "last_service_years_ago": 3,
                "seasonality": "Winter"  # Will be auto-detected in get_smart_defaults
            }
            estimate_fields = dict(merged_fields)
            
            # Use smart defaults that consider provided context
            smart_defaults = self.get_smart_defaults(merged_fields)
            for k in missing:
                estimate_fields[k] = smart_defaults.get(k, avg_defaults[k])

            # Show the prediction result with more personality
            result_md = self.run_model(estimate_fields)
            
            # Add missing fields note at the bottom with conversational touch
            missing_to_show = [k for k in missing]
            
            response = result_md
            
            if missing_to_show:
                response += "\n\n---\n\n"
                response += "💭 **Want an even sharper analysis?** Drop me these details and I'll give you the full picture: "
                field_names = {
                    "address": "**exact location**",
                    "age_years": "**property age**", 
                    "last_service_years_ago": "**last maintenance date**",
                    "seasonality": "**current season**"
                }
                response += ", ".join([f"{field_names.get(k, k.replace('_', ' '))}" for k in missing_to_show])
                response += " and I'll re-run everything! 🎯"
            
            return {
                "response": response,
                "action": "show_estimate_with_result",
                "fields": estimate_fields
            }

        # Fallback to LLM conversation
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        response = invoke_with_tracking(
            self.chat,
            [HumanMessage(content=m["content"]) for m in messages],
            "Maintenance Prediction - Conversation"
        )
        reply = response.content.strip()

        # Remove any problematic phrases and replace with more conversational alternatives
        replacements = {
            "please wait": "hang tight",
            "processing": "working on it",
            "hold on": "give me a sec",
            "one moment": "just a moment",
            "I'll process": "I'll work on",
            "wait a moment": "bear with me",
            "it's important to exercise caution": "worth being careful here",
            "you may wish to consider": "you might want to think about",
            "you may want to explore": "might be worth looking into",
            "Based on the information provided": "From what you've told me",
            "Summary:": "Here's the deal:"
        }
        
        for old_phrase, new_phrase in replacements.items():
            reply = reply.replace(old_phrase, new_phrase)
            reply = reply.replace(old_phrase.capitalize(), new_phrase.capitalize())

        return {"response": reply, "action": "chat", "fields": merged_fields}

    def needs_confirmation(self, user_message):
        confirmation_keywords = [
            "yes", "correct", "confirm", "ok", "okay", "proceed", "continue", "that's right",
            "sounds good", "let's go", "do it", "sure", "absolutely", "yep", "yeah", 
            "thumbs up", "green light", "go ahead", "let's roll", "perfect"
        ]
        return any(kw in user_message.lower().strip() for kw in confirmation_keywords)

# --- Enhanced Intent Detection and Entity Recognition ---
def detect_intent(user_message, conversation_history=None):
    """
    Enhanced intent detection using the new conversation intelligence system.
    Falls back to basic keyword matching if enhanced system is unavailable.
    """
    # PRIORITY: Strong maintenance keyword detection FIRST (before any AI/LLM processing)
    user_message_lower = user_message.lower()
    maintenance_keywords = [
        'maintenance', 'maintice', 'maintnance', 'maintnence', 'maintanence', 'maintiance', 'maintince', 
        'maintnce', 'maintence', 'maintainance', 'repair', 'fix', 'upkeep', 'service', 'broken', 'issue'
    ]
    predict_keywords = ['predict', 'prediction', 'check', 'see if', 'need to check', 'should we check', 'do we need', 'time to check']
    property_keywords = ['flat', 'property', 'house', 'apartment', 'building', 'unit', 'place']
    
    # Check for maintenance + prediction patterns
    has_maintenance = any(keyword in user_message_lower for keyword in maintenance_keywords)
    has_predict = any(keyword in user_message_lower for keyword in predict_keywords)
    has_property = any(keyword in user_message_lower for keyword in property_keywords)
    
    # Strong maintenance patterns that should ALWAYS be maintenance_prediction
    maintenance_phrases = [
        'check this flat for maintenance', 'check for maintenance', 'maintenance needed',
        'last maintenance', 'maintenance done', 'maintenance check', 'need maintenance',
        'see if we need to check', 'predict maintenance', 'maintenance prediction'
    ]
    
    strong_maintenance_signal = any(phrase in user_message_lower for phrase in maintenance_phrases)
    
    if has_maintenance or strong_maintenance_signal or (has_predict and has_property and any(kw in user_message_lower for kw in maintenance_keywords)):
        print(f"[DEBUG] PRIORITY detect_intent: Strong maintenance detection - has_maintenance: {has_maintenance}, strong_signal: {strong_maintenance_signal}")
        return "maintenance_prediction"
    
    try:
        # Use enhanced conversation intelligence
        conversation_ai = get_conversation_intelligence()
        analysis = conversation_ai.analyze_message(user_message, conversation_history)
        
        # Map to our legacy intent names for compatibility
        intent_map = {
            IntentType.RENT_PREDICTION: "rent_prediction",
            IntentType.TENANT_SCREENING: "tenant_screening", 
            IntentType.MAINTENANCE_PREDICTION: "maintenance_prediction",
            IntentType.GREETING: "greeting",
            IntentType.SMALL_TALK: "greeting",
            IntentType.CLARIFICATION: "clarify_intent"
        }
        
        return intent_map.get(analysis.primary_intent.type, "clarify_intent")
        
    except Exception as e:
        print(f"[WARNING] Enhanced intent detection failed, using fallback: {e}")
        # Fallback to basic keyword matching with improved maintenance detection
        msg = user_message.lower()
        
        # Strong maintenance patterns first
        maintenance_keywords = [
            'maintenance', 'maintice', 'maintnance', 'maintnence', 'maintanence', 'maintiance', 'maintince', 
            'maintnce', 'maintence', 'maintainance', 'maintain', 'repair', 'fix', 'upkeep', 'service'
        ]
        maintenance_phrases = [
            'check this flat for maintenance', 'check for maintenance', 'maintenance needed',
            'last maintenance', 'maintenance done', 'maintenance check', 'need maintenance',
            'see if we need to check', 'predict maintenance', 'maintenance prediction'
        ]
        
        # Check for maintenance first (highest priority for disambiguation)
        if any(phrase in msg for phrase in maintenance_phrases) or any(word in msg for word in maintenance_keywords):
            return "maintenance_prediction"
        
        # Then check other intents
        if any(word in msg for word in ["hello", "hi", "hey", "good morning", "good afternoon", "thanks", "thank you"]):
            return "greeting"
        elif any(word in msg for word in ["rent", "price", "how much", "estimate"]) and "maintenance" not in msg:
            return "rent_prediction"
        elif any(word in msg for word in ["tenant", "screen", "applicant", "background", "credit", "income"]) and "maintenance" not in msg:
            return "tenant_screening"
        
        # If no intent is detected, return None
        return None

# --- Enhanced LLM-based Intent Detection with Entity Recognition ---
def llm_detect_intent(conversation_history, user_message):
    """
    Enhanced LLM-based intent detection with entity recognition and multi-intent support.
    """
    # PRIORITY: Strong maintenance keyword detection FIRST (before any AI/LLM processing)
    user_message_lower = user_message.lower()
    maintenance_keywords = [
        'maintenance', 'maintice', 'maintnance', 'maintnence', 'maintanence', 'maintiance', 'maintince', 
        'maintnce', 'maintence', 'maintainance', 'repair', 'fix', 'upkeep', 'service', 'broken', 'issue'
    ]
    predict_keywords = ['predict', 'prediction', 'check', 'see if', 'need to check', 'should we check', 'do we need', 'time to check']
    property_keywords = ['flat', 'property', 'house', 'apartment', 'building', 'unit', 'place']
    
    # Check for maintenance + prediction patterns
    has_maintenance = any(keyword in user_message_lower for keyword in maintenance_keywords)
    has_predict = any(keyword in user_message_lower for keyword in predict_keywords)
    has_property = any(keyword in user_message_lower for keyword in property_keywords)
    
    # Strong maintenance patterns that should ALWAYS be maintenance_prediction
    maintenance_phrases = [
        'check this flat for maintenance', 'check for maintenance', 'maintenance needed',
        'last maintenance', 'maintenance done', 'maintenance check', 'need maintenance',
        'see if we need to check', 'predict maintenance', 'maintenance prediction'
    ]
    
    strong_maintenance_signal = any(phrase in user_message_lower for phrase in maintenance_phrases)
    
    if has_maintenance or strong_maintenance_signal or (has_predict and has_property and any(kw in user_message_lower for kw in maintenance_keywords)):
        print(f"[DEBUG] PRIORITY: Strong maintenance detection - has_maintenance: {has_maintenance}, strong_signal: {strong_maintenance_signal}")
        print(f"[DEBUG] Maintenance keywords found: {[kw for kw in maintenance_keywords if kw in user_message_lower]}")
        print(f"[DEBUG] Strong phrases found: {[phrase for phrase in maintenance_phrases if phrase in user_message_lower]}")
        return "maintenance_prediction"
    
    try:
        # Use the enhanced conversation intelligence system
        conversation_ai = get_conversation_intelligence()
        analysis = conversation_ai.analyze_message(user_message, conversation_history)
        
        print(f"[DEBUG] Enhanced intent detection - Primary intent: {analysis.primary_intent.type.value}, Confidence: {analysis.confidence}")
        
        # Return comprehensive analysis instead of just intent
        return {
            'primary_intent': analysis.primary_intent.type.value,
            'confidence': analysis.confidence,
            'all_intents': [intent.type.value for intent in analysis.intents],
            'entities': [{'label': e.label, 'text': e.text, 'confidence': e.confidence} for e in analysis.entities],
            'requires_clarification': analysis.requires_clarification,
            'clarification_message': analysis.clarification_message
        }
        
    except Exception as e:
        print(f"[WARNING] Enhanced LLM intent detection failed, using fallback: {e}")
        # Fallback to basic LLM detection
        from langchain_openai import ChatOpenAI
        chat = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
        
        # First, do a quick keyword-based check for maintenance prediction
        user_message_lower = user_message.lower()
        maintenance_keywords = [
            'maintenance', 'maintice', 'maintnance', 'maintnence', 'maintanence', 'maintiance', 'maintince', 
            'maintnce', 'maintence', 'maintainance', 'repair', 'fix', 'upkeep', 'service', 'broken', 'issue'
        ]
        predict_keywords = ['predict', 'prediction', 'check', 'see if', 'need to check', 'should we check', 'do we need', 'time to check']
        property_keywords = ['flat', 'property', 'house', 'apartment', 'building', 'unit', 'place']
        
        # Check for maintenance + prediction patterns
        has_maintenance = any(keyword in user_message_lower for keyword in maintenance_keywords)
        has_predict = any(keyword in user_message_lower for keyword in predict_keywords)
        has_property = any(keyword in user_message_lower for keyword in property_keywords)
        
        # Strong maintenance patterns
        maintenance_phrases = [
            'check this flat for maintenance', 'check for maintenance', 'maintenance needed',
            'last maintenance', 'maintenance done', 'maintenance check', 'need maintenance',
            'see if we need to check', 'predict maintenance', 'maintenance prediction'
        ]
        
        strong_maintenance_signal = any(phrase in user_message_lower for phrase in maintenance_phrases)
        
        if has_maintenance or strong_maintenance_signal or (has_predict and has_property and 'maintenance' in user_message_lower):
            print(f"[DEBUG] Fallback: Strong maintenance detection - has_maintenance: {has_maintenance}, strong_signal: {strong_maintenance_signal}, predict+property+maintenance: {has_predict and has_property}")
            print(f"[DEBUG] Maintenance keywords found: {[kw for kw in maintenance_keywords if kw in user_message_lower]}")
            print(f"[DEBUG] Strong phrases found: {[phrase for phrase in maintenance_phrases if phrase in user_message_lower]}")
            return "maintenance_prediction"
        
        # Use last 4-5 messages for context
        history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        context = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        
        # More sophisticated prompt that considers conversation context
        prompt = (
            "You are an expert assistant for landlords. "
            "Given the following conversation, determine the user's current intent. "
            "Pay attention to the conversation flow - if the user is already in a tenant screening conversation and providing information like rent/income amounts, they are likely CONTINUING tenant screening, not starting rent prediction.\n"
            "Similarly, if they're in a rent prediction conversation and providing property details, they're continuing rent prediction.\n\n"
            "Available intents:\n"
            "- 'tenant_screening': User wants to screen a tenant or is providing tenant information (credit score, income, employment, eviction history)\n"
            "- 'rent_prediction': User wants to estimate rent for a property or is providing property details (address, bedrooms, bathrooms, size, property type)\n"
            "- 'maintenance_prediction': User wants to predict maintenance needs, check if maintenance is needed, or asking about repairs/upkeep. Keywords include: maintenance, repair, fix, upkeep, service, broken, issue, predict maintenance, check maintenance, need maintenance, maintice, maintnance (misspellings). IMPORTANT: phrases like 'see if we need to check this flat for maintenance' or 'check for maintenance' are ALWAYS maintenance_prediction.\n"
            "- 'greeting': User is greeting or starting conversation\n"
            "- 'other': Anything else\n\n"
            "CRITICAL RULES:\n"
            "1. ANY message containing 'maintenance' (including misspellings like 'maintice') should be classified as 'maintenance_prediction'\n"
            "2. Phrases like 'see if we need to check...for maintenance' are ALWAYS 'maintenance_prediction'\n"
            "3. 'check this flat for maintenance' is ALWAYS 'maintenance_prediction'\n"
            "4. If the conversation shows the user is already engaged in tenant screening and they provide rent/income information, classify as 'tenant_screening' (they're providing data for the screening).\n"
            "5. Only classify as 'rent_prediction' if they explicitly ask for rent estimation or start providing property details for rent estimation.\n\n"
            f"Conversation context:\n{context}\n\nCurrent user message: {user_message}\n\n"
            "Based on the context and current message, what is the user's intent? Respond with only the intent keyword."
        )
        
        response = invoke_with_tracking(
            chat,
            [HumanMessage(content=prompt)],
            "Intent Detection - Fallback"
        )
        intent = response.content.strip().lower()
        
        print(f"[DEBUG] Fallback LLM intent response: '{intent}'")
        
        # Map response to our intent system
        if "greeting" in intent or "hello" in intent or "hi" in intent:
            return "greeting"
        elif "tenant_screening" in intent or "tenant" in intent or "screen" in intent:
            return "tenant_screening"
        elif "rent_prediction" in intent or "rent" in intent:
            return "rent_prediction"  
        elif "maintenance" in intent or "repair" in intent or "predict" in intent or "maintice" in intent or "maintnance" in intent:
            return "maintenance_prediction"
        
        print(f"[DEBUG] No intent match found for: '{intent}', returning None")
        return None

# --- Explicit Intent Switch Detection ---
def user_requests_intent_switch(user_message):
    msg = user_message.lower()
    print(f"[DEBUG] user_requests_intent_switch called with: '{user_message}'")
    
    # Explicit switch phrases
    explicit_switch_phrases = [
        "forget it", "let's do", "i want to do", "switch to", "change to", "do rent instead", 
        "do tenant instead", "do maintenance instead", "not this", "wrong task", 
        "that's not what i meant", "i want rent", "i want tenant", "i want maintenance"
    ]
    
    # Action request phrases that indicate new intent
    action_request_phrases = [
        # Maintenance prediction action requests
        "see if we need", "check if", "predict if", "do we need", "should we", 
        "maintenance prediction", "predict maintenance", "check maintenance",
        "need to check", "time to check", "maintenance check",
        # Rent prediction action requests  
        "estimate rent", "predict rent", "rent prediction", "how much rent", "property value",
        "what's the rent", "rent estimate", "price estimate",
        # Tenant screening action requests
        "screen tenant", "tenant screening", "check tenant", "approve tenant",
        "screen this tenant", "evaluate tenant", "assess tenant"
    ]
    
    # Check for maintenance-specific patterns
    maintenance_patterns = [
        "maintenance", "maintice", "maintnance", "maintnence", "maintanence", "maintiance", "maintince", 
        "maintnce", "maintence", "maintainance", "maintain", "repair", "fix", "upkeep", "service"
    ]
    action_words = ["see", "check", "predict", "do", "need", "should", "time", "assess", "evaluate"]
    property_words = ["flat", "property", "house", "apartment", "building", "unit", "place"]
    
    # Strong maintenance request patterns
    maintenance_request_patterns = [
        "check this flat for maintenance", "check for maintenance", "maintenance needed",
        "last maintenance", "maintenance done", "maintenance check", "need maintenance",
        "see if we need to check", "predict maintenance", "maintenance prediction",
        "do we need maintenance", "should we check", "time for maintenance"
    ]
    
    # If message contains strong maintenance request patterns, it's definitely a new maintenance request
    if any(pattern in msg for pattern in maintenance_request_patterns):
        return True
    
    # If message contains maintenance words + action words + property words, it's likely a new maintenance request
    has_maintenance = any(pattern in msg for pattern in maintenance_patterns)
    has_action = any(action in msg for action in action_words)
    has_property = any(prop in msg for prop in property_words)
    
    if has_maintenance and has_action:
        return True
    
    # Special case: "see if we need to check" + property context is almost always maintenance
    if ("see if" in msg and "check" in msg and has_property) or ("do we need" in msg and has_property):
        print(f"[DEBUG] Special maintenance case detected: see if + check + property OR do we need + property")
        return True
    
    # Check explicit and action request phrases
    explicit_match = any(phrase in msg for phrase in explicit_switch_phrases)
    action_match = any(phrase in msg for phrase in action_request_phrases)
    
    print(f"[DEBUG] Explicit match: {explicit_match}, Action match: {action_match}")
    print(f"[DEBUG] Has maintenance: {has_maintenance}, Has action: {has_action}, Has property: {has_property}")
    
    result = explicit_match or action_match
    print(f"[DEBUG] user_requests_intent_switch returning: {result}")
    return result

def is_providing_information(user_message, current_intent):
    """
    Check if the user is providing information for the current intent rather than switching
    """
    if not current_intent:
        return False
        
    msg = user_message.lower()
    
    # First, check for clear intent switching patterns that override any information detection
    intent_switch_patterns = [
        # Maintenance prediction requests
        "maintenance", "maintice", "maintnance", "repair", "fix", "upkeep", "service", "broken", "issue",
        "predict maintenance", "check maintenance", "need maintenance", "maintenance prediction",
        # Rent prediction requests
        "estimate rent", "predict rent", "rent prediction", "how much rent", "property value",
        # Tenant screening requests (when coming from other intents)
        "screen tenant", "tenant screening", "check tenant", "approve tenant"
    ]
    
    # If message contains intent switching patterns, it's NOT providing info for current intent
    if any(pattern in msg for pattern in intent_switch_patterns):
        print(f"[DEBUG] Intent switch detected: '{user_message}' contains switching patterns")
        return False
    
    # More specific patterns based on intent
    if current_intent == "tenant_screening":
        # Tenant screening specific information patterns
        tenant_info_patterns = [
            # Credit/Financial info
            "credit score", "score is", "score of", "credit is",
            "income is", "income of", "monthly income", "earns", "makes",
            # Employment info  
            "employed", "unemployed", "job", "work", "self-employed",
            # Rent info (but only when not asking for rent estimation)
            "rent is", "rent of", "paying", "monthly rent",
            # Eviction info
            "eviction", "evicted", "no eviction", "never evicted"
        ]
        
        # Check for tenant screening information patterns
        return any(pattern in msg for pattern in tenant_info_patterns)
                
    elif current_intent == "rent_prediction":
        # Rent prediction specific information patterns
        rent_info_patterns = [
            "bedroom", "bathroom", "size", "sq ft", "square feet",
            "address is", "located", "postcode", "flat", "house", 
            "apartment", "property type", "it's a"
        ]
        
        if any(pattern in msg for pattern in rent_info_patterns):
            new_request_patterns = [
                "can you", "could you", "i want", "i need", "help me",
                "screen tenant", "tenant screening", "check tenant",
                "maintenance", "repair"
            ]
            if not any(pattern in msg for pattern in new_request_patterns):
                return True
    
    return False

# --- Enhanced Conversational Engine with Milvus and Advanced Intelligence ---
def enhanced_conversational_engine(conversation_history, user_message, last_candidate_fields=None, 
                                 last_intent=None, intent_completed=False, session_id=None, user_id=None):
    """
    Enhanced modular conversational engine for LandlordBuddy.
    Uses Milvus for memory and advanced NER/intent detection.
    """
    try:
        # Initialize services
        milvus_store = get_milvus_store()
        conversation_ai = get_conversation_intelligence()
        
        # Store user message in Milvus
        if session_id and user_id:
            milvus_store.store_chat_message(
                session_id=session_id,
                user_id=user_id,
                message_type="user",
                content=user_message,
                intent="",  # Will be filled after detection
                entities={}
            )
        
        # Get relevant conversation memory
        relevant_memory = []
        if session_id:
            relevant_memory = milvus_store.retrieve_chat_memory(
                session_id=session_id,
                query_text=user_message,
                limit=5
            )
        
        # Enhance conversation history with memory
        enhanced_history = conversation_history.copy() if conversation_history else []
        for memory in relevant_memory:
            if memory not in enhanced_history:  # Avoid duplicates
                enhanced_history.insert(0, {
                    "role": memory["message_type"],
                    "content": memory["content"]
                })
        
        # Analyze the message with advanced AI
        analysis = conversation_ai.analyze_message(
            user_message=user_message,
            conversation_history=enhanced_history,
            current_fields=last_candidate_fields,
            session_id=session_id
        )
        
        primary_intent = analysis.primary_intent.type
        extracted_entities = {entity.label: entity.normalized_value or entity.text 
                            for entity in analysis.entities}
        
        # Handle greeting intent
        if primary_intent == IntentType.GREETING:
            response = conversation_ai.handle_greeting()
            result = {
                "response": response,
                "action": "greeting",
                "fields": last_candidate_fields or {},
                "last_intent": None,
                "intent_completed": True
            }
        
        # Handle small talk intent
        elif primary_intent == IntentType.SMALL_TALK:
            response = conversation_ai.handle_small_talk(user_message)
            result = {
                "response": response,
                "action": "small_talk",
                "fields": last_candidate_fields or {},
                "last_intent": None,
                "intent_completed": True
            }
        
        # Handle clarification needed
        elif analysis.requires_clarification:
            result = {
                "response": analysis.clarification_message,
                "action": "clarify_intent",
                "fields": last_candidate_fields or {},
                "last_intent": None,
                "intent_completed": True
            }
        
        # Handle multiple intents
        elif len(analysis.intents) > 1 and all(i.confidence > 0.7 for i in analysis.intents[:2]):
            # Multi-intent handling - ask user to choose
            intent_names = [intent.type.value.replace('_', ' ').title() for intent in analysis.intents[:2]]
            response = (f"I can see you're interested in multiple things: {' and '.join(intent_names)}. "
                       f"Which would you like to start with first?")
            result = {
                "response": response,
                "action": "clarify_intent",
                "fields": last_candidate_fields or {},
                "last_intent": None,
                "intent_completed": True
            }
        
        # Handle specific intents
        elif primary_intent == IntentType.RENT_PREDICTION:
            handler = RentPredictionHandler()
            # Merge AI-extracted entities with existing fields
            merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
            merged_fields.update(extracted_entities)
            result = handler.handle(enhanced_history, user_message, merged_fields)
            result["last_intent"] = "rent_prediction" if not result.get("action") == "rent_prediction" else None
            result["intent_completed"] = result.get("action") == "rent_prediction"
        
        elif primary_intent == IntentType.TENANT_SCREENING:
            handler = TenantScreeningHandler()
            merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
            merged_fields.update(extracted_entities)
            result = handler.handle(enhanced_history, user_message, merged_fields)
            result["last_intent"] = "tenant_screening" if not result.get("action") == "screen_tenant" else None
            result["intent_completed"] = result.get("action") == "screen_tenant"
        
        elif primary_intent == IntentType.MAINTENANCE_PREDICTION:
            handler = MaintenancePredictionHandler()
            merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
            merged_fields.update(extracted_entities)
            result = handler.handle(enhanced_history, user_message, merged_fields)
            result["last_intent"] = "maintenance_prediction" if not result.get("action") == "maintenance_prediction" else None
            result["intent_completed"] = result.get("action") == "maintenance_prediction"
        
        # Handle continuation of previous intent
        elif last_intent and not intent_completed:
            if last_intent == "rent_prediction":
                handler = RentPredictionHandler()
                merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
                merged_fields.update(extracted_entities)
                result = handler.handle(enhanced_history, user_message, merged_fields)
                result["last_intent"] = last_intent if not result.get("action") == "rent_prediction" else None
                result["intent_completed"] = result.get("action") == "rent_prediction"
            elif last_intent == "tenant_screening":
                handler = TenantScreeningHandler()
                merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
                merged_fields.update(extracted_entities)
                result = handler.handle(enhanced_history, user_message, merged_fields)
                result["last_intent"] = last_intent if not result.get("action") == "screen_tenant" else None
                result["intent_completed"] = result.get("action") == "screen_tenant"
            elif last_intent == "maintenance_prediction":
                handler = MaintenancePredictionHandler()
                merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
                merged_fields.update(extracted_entities)
                result = handler.handle(enhanced_history, user_message, merged_fields)
                result["last_intent"] = last_intent if not result.get("action") == "maintenance_prediction" else None
                result["intent_completed"] = result.get("action") == "maintenance_prediction"
            else:
                result = {
                    "response": "I'm not sure how to continue. What would you like me to help you with?",
                    "action": "clarify_intent",
                    "fields": {},
                    "last_intent": None,
                    "intent_completed": True
                }
        
        # Unknown or unsupported intent
        else:
            result = {
                "response": (
                    "I'm sorry, I couldn't clearly understand your request. "
                    "You may be using different wording or asking for something else.\n\n"
                    "Here are the tasks I can help you with:\n"
                    "- **Rent Prediction**: Estimate the rent for your property.\n"
                    "- **Tenant Screening**: Assess a tenant's suitability.\n"
                    "- **Maintenance Prediction**: Predict potential maintenance needs.\n\n"
                    "Please specify which of these you'd like help with, and provide any relevant details."
                ),
                "action": "clarify_intent",
                "fields": {},
                "last_intent": None,
                "intent_completed": True
            }
        
        # Store assistant response in Milvus
        if session_id and user_id:
            milvus_store.store_chat_message(
                session_id=session_id,
                user_id=user_id,
                message_type="assistant",
                content=result["response"],
                intent=primary_intent.value if primary_intent else "",
                entities=extracted_entities
            )
        
        return result
        
    except Exception as e:
        # Fallback to original engine if enhanced version fails
        print(f"[WARNING] Enhanced engine failed, falling back to original: {e}")
        return conversational_engine(conversation_history, user_message, last_candidate_fields, 
                                   last_intent, intent_completed)

# --- Original Conversational Engine (Fallback) ---
def conversational_engine(conversation_history, user_message, last_candidate_fields=None, last_intent=None, intent_completed=False):
    """
    Modular conversational engine for LandlordBuddy.
    Routes to the correct module handler based on detected intent.
    """
    # Add call counter for cost tracking
    if not hasattr(conversational_engine, 'call_count'):
        conversational_engine.call_count = 0
    conversational_engine.call_count += 1
    
    # Print cost summary every 10 calls
    if conversational_engine.call_count % 10 == 0:
        print(f"\n[COST SUMMARY] After {conversational_engine.call_count} calls:")
        summary = get_session_summary()
        print(f"Total API calls: {summary['total_calls']}")
        print(f"Total cost: ${summary['total_cost']:.6f}")
        print(f"Average cost per call: ${summary['avg_cost_per_call']:.6f}")
    
    # If user requests to switch/cancel, re-detect intent and reset fields
    intent_switch_detected = user_requests_intent_switch(user_message)
    print(f"[DEBUG] Intent switch detected: {intent_switch_detected}")
    
    if intent_switch_detected:
        detected_intent = llm_detect_intent(conversation_history, user_message)
        print(f"[DEBUG] After intent switch, detected intent: {detected_intent}")
        if isinstance(detected_intent, dict):
            intent = detected_intent.get('primary_intent')
        else:
            intent = detected_intent
        intent_completed = False
        # Clear fields when switching intents
        last_candidate_fields = {}
    elif last_intent and not intent_completed:
        # Check if user is providing information for current intent
        if is_providing_information(user_message, last_intent):
            # Continue with the current intent
            intent = last_intent
        else:
            # User might be switching topics, re-detect intent
            detected_intent = llm_detect_intent(conversation_history, user_message)
            if isinstance(detected_intent, dict):
                new_intent = detected_intent.get('primary_intent')
            else:
                new_intent = detected_intent
            
            # If it's the same intent, continue; if different, switch
            if new_intent == last_intent:
                intent = last_intent
            else:
                intent = new_intent
                intent_completed = False
                last_candidate_fields = {}
    else:
        # No active intent or just completed, detect new intent
        detected_intent = llm_detect_intent(conversation_history, user_message)
        
        # Handle case where llm_detect_intent returns a dict (enhanced) or string (fallback)
        if isinstance(detected_intent, dict):
            intent = detected_intent.get('primary_intent')
        else:
            intent = detected_intent
            
        intent_completed = False
        # Only clear fields when explicitly starting a completely new intent (not when continuing)
        # If the detected intent is the same as last_intent, don't clear fields
        if intent != last_intent:
            last_candidate_fields = {}

    confirmation_phrases = ["yes", "correct", "that's right", "yep", "confirmed", "go ahead", "proceed"]
    is_confirmation = user_message.strip().lower() in confirmation_phrases

    # Handle each intent
    if intent == "greeting":
        # Handle greeting with friendly response
        return {
            "response": (
                "Hello! I'm LandlordBuddy, your AI assistant for property management. "
                "I can help you with rent pricing, tenant screening, and maintenance predictions.\n\n"
                "What would you like to do today?"
            ),
            "action": "greeting",
            "fields": {},
            "last_intent": None,
            "intent_completed": True
        }
    elif intent == "rent_prediction":
        handler = RentPredictionHandler()
        # Filter fields to only rent prediction fields
        rent_fields = {}
        if last_candidate_fields:
            for field in handler.required_fields:
                if field in last_candidate_fields:
                    rent_fields[field] = last_candidate_fields[field]
        result = handler.handle(conversation_history, user_message, rent_fields)
        # If model was run, mark intent as completed
        if result.get("action") == "screen_tenant" or result.get("action") == "rent_prediction":
            intent_completed = True
        return {**result, "last_intent": intent if not intent_completed else None, "intent_completed": intent_completed}
    elif intent == "tenant_screening":
        handler = TenantScreeningHandler()
        # Filter fields to only tenant screening fields
        tenant_fields = {}
        if last_candidate_fields:
            for field in handler.required_fields:
                if field in last_candidate_fields:
                    tenant_fields[field] = last_candidate_fields[field]
        result = handler.handle(conversation_history, user_message, tenant_fields)
        if result.get("action") == "screen_tenant":
            intent_completed = True
        return {**result, "last_intent": intent if not intent_completed else None, "intent_completed": intent_completed}
    elif intent == "maintenance_prediction":
        handler = MaintenancePredictionHandler()
        # Filter fields to only maintenance prediction fields
        maintenance_fields = {}
        if last_candidate_fields:
            for field in handler.required_fields:
                if field in last_candidate_fields:
                    maintenance_fields[field] = last_candidate_fields[field]
        result = handler.handle(conversation_history, user_message, maintenance_fields)
        if result.get("action") == "maintenance_prediction" or result.get("action") == "maintenance_alerts":
            intent_completed = True
        return {**result, "last_intent": intent if not intent_completed else None, "intent_completed": intent_completed}
    else:
        return {
            "response": (
                "I'm sorry, I couldn't clearly understand your request. "
                "You may be using different wording or asking for something else.\n\n"
                "Here are the tasks I can help you with:\n"
                "- **Rent Prediction**: Estimate the rent for your property.\n"
                "- **Tenant Screening**: Assess a tenant's suitability.\n"
                "- **Maintenance Prediction**: Predict potential maintenance needs.\n\n"
                "Please specify which of these you'd like help with, and provide any relevant details."
            ),
            "action": "clarify_intent",
            "fields": {},
            "last_intent": None,
            "intent_completed": True
        }

def predict_rent(fields):
    """
    Simple wrapper for legacy compatibility: predicts rent given a dict of fields.
    """
    handler = RentPredictionHandler()
    return handler.run_model(fields)

def llm_web_compare(pred, rent_range):
    """
    Use LLM to search the web for similar rental listings and compare to prediction.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=openai_api_key)
    prompt = f"""
You are a real estate assistant. Search the web for 3–5 recent rental listings similar to the following property:

- Address/Area: {pred.get('address', '')}
- Subdistrict Code: {pred.get('subdistrict_code', '')}
- Bedrooms: {pred.get('BEDROOMS', '')}
- Bathrooms: {pred.get('BATHROOMS', '')}
- Size: {pred.get('SIZE', '')} sq ft
- Property Type: {pred.get('PROPERTY TYPE', '')}

For each, provide:
- Address or area
- Monthly rent (in GBP)
- Brief description

Then, compare these rents to the predicted range: £{rent_range[0]}–£{rent_range[1]} and state if the prediction is in line with the market, too high, or too low.
"""
    response = invoke_with_tracking(
        llm,
        [HumanMessage(content=prompt)],
        "Market Comparison Analysis"
    )
    return response.content.strip()

from faiss_utils import semantic_search, load_faiss_index, record_to_text

# --- Enhanced Semantic Search with Milvus ---
def milvus_semantic_search(query_text: str, source_filter: str = None, top_k: int = 5):
    """
    Perform semantic search using Milvus instead of FAISS.
    
    Args:
        query_text: Search query
        source_filter: Optional source filter (e.g., 'rent_data')
        top_k: Number of results to return
        
    Returns:
        List of matching records
    """
    try:
        milvus_store = get_milvus_store()
        results = milvus_store.semantic_search(
            query_text=query_text,
            source_filter=source_filter,
            limit=top_k,
            similarity_threshold=0.6
        )
        return results
    except Exception as e:
        print(f"[WARNING] Milvus semantic search failed: {e}")
        # Fallback to FAISS if available
        try:
            return semantic_search(query_text, None, None, top_k)
        except:
            return []

def record_to_text_enhanced(record_dict: dict) -> str:
    """
    Enhanced record to text conversion with better formatting.
    """
    try:
        # Try the original function first
        return record_to_text(record_dict)
    except:
        # Fallback implementation
        parts = []
        for key, value in record_dict.items():
            if value and str(value).strip() and str(value) != 'nan':
                parts.append(f"{key}: {value}")
        return " | ".join(parts)

# --- Milvus Data Migration Functions ---
def migrate_existing_data_to_milvus():
    """
    Migrate existing FAISS data to Milvus.
    """
    try:
        from milvus_utils import migrate_faiss_to_milvus
        
        base_dir = os.path.dirname(__file__)
        
        # Migrate rent data
        rent_csv = os.path.join(base_dir, '../Rent Pricing AI/data/cleaned_rent_data.csv')
        if os.path.exists(rent_csv):
            migrate_faiss_to_milvus(
                faiss_index_path="",  # Not needed for CSV migration
                csv_data_path=rent_csv,
                source_name="rent_data"
            )
            print("Migrated rent data to Milvus")
        
        # Migrate raw rent data
        raw_rent_csv = os.path.join(base_dir, '../Rent Pricing AI/data/rent_ads_rightmove_extended.csv')
        if os.path.exists(raw_rent_csv):
            migrate_faiss_to_milvus(
                faiss_index_path="",
                csv_data_path=raw_rent_csv,
                source_name="raw_rent_data"
            )
            print("Migrated raw rent data to Milvus")
            
        print("Data migration to Milvus completed successfully")
        
    except Exception as e:
        print(f"Data migration failed: {e}")

# --- Main Conversation Function (Use This in Django Backend) ---
def handle_conversation(conversation_history, user_message, last_candidate_fields=None, 
                       last_intent=None, intent_completed=False, session_id=None, user_id=None):
    """
    Main conversation handler - automatically uses enhanced engine with Milvus when available,
    gracefully falls back to original engine if needed.
    
    This is the function your Django backend should call.
    """
    try:
        return enhanced_conversational_engine(
            conversation_history=conversation_history,
            user_message=user_message,
            last_candidate_fields=last_candidate_fields,
            last_intent=last_intent,
            intent_completed=intent_completed,
            session_id=session_id,
            user_id=user_id
        )
    except Exception as e:
        print(f"[ERROR] Enhanced conversation handler failed: {e}")
        # Fallback to original engine
        return conversational_engine(
            conversation_history=conversation_history,
            user_message=user_message,
            last_candidate_fields=last_candidate_fields,
            last_intent=last_intent,
            intent_completed=intent_completed
        )

# --- Demo and Testing Functions ---
def test_enhanced_features():
    """
    Test function for the enhanced conversational features.
    """
    try:
        print("🚀 Testing Enhanced Conversational AI...")
        print("=" * 60)
        
        # Test conversation intelligence
        conversation_ai = get_conversation_intelligence()
        
        test_messages = [
            "Hello! I'm new here",
            "Hi, I want to estimate rent for my 2 bedroom flat in London",
            "Can you help me screen a tenant with credit score 750, income £3000?",
            "What maintenance issues might occur this winter for a 15-year-old property?",
            "Thank you for your help!",
        ]
        
        print("\n🧠 Testing Advanced NER and Intent Detection:")
        print("-" * 50)
        
        for i, msg in enumerate(test_messages, 1):
            print(f"\n--- Test {i} ---")
            print(f"💬 User: {msg}")
            
            analysis = conversation_ai.analyze_message(msg)
            print(f"🎯 Primary Intent: {analysis.primary_intent.type.value} (confidence: {analysis.confidence:.2f})")
            
            if analysis.entities:
                print(f"📝 Extracted Entities:")
                for entity in analysis.entities:
                    print(f"   • {entity.label}: {entity.text} (confidence: {entity.confidence:.2f})")
            
            if len(analysis.intents) > 1:
                other_intents = [f"{intent.type.value} ({intent.confidence:.2f})" 
                               for intent in analysis.intents[1:3]]
                print(f"🔍 Other Intents: {', '.join(other_intents)}")
                
        print("\n" + "=" * 60)
        print("\n🤖 Testing Enhanced Conversation Engine:")
        print("-" * 50)
        
        # Test conversation flow
        conversation_history = []
        test_conversations = [
            "Hi there!",
            "I want to estimate rent for my property",
            "It's a 2 bedroom flat in Manchester, about 800 sq ft",
        ]
        
        for i, msg in enumerate(test_conversations, 1):
            print(f"\n--- Conversation Turn {i} ---")
            print(f"👤 User: {msg}")
            
            # Use enhanced conversation engine
            result = handle_conversation(
                conversation_history=conversation_history,
                user_message=msg,
                session_id="test_session",
                user_id="test_user"
            )
            
            print(f"🎯 Action: {result.get('action', 'unknown')}")
            print(f"🤖 Assistant: {result.get('response', 'No response')[:200]}...")
            
            if result.get('fields'):
                print(f"📋 Fields Collected: {list(result['fields'].keys())}")
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": msg})
            conversation_history.append({"role": "assistant", "content": result.get('response', '')})
            
        print("\n" + "=" * 60)
        print("\n✅ Enhanced features test completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Advanced NER with real estate domain knowledge")
        print("✅ Multi-intent detection and confidence scoring")
        print("✅ Context-aware entity extraction")
        print("✅ Greeting and small talk handling")
        print("✅ Enhanced conversation flow management")
        print("✅ Graceful fallback to original engine")
        print("✅ Session-based memory integration")
        
    except Exception as e:
        print(f"❌ Enhanced features test failed: {e}")
        import traceback
        traceback.print_exc()

def demo_greeting_intelligence():
    """
    Specific demo for greeting and small talk handling.
    """
    print("\n🤝 Testing Greeting Intelligence:")
    print("-" * 40)
    
    greetings = [
        "Hi",
        "Hello there!",
        "Good morning",
        "Hey, how are you?",
        "Thanks for your help",
        "Thank you so much!",
    ]
    
    for greeting in greetings:
        print(f"\n👤 User: {greeting}")
        result = handle_conversation([], greeting)
        print(f"🤖 Assistant: {result.get('response', 'No response')}")

if __name__ == "__main__":
    test_enhanced_features()
    demo_greeting_intelligence()
