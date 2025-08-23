import os
import re
import json
import csv
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import time
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
            
            # response += "If you'd like a more precise estimation, just let me know the missing details! "
            
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
                    response += f"If you'd like a more precise estimation, just let me know the missing details! I'd particularly love to know the {', '.join(missing_friendly[:-1])}"
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
            summary += f"- • **{k}**: {v}\n"
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
            f"- • **Estimated Monthly Rent:** £{int(predicted_rent)}\n"
            f"- • **Suggested Range:** £{int(lower_rent)}–£{int(upper_rent)}\n"
            f"- • **Confidence Level:** {round(float(confidence_percentage), 2)}%\n"
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
                md += f"- • {line.strip()}\n\n"
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
            
            # Special handling for different field types
            if k == "eviction_record":
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
            else:
                # For credit_score, income, rent - if value exists and was mentioned, consider it provided
                if v not in (None, '', 0, 0.0, False) and (current_has_field or recent_has_field):
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
                prompt += f"- • {field_descriptions[k]}\n"
            
            prompt += "\n\nOnce you drop a couple of these details on me, I'll whip up a preliminary screening for you! 🚀"
            
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
        'address': ['address', 'property address', 'location'],
        'age_years': ['age', 'property age', 'years old', 'age_years'],
        'last_service_years_ago': ['last service', 'last serviced', 'last maintenance', 'last_service_years_ago', 'time since last service'],
        'seasonality': ['seasonality', 'season', 'current season']
    }
    
    _model = None
    _address_map = None
    _model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../predictive_maintenance_ai/models/maintenance_rf_model.pkl'))
    _address_map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent Pricing AI/address_map.json'))
    
    def __init__(self):
        """Initialize maintenance prediction handler."""
        self.system_prompt = (
            "You are LandlordBuddy, an expert and professional AI assistant for landlords. "
            "You support rent pricing, tenant screening, and maintenance prediction. "
            "For maintenance prediction, you ONLY need these 4 specific pieces of information: "
            "1. Property address/location, 2. Property age in years, 3. Years since last maintenance/service, 4. Current season. "
            "Do NOT ask for number of units, property type, existing conditions, or other details - the model doesn't use those. "
            "You can provide estimates even with partial information using intelligent defaults. "
            "Always be helpful and provide immediate value while being transparent about assumptions. "
            "When any of the 4 required fields are missing, use contextual defaults and explain your assumptions clearly. "
            "Keep the conversation natural and focused only on these 4 maintenance prediction inputs. "
            "Never mention OpenAI or your own limitations. Always respond in markdown."
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
        """Generate contextually intelligent defaults based on provided information."""
        import datetime
        
        defaults = {}
        assumptions = []
        
        # Smart address default
        if not provided_fields.get('address'):
            address_map = self.get_address_map()
            # Use first address as default, but this should ideally be region-based
            defaults['address'] = next(iter(address_map.keys())) if address_map else "Unknown Location"
            assumptions.append("assuming average location")
        
        # Smart age defaults based on context
        if not provided_fields.get('age_years'):
            # If we have service info, infer reasonable age
            if provided_fields.get('last_service_years_ago'):
                service_gap = provided_fields['last_service_years_ago']
                if service_gap > 5:
                    defaults['age_years'] = max(15, service_gap * 2)  # Older property
                    assumptions.append(f"assuming {defaults['age_years']} year old property based on service history")
                else:
                    defaults['age_years'] = 10  # Moderate age
                    assumptions.append("assuming 10 year old property based on recent service")
            else:
                defaults['age_years'] = 12  # Average residential property age
                assumptions.append("assuming 12 year old property (average)")
        
        # Smart service history defaults
        if not provided_fields.get('last_service_years_ago'):
            age = provided_fields.get('age_years', defaults.get('age_years', 12))
            if age < 5:
                defaults['last_service_years_ago'] = 1  # New properties, recent service
                assumptions.append("assuming 1 year since last service (newer property)")
            elif age < 15:
                defaults['last_service_years_ago'] = 2  # Moderate age, regular service
                assumptions.append("assuming 2 years since last service (moderate age)")
            else:
                defaults['last_service_years_ago'] = 3  # Older properties might have longer gaps
                assumptions.append("assuming 3 years since last service (older property)")
        
        # Smart seasonality default
        if not provided_fields.get('seasonality'):
            current_month = datetime.datetime.now().month
            if current_month in [12, 1, 2]:
                defaults['seasonality'] = 'Winter'
            elif current_month in [3, 4, 5]:
                defaults['seasonality'] = 'Spring'
            elif current_month in [6, 7, 8]:
                defaults['seasonality'] = 'Summer'
            else:
                defaults['seasonality'] = 'Autumn'
            assumptions.append(f"assuming current season ({defaults['seasonality']})")
        
        return defaults, assumptions

    def calculate_confidence_score(self, provided_fields, total_fields):
        """Calculate confidence score based on provided vs estimated fields."""
        provided_count = sum(1 for field in self.required_fields 
                           if field in provided_fields and provided_fields[field] not in (None, '', 0, 0.0))
        
        base_confidence = (provided_count / len(self.required_fields)) * 100
        
        # Adjust confidence based on field importance and reasonableness
        if provided_fields.get('address') and provided_fields['address'] != "Unknown Location":
            base_confidence += 5  # Address is important for location-specific factors
        
        if provided_fields.get('age_years') and provided_fields.get('last_service_years_ago'):
            # Check if the combination is reasonable
            age = provided_fields['age_years']
            service_gap = provided_fields['last_service_years_ago']
            if service_gap <= age:  # Reasonable combination
                base_confidence += 5
        
        return min(95, max(25, int(base_confidence)))  # Cap between 25-95%

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
        # Use LLM/PydanticOutputParser for robust extraction (like rent/tenant handlers), with improved fallback
        from langchain.output_parsers import PydanticOutputParser
        from pydantic import BaseModel, Field, ValidationError
        from langchain_core.prompts import ChatPromptTemplate
        import re
        
        class MaintFields(BaseModel):
            address: str = Field('', description="The property address or location")
            age_years: int = Field(0, description="Property age in years")
            last_service_years_ago: int = Field(0, description="Years since last service")
            seasonality: str = Field('', description="Current season (Winter, Spring, Summer, Autumn)")

        all_text = "\n".join([m["content"] for m in conversation_history if m["role"] in ("user", "assistant")])
        all_text += "\n" + user_message

        # 1. Try LLM/Pydantic extraction
        parser = PydanticOutputParser(pydantic_object=MaintFields)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant for landlords. Extract the following fields from the conversation and user message. If a field is missing, use an empty string or 0. Output only the JSON object as specified by the schema: {format_instructions}"),
            ("user", "Conversation so far:\n{conversation}\nUser message:\n{user_message}")
        ])
        format_instructions = parser.get_format_instructions()
        prompt_value = prompt.format_prompt(
            conversation=all_text,
            user_message=user_message,
            format_instructions=format_instructions
        )
        try:
            response = invoke_with_tracking(
                self.chat,
                [HumanMessage(content=prompt_value.to_string())],
                "Maintenance Prediction - Field Extraction"
            )
            content = response.content.strip()
            parsed = parser.parse(content)
            fields = parsed.dict()
        except Exception:
            fields = dict(last_candidate_fields) if last_candidate_fields else {}

        # 2. Fallback: regex/natural language extraction for robust field parsing
        # Address: look for 'property at|property on|property in|property' ... up to 'constructed' or 'built' or 'last service' or ','
        if not fields.get('address'):
            addr_match = re.search(r"property (?:at|on|in)?\s*([A-Za-z0-9,\- ]+?)(?:,| constructed| built| last service| last serviced|\.|$)", user_message, re.IGNORECASE)
            if addr_match:
                fields['address'] = addr_match.group(1).strip()
            else:
                # Try to grab first location-like phrase
                addr_match2 = re.search(r"at ([A-Za-z0-9,\- ]+?)(?:,| constructed| built| last service| last serviced|\.|$)", user_message, re.IGNORECASE)
                if addr_match2:
                    fields['address'] = addr_match2.group(1).strip()

        # Age: "constructed X years ago" or "built X years ago"
        if not fields.get('age_years'):
            age_match = re.search(r"(?:constructed|built)\s*(\d{1,3})\s*years? ago", user_message, re.IGNORECASE)
            if age_match:
                try:
                    fields['age_years'] = int(age_match.group(1))
                except Exception:
                    fields['age_years'] = 0

        # Last service: "last service Y years ago" or "last serviced Y years ago"
        if not fields.get('last_service_years_ago'):
            svc_match = re.search(r"last (?:service|serviced|maintenance)[^\d]*(\d{1,3})\s*years? ago", user_message, re.IGNORECASE)
            if svc_match:
                try:
                    fields['last_service_years_ago'] = int(svc_match.group(1))
                except Exception:
                    fields['last_service_years_ago'] = 0

        # Seasonality: "this winter", "this summer", etc.
        if not fields.get('seasonality'):
            season_match = re.search(r"this (winter|spring|summer|autumn|fall)", user_message, re.IGNORECASE)
            if season_match:
                fields['seasonality'] = season_match.group(1).capitalize()

        # Only keep required fields, with correct types/defaults
        clean_fields = {
            'address': str(fields.get('address', '')),
            'age_years': int(fields.get('age_years', 0) or 0),
            'last_service_years_ago': int(fields.get('last_service_years_ago', 0) or 0),
            'seasonality': str(fields.get('seasonality', '')),
        }
        return clean_fields

    def run_model_with_smart_defaults(self, provided_fields):
        """Run model with smart defaults for missing fields."""
        # Get smart defaults for missing fields
        defaults, assumptions = self.get_smart_defaults(provided_fields)
        
        # Merge provided fields with defaults
        complete_fields = {}
        for field in self.required_fields:
            if field in provided_fields and provided_fields[field] not in (None, '', 0, 0.0):
                complete_fields[field] = provided_fields[field]
            else:
                complete_fields[field] = defaults.get(field, '')
        
        # Calculate confidence score
        confidence = self.calculate_confidence_score(provided_fields, complete_fields)
        
        # Run the model
        model = self.get_model()
        encoded_fields = self.encode_fields_for_model(complete_fields)
        input_df = pd.DataFrame([{
            'address': encoded_fields['address'],
            'age_years': encoded_fields['age_years'],
            'last_service_years_ago': encoded_fields['last_service_years_ago'],
            'seasonality': encoded_fields['seasonality']
        }])
        risk_score = model.predict(input_df)[0]
        
        # Determine missing fields for user feedback
        missing_fields = [field for field in self.required_fields 
                         if field not in provided_fields or provided_fields[field] in (None, '', 0, 0.0)]
        
        return risk_score, confidence, assumptions, missing_fields, complete_fields

    def format_prediction_result(self, risk_score, confidence, assumptions, missing_fields, complete_fields):
        """Format the prediction result with assumptions and confidence."""
        
        # Map risk score to recommended action with specific recommendations
        if risk_score > 7:
            action = 'Immediate Action'
            recommendations = [
                "Schedule emergency maintenance inspection within 1-2 days",
                "Check HVAC, plumbing, and electrical systems immediately",
                "Consider temporary tenant relocation if safety concerns arise",
                "Contact professional maintenance contractors urgently"
            ]
        elif risk_score > 4:
            action = 'Monitor'
            recommendations = [
                "Schedule comprehensive maintenance inspection within 2-4 weeks",
                "Perform preventive maintenance on aging systems",
                "Monitor tenant reports of any issues closely",
                "Plan budget for potential repairs in the next 3-6 months"
            ]
        else:
            action = 'Routine'
            recommendations = [
                "Continue with regular maintenance schedule",
                "Perform annual safety checks and inspections",
                "Keep maintenance reserves for unexpected issues",
                "Monitor seasonal maintenance needs (heating/cooling systems)"
            ]
        
        # Create the main result
        result = f"## 🏠 Maintenance Risk Assessment\n\n"
        
        # Show what was used for prediction
        result += "**Property Details Used:**\n"
        for field, value in complete_fields.items():
            field_name = field.replace('_', ' ').title()
            result += f"- **{field_name}:** {value}\n"
        
        if assumptions:
            result += f"\n*Assumptions made: {', '.join(assumptions)}*\n"
        
        result += f"\n**Maintenance Risk Score:** {risk_score:.1f}/10\n"
        result += f"**Recommended Action:** {action}\n"
        result += f"**Confidence Level:** {confidence}%\n\n"
        
        result += "**Recommended Actions:**\n"
        for i, rec in enumerate(recommendations, 1):
            result += f"{i}. {rec}\n"
        
        # Add missing fields request at the end if any
        if missing_fields:
            missing_friendly = []
            for field in missing_fields:
                if field == 'age_years':
                    missing_friendly.append('property age')
                elif field == 'last_service_years_ago':
                    missing_friendly.append('years since last maintenance')
                else:
                    missing_friendly.append(field.replace('_', ' '))
            
            result += f"\n---\n*For a more precise prediction, please provide: {', '.join(missing_friendly)}*"
        
        return result

    def summarize_fields(self, fields):
        summary = "**Property Information for Maintenance Prediction:**\n\n"
        for k, v in fields.items():
            summary += f"- **{k}**: {v}\n"
        summary += "\nIs this information correct? Please confirm to proceed with the maintenance risk assessment."
        return summary

    def needs_confirmation(self, user_message):
        confirmation_phrases = ["yes", "correct", "that's right", "yep", "confirmed", "go ahead", "proceed"]
        return user_message.strip().lower() in confirmation_phrases

    def handle(self, conversation_history, user_message, last_candidate_fields=None):
        # Extract fields from the current message
        candidate_fields = self.extract_fields(user_message, conversation_history, last_candidate_fields)
        
        # Merge with last candidate fields to preserve previously collected info
        merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
        for k, v in candidate_fields.items():
            if v not in (None, '', 0, 0.0):  # Only update if we have a meaningful value
                merged_fields[k] = v
        
        # If user has provided ANY information related to maintenance prediction, run the model
        if any(merged_fields.get(field) not in (None, '', 0, 0.0) for field in self.required_fields):
            # Run model with smart defaults
            risk_score, confidence, assumptions, missing_fields, complete_fields = self.run_model_with_smart_defaults(merged_fields)
            result = self.format_prediction_result(risk_score, confidence, assumptions, missing_fields, complete_fields)
            return {"response": result, "action": "maintenance_prediction", "fields": merged_fields}
        
        # If no relevant information provided, use LLM to handle the conversation
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
        
        return {"response": reply, "action": "chat", "fields": merged_fields}

# --- Enhanced Intent Detection and Entity Recognition ---
def detect_intent(user_message, conversation_history=None):
    """
    Enhanced intent detection with better context awareness and no context dropping.
    """
    try:
        # Use enhanced conversation intelligence if available
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
        print(f"[WARNING] Enhanced intent detection failed, using improved fallback: {e}")
        return improved_fallback_intent_detection(user_message, conversation_history)

def improved_fallback_intent_detection(user_message, conversation_history=None):
    """
    Improved fallback intent detection with better context awareness.
    """
    msg = user_message.lower().strip()
    
    # Strong greeting indicators
    greeting_patterns = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "appreciate", "great", "awesome", "perfect"
    ]
    
    # Strong intent indicators with context
    rent_patterns = [
        "rent", "rental price", "how much", "estimate", "pricing", "cost", 
        "market rate", "charge", "monthly", "price prediction", "valuation"
    ]
    
    tenant_patterns = [
        "tenant", "screen", "applicant", "background", "credit", "income",
        "references", "application", "qualify", "approve", "reject"
    ]
    
    maintenance_patterns = [
        "maintenance", "repair", "fix", "upkeep", "service", "predict",
        "when will", "how long", "age", "last service", "risk", "inspection"
    ]
    
    # Check for explicit intent switches
    switch_patterns = [
        "instead", "actually", "wait", "no", "different", "other",
        "rent prediction", "tenant screening", "maintenance prediction"
    ]
    
    # Context-aware detection
    if any(pattern in msg for pattern in greeting_patterns) and len(msg.split()) <= 5:
        return "greeting"
    elif any(pattern in msg for pattern in switch_patterns):
        # User wants to change intent - detect new intent
        if any(pattern in msg for pattern in rent_patterns):
            return "rent_prediction"
        elif any(pattern in msg for pattern in tenant_patterns):
            return "tenant_screening"
        elif any(pattern in msg for pattern in maintenance_patterns):
            return "maintenance_prediction"
    elif any(pattern in msg for pattern in rent_patterns):
        return "rent_prediction"
    elif any(pattern in msg for pattern in tenant_patterns):
        return "tenant_screening"
    elif any(pattern in msg for pattern in maintenance_patterns):
        return "maintenance_prediction"
    
    # Context from conversation history
    if conversation_history:
        recent_context = " ".join([m.get('content', '') for m in conversation_history[-3:]])
        recent_context = recent_context.lower()
        
        if any(pattern in recent_context for pattern in rent_patterns):
            return "rent_prediction"
        elif any(pattern in recent_context for pattern in tenant_patterns):
            return "tenant_screening"
        elif any(pattern in recent_context for pattern in maintenance_patterns):
            return "maintenance_prediction"
    
    return None

# --- Enhanced LLM-based Intent Detection with Full Context ---
def llm_detect_intent(conversation_history, user_message):
    """
    Enhanced LLM-based intent detection with full conversation context preservation.
    """
    try:
        # Use the enhanced conversation intelligence system
        conversation_ai = get_conversation_intelligence()
        analysis = conversation_ai.analyze_message(user_message, conversation_history)
        
        # Return comprehensive analysis
        return {
            'primary_intent': analysis.primary_intent.type.value,
            'confidence': analysis.confidence,
            'all_intents': [intent.type.value for intent in analysis.intents],
            'entities': [{'label': e.label, 'text': e.text, 'confidence': e.confidence} for e in analysis.entities],
            'requires_clarification': analysis.requires_clarification,
            'clarification_message': analysis.clarification_message,
            'context_preserved': True
        }
        
    except Exception as e:
        print(f"[WARNING] Enhanced LLM intent detection failed, using improved fallback: {e}")
        return improved_llm_fallback(conversation_history, user_message)

def improved_llm_fallback(conversation_history, user_message):
    """
    Improved LLM fallback with better context handling and no dropping.
    """
    from langchain_openai import ChatOpenAI
    
    try:
        chat = ChatOpenAI(model="gpt-4", temperature=0.1, openai_api_key=openai_api_key)
        
        # Preserve ALL conversation context - no dropping
        full_context = ""
        if conversation_history:
            full_context = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])
        
        # Enhanced prompt with better instructions
        enhanced_prompt = f"""You are LandlordBuddy's intent detection system. Analyze the ENTIRE conversation context to understand what the user wants.

AVAILABLE INTENTS:
- rent_prediction: User wants rent estimates, pricing, market rates, property valuation
- tenant_screening: User wants to screen applicants, check backgrounds, approve/reject tenants
- maintenance_prediction: User wants maintenance forecasts, repair predictions, service scheduling
- greeting: User is greeting, thanking, or making small talk
- clarify_intent: User's intent is unclear or they want to switch tasks

RULES:
1. Consider the FULL conversation context, not just the latest message
2. Look for intent switches: "actually", "instead", "wait", "no", "different"
3. If user provides property details, they likely want predictions
4. If user mentions specific people/applicants, they likely want screening
5. If user asks about repairs/age/service, they likely want maintenance prediction

Full Conversation Context:
{full_context}

Current User Message: {user_message}

Based on the ENTIRE context above, what is the user's intent? Respond with ONLY the intent keyword (rent_prediction, tenant_screening, maintenance_prediction, greeting, or clarify_intent)."""

        response = invoke_with_tracking(
            chat,
            [HumanMessage(content=enhanced_prompt)],
            "Enhanced Intent Detection - LLM Fallback"
        )
        
        intent = response.content.strip().lower()
        
        # Map variations to standard intents
        intent_mapping = {
            'rent': 'rent_prediction',
            'rental': 'rent_prediction',
            'price': 'rent_prediction',
            'pricing': 'rent_prediction',
            'tenant': 'tenant_screening',
            'screening': 'tenant_screening',
            'applicant': 'tenant_screening',
            'maintenance': 'maintenance_prediction',
            'repair': 'maintenance_prediction',
            'service': 'maintenance_prediction',
            'hello': 'greeting',
            'hi': 'greeting',
            'thank': 'greeting'
        }
        
        # Check for partial matches
        for key, mapped_intent in intent_mapping.items():
            if key in intent:
                return {
                    'primary_intent': mapped_intent,
                    'confidence': 0.85,
                    'context_preserved': True,
                    'method': 'llm_fallback'
                }
        
        # Direct intent matches
        valid_intents = ['rent_prediction', 'tenant_screening', 'maintenance_prediction', 'greeting', 'clarify_intent']
        if intent in valid_intents:
            return {
                'primary_intent': intent,
                'confidence': 0.9,
                'context_preserved': True,
                'method': 'llm_fallback'
            }
        
        return {
            'primary_intent': 'clarify_intent',
            'confidence': 0.6,
            'context_preserved': True,
            'method': 'llm_fallback_uncertain'
        }
        
    except Exception as e:
        print(f"[ERROR] LLM fallback failed: {e}")
        # Last resort - keyword detection with context
        return {
            'primary_intent': improved_fallback_intent_detection(user_message, conversation_history) or 'clarify_intent',
            'confidence': 0.7,
            'context_preserved': True,
            'method': 'keyword_fallback'
        }

# --- Enhanced Intent Switch Detection ---
def user_requests_intent_switch(user_message, conversation_history=None):
    """
    Enhanced detection of intent switching with context awareness.
    """
    msg = user_message.lower().strip()
    
    # Strong switch indicators
    strong_switch_phrases = [
        "forget it", "never mind", "actually", "instead", "wait", "no",
        "different", "other", "wrong", "not what i meant", "change",
        "switch to", "let's do", "i want to do"
    ]
    
    # Explicit intent mentions
    explicit_intents = [
        "rent prediction", "tenant screening", "maintenance prediction",
        "rent pricing", "tenant check", "maintenance forecast",
        "do rent", "do tenant", "do maintenance",
        "help with rent", "help with tenant", "help with maintenance"
    ]
    
    # Context-aware switching
    has_switch_indicator = any(phrase in msg for phrase in strong_switch_phrases)
    has_explicit_intent = any(intent in msg for intent in explicit_intents)
    
    # Check if user is providing new information that suggests different intent
    if conversation_history:
        last_assistant_msg = next((m for m in reversed(conversation_history) if m['role'] == 'assistant'), None)
        if last_assistant_msg:
            last_content = last_assistant_msg.get('content', '').lower()
            
            # If assistant was asking for rent info but user mentions tenant/maintenance
            if 'rent' in last_content and ('tenant' in msg or 'applicant' in msg):
                return True
            elif 'tenant' in last_content and ('rent' in msg or 'price' in msg):
                return True
            elif 'maintenance' in last_content and ('rent' in msg or 'tenant' in msg):
                return True
    
    return has_switch_indicator or has_explicit_intent

# --- Context-Preserving Conversational Engine ---
def enhanced_conversational_engine(conversation_history, user_message, last_candidate_fields=None, 
                                 last_intent=None, intent_completed=False, session_id=None, user_id=None):
    """
    Enhanced modular conversational engine with complete context preservation.
    """
    try:
        # Initialize services with error handling
        milvus_store = None
        conversation_ai = None
        
        try:
            milvus_store = get_milvus_store()
            conversation_ai = get_conversation_intelligence()
        except Exception as e:
            print(f"[WARNING] Could not initialize advanced services: {e}")
        
        # Preserve full conversation history - NO DROPPING
        full_history = conversation_history.copy() if conversation_history else []
        
        # Store user message in Milvus if available
        if milvus_store and session_id and user_id:
            try:
                milvus_store.store_chat_message(
                    session_id=session_id,
                    user_id=user_id,
                    message_type="user",
                    content=user_message,
                    intent="",  # Will be filled after detection
                    entities={}
                )
            except Exception as e:
                print(f"[WARNING] Could not store in Milvus: {e}")
        
        # Get relevant conversation memory without dropping current context
        relevant_memory = []
        if milvus_store and session_id:
            try:
                relevant_memory = milvus_store.retrieve_chat_memory(
                    session_id=session_id,
                    query_text=user_message,
                    limit=10  # Increased for better context
                )
            except Exception as e:
                print(f"[WARNING] Could not retrieve from Milvus: {e}")
        
        # Merge memory with current history (avoid duplicates but preserve order)
        enhanced_history = full_history.copy()
        for memory in reversed(relevant_memory):  # Add oldest first
            memory_content = memory.get("content", "")
            # Only add if not already in recent history
            if not any(msg.get("content", "") == memory_content for msg in enhanced_history[-5:]):
                enhanced_history.insert(0, {
                    "role": memory["message_type"],
                    "content": memory_content
                })
        
        # Enhanced intent detection with full context
        primary_intent = None
        extracted_entities = {}
        analysis_confidence = 0.8
        
        if conversation_ai:
            try:
                analysis = conversation_ai.analyze_message(
                    user_message=user_message,
                    conversation_history=enhanced_history,
                    current_fields=last_candidate_fields,
                    session_id=session_id
                )
                primary_intent = analysis.primary_intent.type
                extracted_entities = {entity.label: entity.normalized_value or entity.text 
                                    for entity in analysis.entities}
                analysis_confidence = analysis.confidence
            except Exception as e:
                print(f"[WARNING] AI analysis failed: {e}")
        
        # Fallback to enhanced LLM detection
        if not primary_intent:
            detection_result = llm_detect_intent(enhanced_history, user_message)
            if isinstance(detection_result, dict):
                primary_intent = detection_result.get('primary_intent')
                analysis_confidence = detection_result.get('confidence', 0.8)
            else:
                primary_intent = detection_result
        
        # Handle intent switching with full context awareness
        if user_requests_intent_switch(user_message, enhanced_history):
            # User wants to switch - detect new intent but preserve some context
            new_intent_detection = llm_detect_intent(enhanced_history, user_message)
            if isinstance(new_intent_detection, dict):
                primary_intent = new_intent_detection.get('primary_intent')
            else:
                primary_intent = new_intent_detection
            
            # Reset fields when switching but preserve conversation history
            last_candidate_fields = {}
            intent_completed = False
            last_intent = None
        
        # Continue with current intent if active and not switching
        elif last_intent and not intent_completed and not user_requests_intent_switch(user_message, enhanced_history):
            primary_intent = last_intent
        
        # Handle different intents with enhanced context
        result = None
        
        if primary_intent == IntentType.GREETING or primary_intent == "greeting":
            result = handle_enhanced_greeting(user_message, enhanced_history)
        
        elif primary_intent == IntentType.SMALL_TALK or primary_intent == "small_talk":
            result = handle_enhanced_small_talk(user_message, enhanced_history)
        
        elif primary_intent == IntentType.RENT_PREDICTION or primary_intent == "rent_prediction":
            handler = RentPredictionHandler()
            merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
            merged_fields.update(extracted_entities)
            result = handler.handle(enhanced_history, user_message, merged_fields)
            result["last_intent"] = "rent_prediction" if result.get("action") != "rent_prediction" else None
            result["intent_completed"] = result.get("action") == "rent_prediction"
        
        elif primary_intent == IntentType.TENANT_SCREENING or primary_intent == "tenant_screening":
            handler = TenantScreeningHandler()
            merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
            merged_fields.update(extracted_entities)
            result = handler.handle(enhanced_history, user_message, merged_fields)
            result["last_intent"] = "tenant_screening" if result.get("action") != "screen_tenant" else None
            result["intent_completed"] = result.get("action") == "screen_tenant"
        
        elif primary_intent == IntentType.MAINTENANCE_PREDICTION or primary_intent == "maintenance_prediction":
            handler = MaintenancePredictionHandler()
            merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
            merged_fields.update(extracted_entities)
            result = handler.handle(enhanced_history, user_message, merged_fields)
            result["last_intent"] = "maintenance_prediction" if result.get("action") != "maintenance_prediction" else None
            result["intent_completed"] = result.get("action") == "maintenance_prediction"
        
        # Handle unclear or multiple intents
        else:
            result = handle_unclear_intent(user_message, enhanced_history, analysis_confidence)
        
        # Store assistant response in Milvus if available
        if milvus_store and session_id and user_id and result:
            try:
                milvus_store.store_chat_message(
                    session_id=session_id,
                    user_id=user_id,
                    message_type="assistant",
                    content=result.get("response", ""),
                    intent=str(primary_intent) if primary_intent else "",
                    entities=extracted_entities
                )
            except Exception as e:
                print(f"[WARNING] Could not store assistant response: {e}")
        
        # Add metadata about context preservation
        if result:
            result["context_preserved"] = True
            result["history_length"] = len(enhanced_history)
            result["confidence"] = analysis_confidence
        
        return result or handle_fallback_response()
        
    except Exception as e:
        print(f"[ERROR] Enhanced engine failed: {e}")
        # Fallback to improved original engine
        return improved_conversational_engine(conversation_history, user_message, last_candidate_fields, 
                                            last_intent, intent_completed)

def handle_enhanced_greeting(user_message, conversation_history):
    """Enhanced greeting handler with context awareness."""
    user_greeting = user_message.lower()
    
    # Check conversation context for personalization
    context_info = ""
    if conversation_history:
        recent_topics = []
        for msg in conversation_history[-5:]:
            content = msg.get('content', '').lower()
            if 'rent' in content:
                recent_topics.append('rent')
            elif 'tenant' in content:
                recent_topics.append('tenant')
            elif 'maintenance' in content:
                recent_topics.append('maintenance')
        
        if recent_topics:
            context_info = f" Ready to continue with {', '.join(set(recent_topics))} or explore something new?"

    if "morning" in user_greeting:
        response_text = f"🌅 Good morning! I'm LandlordBuddy, your AI property management assistant.{context_info}"
    elif "evening" in user_greeting or "night" in user_greeting:
        response_text = f"🌙 Good evening! LandlordBuddy here to help with your property needs.{context_info}"
    elif any(word in user_greeting for word in ["thanks", "thank you", "appreciate"]):
        response_text = "You're very welcome! 😊 Happy to help with your property management needs. What can I assist you with next?"
    elif "hi" in user_greeting or "hello" in user_greeting:
        response_text = f"👋 Hey there! I'm LandlordBuddy, your AI property partner.{context_info}"
    else:
        response_text = f"✨ Welcome! I'm LandlordBuddy, here to make property management easier.{context_info}"

    return {
        "response": response_text,
        "action": "greeting",
        "fields": {},
        "last_intent": None,
        "intent_completed": True
    }

def handle_enhanced_small_talk(user_message, conversation_history):
    """Enhanced small talk handler."""
    responses = [
        "I appreciate the conversation! How can I help you with your property management today?",
        "Thanks for sharing! What property task can I assist you with?",
        "That's interesting! Let's see how I can help with your rental business.",
        "I'm here to help with your property needs. What would you like to work on?"
    ]
    
    import random
    response = random.choice(responses)
    
    return {
        "response": response,
        "action": "small_talk",
        "fields": {},
        "last_intent": None,
        "intent_completed": True
    }

def handle_unclear_intent(user_message, conversation_history, confidence):
    """Handle unclear or low-confidence intents with context."""
    
    # Analyze context for hints
    context_hints = []
    if conversation_history:
        recent_content = " ".join([m.get('content', '') for m in conversation_history[-3:]])
        if 'rent' in recent_content.lower():
            context_hints.append('rent prediction')
        if 'tenant' in recent_content.lower():
            context_hints.append('tenant screening')  
        if 'maintenance' in recent_content.lower():
            context_hints.append('maintenance prediction')
    
    base_message = "I want to help you, but I'm not quite sure what you're looking for. "
    
    if context_hints:
        base_message += f"Based on our conversation, you might want help with {', '.join(context_hints)}. "
    
    base_message += (
        "\n\n**Here's what I can help you with:**\n"
        "🏠 **Rent Prediction** - Estimate market rent for your property\n"
        "👥 **Tenant Screening** - Evaluate potential tenants\n" 
        "🔧 **Maintenance Prediction** - Forecast maintenance needs\n\n"
        "Just let me know which one interests you, or describe what you need help with!"
    )
    
    return {
        "response": base_message,
        "action": "clarify_intent", 
        "fields": {},
        "last_intent": None,
        "intent_completed": True
    }

def handle_fallback_response():
    """Final fallback response."""
    return {
        "response": (
            "I'm having trouble processing your request right now. "
            "Could you please tell me specifically if you need help with:\n"
            "- Rent pricing\n- Tenant screening\n- Maintenance predictions\n\n"
            "I'm here to help once I understand what you're looking for!"
        ),
        "action": "error_fallback",
        "fields": {},
        "last_intent": None,
        "intent_completed": True
    }

# --- Improved Original Engine (Enhanced Fallback) ---
def improved_conversational_engine(conversation_history, user_message, last_candidate_fields=None, 
                                 last_intent=None, intent_completed=False):
    """
    Improved version of the original conversational engine with better context handling.
    """
    # Preserve full conversation history
    full_history = conversation_history.copy() if conversation_history else []
    
    # Enhanced intent switching detection
    if user_requests_intent_switch(user_message, full_history):
        detected_intent = llm_detect_intent(full_history, user_message)
        if isinstance(detected_intent, dict):
            intent = detected_intent.get('primary_intent')
        else:
            intent = detected_intent
        intent_completed = False
        last_candidate_fields = {}
    elif last_intent and not intent_completed:
        intent = last_intent
    else:
        detected_intent = llm_detect_intent(full_history, user_message)
        if isinstance(detected_intent, dict):
            intent = detected_intent.get('primary_intent')
        else:
            intent = detected_intent
        intent_completed = False
        if intent != last_intent:
            last_candidate_fields = {}

    # Handle intents with full context
    if intent == "greeting":
        return handle_enhanced_greeting(user_message, full_history)
    
    elif intent == "rent_prediction":
        handler = RentPredictionHandler()
        rent_fields = {}
        if last_candidate_fields:
            for field in handler.required_fields:
                if field in last_candidate_fields:
                    rent_fields[field] = last_candidate_fields[field]
        result = handler.handle(full_history, user_message, rent_fields)
        if result.get("action") == "rent_prediction":
            intent_completed = True
        return {**result, "last_intent": intent if not intent_completed else None, "intent_completed": intent_completed}
    
    elif intent == "tenant_screening":
        handler = TenantScreeningHandler()
        tenant_fields = {}
        if last_candidate_fields:
            for field in handler.required_fields:
                if field in last_candidate_fields:
                    tenant_fields[field] = last_candidate_fields[field]
        result = handler.handle(full_history, user_message, tenant_fields)
        if result.get("action") == "screen_tenant":
            intent_completed = True
        return {**result, "last_intent": intent if not intent_completed else None, "intent_completed": intent_completed}
    
    elif intent == "maintenance_prediction":
        handler = MaintenancePredictionHandler()
        maintenance_fields = {}
        if last_candidate_fields:
            for field in handler.required_fields:
                if field in last_candidate_fields:
                    maintenance_fields[field] = last_candidate_fields[field]
        result = handler.handle(full_history, user_message, maintenance_fields)
        if result.get("action") == "maintenance_prediction":
            intent_completed = True
        return {**result, "last_intent": intent if not intent_completed else None, "intent_completed": intent_completed}
    
    else:
        return handle_unclear_intent(user_message, full_history, 0.6)
        
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
        # Fallback to improved original engine
        return improved_conversational_engine(
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
                    print(f"   - • {entity.label}: {entity.text} (confidence: {entity.confidence:.2f})")
            
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

# --- Main Entry Point Function ---
def conversational_engine(conversation_history, user_message, last_candidate_fields=None, last_intent=None, intent_completed=False):
    """
    Main entry point for the conversational engine - routes to the best available handler.
    This is called by the Django consumer.
    """
    return handle_conversation(
        conversation_history=conversation_history,
        user_message=user_message,
        last_candidate_fields=last_candidate_fields,
        last_intent=last_intent,
        intent_completed=intent_completed
    )

if __name__ == "__main__":
    test_enhanced_features()
    demo_greeting_intelligence()