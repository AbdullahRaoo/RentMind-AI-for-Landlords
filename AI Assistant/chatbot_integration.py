import os
import pandas as pd
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Dynamically construct the absolute path to the model file
current_dir = os.path.dirname(__file__)  # Directory of the current script
model_path = os.path.join(current_dir, "../Rent Pricing AI/rent_xgboost_model.json")

# Load the trained model
model = xgb.Booster()
model.load_model(model_path)

def predict_rent(input_data):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Prepare the data for XGBoost
    dinput = xgb.DMatrix(input_df, missing=np.nan)

    # Make predictions
    predicted_log_rent = model.predict(dinput)
    predicted_rent = np.expm1(predicted_log_rent)  # Reverse log transformation

    # Calculate rent range (±10%)
    predicted_rent = predicted_rent[0]
    lower_rent = int(predicted_rent - 0.10 * predicted_rent)
    upper_rent = int(predicted_rent + 0.10 * predicted_rent)

    # Define RMSE (use the RMSE from your training process, e.g., £1039.64)
    rmse = 1039.64  # Replace with the actual RMSE from your training output

    # Calculate confidence score
    confidence = max(0, 1 - (rmse / predicted_rent))  # Confidence level as a percentage
    confidence_percentage = round(confidence * 100, 2)

    return predicted_rent, lower_rent, upper_rent, confidence_percentage

def generate_explanation_with_langchain(input_data, predicted_rent, lower_rent, upper_rent, confidence_percentage):
    # Initialize LangChain's ChatOpenAI
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=[
            "address", "subdistrict_code", "bedrooms", "bathrooms", "size",
            "property_type", "avg_distance", "station_count", "predicted_rent",
            "lower_rent", "upper_rent", "confidence_percentage"
        ],
        template="""
        You are LandlordBuddy, an expert and friendly AI assistant for landlords. 
        Please explain the rent prediction below in a helpful, conversational, and reassuring tone. Use markdown formatting for clarity (bold, bullet points, etc). 
        Make your response easy to read and actionable for a landlord who may not be an expert.

        ---
        **Property Details:**
        - **Address (encoded):** {address}
        - **Subdistrict Code:** {subdistrict_code}
        - **Bedrooms:** {bedrooms}
        - **Bathrooms:** {bathrooms}
        - **Size:** {size} sq ft
        - **Property Type (encoded):** {property_type}
        - **Average Distance to Nearest Station:** {avg_distance} miles
        - **Nearest Station Count:** {station_count}

        **Predicted Rent:**
        - **Estimated Monthly Rent:** £{predicted_rent}
        - **Suggested Range:** £{lower_rent}–£{upper_rent}
        - **Confidence Level:** {confidence_percentage}%

        ---        Please provide:
        - A concise, professional summary of the prediction (avoid casual or overly friendly language)
        - What factors most influenced the price
        - Any tips or next steps for the landlord
        - Use markdown for formatting (bold, lists, etc)
        """
    )

    # Format the prompt with input data
    prompt = prompt_template.format(
        address=input_data['address'],
        subdistrict_code=input_data['subdistrict_code'],
        bedrooms=input_data['BEDROOMS'],
        bathrooms=input_data['BATHROOMS'],
        size=input_data['SIZE'],
        property_type=input_data['PROPERTY TYPE'],
        avg_distance=input_data['avg_distance_to_nearest_station'],
        station_count=input_data['nearest_station_count'],
        predicted_rent=int(predicted_rent),
        lower_rent=lower_rent,
        upper_rent=upper_rent,
        confidence_percentage=confidence_percentage
    )

    # Generate explanation using LangChain
    response = chat.invoke([HumanMessage(content=prompt)])
    explanation = response.content.strip()
    return explanation

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
        "address", "subdistrict_code", "BEDROOMS", "BATHROOMS", "SIZE", "PROPERTY TYPE", "avg_distance_to_nearest_station", "nearest_station_count"
    ]
    FIELD_SYNONYMS = {
        "address": ["address", "location", "property address"],
        "subdistrict_code": ["subdistrict code", "code", "postcode", "postal code", "ub8"],
        "BEDROOMS": ["bedrooms", "number of bedrooms", "bed room", "beds", "bed"],
        "BATHROOMS": ["bathrooms", "number of bathrooms", "bathroom", "washroom", "baths", "bath"],
        "SIZE": ["size", "area", "square feet", "sq ft", "sqft", "foot", "feet"],
        "PROPERTY TYPE": ["property type", "type", "apartment", "house", "flat"],
        "avg_distance_to_nearest_station": ["average distance to nearest station", "avg distance", "station distance", "distance to station", "average station distance", "avg station distance", "distance"],
        "nearest_station_count": ["number of nearby stations", "number of stations", "station count", "stations", "nearby stations", "stations nearby"]
    }

    def __init__(self):
        self.system_prompt = (
            "You are LandlordBuddy, an expert and friendly AI assistant for landlords. "
            "You support rent pricing, tenant screening, and maintenance prediction. "
            "If the user asks for a feature you support, guide them to provide all required info. "
            "If info is missing, ask for it in a friendly, casual way. "
            "You must be flexible in understanding user input: users may provide information in any format, not just JSON or structured lists. "
            "You should do your best to interpret and extract the required details for rent prediction even if the user uses casual language, synonyms, or different phrasings. "
            "For example, if the user says 'avg distance', 'average station distance', 'station distance', or any similar phrase, you should map it to 'avg_distance_to_nearest_station'. "
            "Similarly, for all required fields, try to match and extract the information even if the user does not use the exact field names. "
            "The required information for rent prediction is: address, subdistrict_code, BEDROOMS, BATHROOMS, SIZE (in sq ft), PROPERTY TYPE, avg_distance_to_nearest_station, nearest_station_count. "
            "When you have all the required information, summarize the details in a clear, markdown-formatted list (not JSON or code block), and politely ask the user to confirm if the details are correct for rent estimation. "
            "Do NOT say you will provide the rent estimate later or mention any limitations. Wait for user confirmation before proceeding. "
            "Once the user confirms, proceed to provide the rent prediction and explanation using the model results, in clear markdown. "
            "If the request is not supported, politely say so. "
            "Never mention OpenAI or your own limitations. "
            "Always keep the conversation natural and helpful. "
            "Respond in markdown."
        )
        self.chat = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

    def extract_fields(self, user_message, conversation_history, last_candidate_fields=None):
        # Use LangChain's PydanticOutputParser for robust extraction
        from langchain.output_parsers import PydanticOutputParser
        from pydantic import BaseModel, Field, ValidationError
        from langchain_core.prompts import ChatPromptTemplate
        import re
        # Only attempt extraction if the intent is rent prediction
        if detect_intent(user_message, conversation_history) != "rent_prediction":
            return last_candidate_fields or {}
        class RentFields(BaseModel):
            address: str = Field(..., description="The property address or location")
            subdistrict_code: str = Field(..., description="The subdistrict code or postcode")
            BEDROOMS: int = Field(..., description="Number of bedrooms")
            BATHROOMS: int = Field(..., description="Number of bathrooms")
            SIZE: float = Field(..., description="Size in square feet")
            PROPERTY_TYPE: str = Field(..., description="Property type (e.g. flat, house, apartment)")
            avg_distance_to_nearest_station: float = Field(..., description="Average distance to nearest station in miles")
            nearest_station_count: int = Field(..., description="Number of nearby stations")

        parser = PydanticOutputParser(pydantic_object=RentFields)
        # Compose the full conversation for context
        all_text = "\n".join([m["content"] for m in conversation_history if m["role"] in ("user", "assistant")])
        all_text += "\n" + user_message
        
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
        response = self.chat.invoke([HumanMessage(content=prompt_value.to_string())])
        content = response.content.strip()
        try:
            parsed = parser.parse(content)
            fields = parsed.dict()
        except ValidationError:
            # Fallback: try regex extraction as before
            fields = dict(last_candidate_fields) if last_candidate_fields else {}
            markdown_field_pattern = re.compile(r"(?:^|\n)[\-\d\.\*\s]*\*?\*?([A-Za-z0-9_\s]+?)\*?\*?\s*[:：]\s*([\w\-,.\/()'’\s]+)", re.IGNORECASE)
            for match in markdown_field_pattern.finditer(all_text):
                raw_field, value = match.group(1).strip(), match.group(2).strip()
                for canonical, synonyms in self.FIELD_SYNONYMS.items():
                    for syn in synonyms:
                        if syn.lower() in raw_field.lower():
                            if canonical in ["BEDROOMS", "BATHROOMS", "SIZE", "avg_distance_to_nearest_station", "nearest_station_count"]:
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
                    pattern = rf"(?:{syn})\s*[:=\-]?\s*(\d+\.?\d*|[\w\s,.'’]+)"
                    match = re.search(pattern, all_text, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        if field in ["BEDROOMS", "BATHROOMS", "SIZE", "avg_distance_to_nearest_station", "nearest_station_count"]:
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
        for k in ['BEDROOMS', 'BATHROOMS', 'SIZE', 'avg_distance_to_nearest_station', 'nearest_station_count']:
            if k in encoded:
                try:
                    encoded[k] = float(encoded[k])
                    if encoded[k].is_integer():
                        encoded[k] = int(encoded[k])
                except Exception:
                    encoded[k] = 0
        return encoded

    def run_model(self, fields):
        print("[DEBUG] User fields (extracted from conversation):", fields)
        encoded_fields = self.encode_fields_for_model(fields)
        print("[DEBUG] Encoded fields for model:", encoded_fields)
        # Only keep model fields
        MODEL_FIELDS = [
            "address", "subdistrict_code", "BEDROOMS", "BATHROOMS", "SIZE",
            "PROPERTY TYPE", "avg_distance_to_nearest_station", "nearest_station_count"
        ]
        model_input = {k: encoded_fields[k] for k in MODEL_FIELDS if k in encoded_fields}
        predicted_rent, lower_rent, upper_rent, confidence_percentage = predict_rent(model_input)
        explanation = generate_explanation_with_langchain(model_input, predicted_rent, lower_rent, upper_rent, confidence_percentage)
        return explanation

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
        # Look for a confirmation request in the last assistant message
        confirmation_keywords = [
            "please confirm", "is this information correct", "is this correct", "can you confirm", "are these details correct"
        ]
        return any(kw in last_assistant["content"].lower() for kw in confirmation_keywords)

    def handle(self, conversation_history, user_message, last_candidate_fields=None):
        candidate_fields = self.extract_fields(user_message, conversation_history, last_candidate_fields)
        # Only keep tenant fields
        tenant_fields = {k: v for k, v in (last_candidate_fields or candidate_fields).items() if k in self.required_fields}
        if self.needs_confirmation(user_message) and self.should_run_model(conversation_history, tenant_fields):
            fields = tenant_fields
            if all(f in fields and fields[f] not in (None, '', 0, 0.0) for f in self.required_fields):
                result = self.run_model(fields)
                return {"response": result, "action": "screen_tenant", "fields": fields}
            else:
                missing = [f for f in self.required_fields if f not in fields or fields[f] in (None, '', 0, 0.0)]
                return {"response": f"I need the following details to screen the tenant: {', '.join(missing)}. Please provide them.", "action": "ask_for_info", "fields": fields}
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        response = self.chat.invoke([HumanMessage(content=m["content"]) for m in messages])
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
            "You are LandlordBuddy, an expert and friendly AI assistant for landlords. "
            "You support rent pricing, tenant screening, and maintenance prediction. "
            "For tenant screening, your job is to gather all required info (credit score, income, rent, employment status, eviction record), confirm with the user, and then run the official tenant screening script. "
            "NEVER provide your own evaluation or advice. If the script cannot be run (e.g., missing info), politely tell the user: 'Sorry, I couldn't run the tenant screening script because some required information is missing. Please provide all the details.' "
            "If the script runs, present its output in a user-friendly, markdown-formatted way, and elaborate only to clarify the script's result, not to add your own judgment. "
            "Be concise, professional, and interactive. "
            "Never say 'please wait', 'processing', or imply that you will automatically proceed. Only ask for confirmation and wait for the user's explicit confirmation before running the script."
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
        parser = PydanticOutputParser(pydantic_object=TenantFields)
        all_text = "\n".join([m["content"] for m in conversation_history if m["role"] in ("user", "assistant")])
        all_text += "\n" + user_message
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant for landlords. Extract the following fields from the conversation and user message. If a field is missing, use 0, empty string, or False. Output only the JSON object as specified by the schema: {format_instructions}"),
            ("user", "Conversation so far:\n{conversation}\nUser message:\n{user_message}")
        ])
        format_instructions = parser.get_format_instructions()
        prompt_value = prompt.format_prompt(
            conversation=all_text,
            user_message=user_message,
            format_instructions=format_instructions
        )
        response = self.chat.invoke([HumanMessage(content=prompt_value.to_string())])
        content = response.content.strip()
        try:
            parsed = parser.parse(content)
            fields = parsed.dict()
        except ValidationError:
            # Fallback: regex extraction as before
            fields = dict(last_candidate_fields) if last_candidate_fields else {}
            markdown_field_pattern = re.compile(r"(?:^|\n)[\-\d\.\*\s]*\*?\*?([A-Za-z0-9_\s]+?)\*?\*?\s*[:：]\s*([\w\-,.\/()'’\s]+)", re.IGNORECASE)
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
                    pattern = rf"(?:{syn})\s*[:=\-]?\s*([\w\-,.\/()'’\s]+)"
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
        return fields

    def summarize_fields(self, fields):
        summary = "**Tenant Screening Details:**\n\n"
        for k in self.required_fields:
            v = fields.get(k, "[missing]")
            if k == "eviction_record":
                v = "Yes" if v is True else ("No" if v is False else v)
            summary += f"- **{k.replace('_', ' ').title()}**: {v}\n"
        summary += "\nIs this correct? Please confirm so I can screen the tenant."
        return summary

    def run_model(self, fields):
        from .tenant_screening import screen_tenant
        credit_score = int(fields.get("credit_score", 0) or 0)
        income = float(fields.get("income", 0) or 0)
        rent = float(fields.get("rent", 0) or 0)
        employment_status = str(fields.get("employment_status", "")).strip() or "unknown"
        eviction_record = bool(fields.get("eviction_record", False))
        result = screen_tenant(credit_score, income, rent, employment_status, eviction_record)
        md = f"**Tenant Screening Result:**\n\n"
        md += f"- **Recommendation:** {result['recommendation']}\n"
        md += f"- **Risk Score:** {result['risk_score']}\n"
        md += f"- **Details:**\n"
        for line in result['explanation'].split('\n'):
            md += f"  - {line}\n"
        return md

    def handle(self, conversation_history, user_message, last_candidate_fields=None):
        candidate_fields = self.extract_fields(user_message, conversation_history, last_candidate_fields)
        tenant_fields = {k: v for k, v in (candidate_fields if any(candidate_fields.values()) else (last_candidate_fields or {})).items() if k in self.required_fields}
        if self.needs_confirmation(user_message) and self.should_run_model(conversation_history, tenant_fields):
            fields = tenant_fields
            if all(f in fields and fields[f] not in (None, '', 0, 0.0, False) for f in self.required_fields):
                result = self.run_model(fields)
                return {"response": result, "action": "screen_tenant", "fields": fields}
            else:
                return {"response": "Sorry, I couldn't run the tenant screening script because some required information is missing. Please provide all the details (credit score, income, rent, employment status, and eviction record).", "action": "ask_for_info", "fields": fields}
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        response = self.chat.invoke([HumanMessage(content=m["content"]) for m in messages])
        reply = response.content.strip()
        # Remove any 'please wait' or 'processing' phrases from the reply
        for phrase in ["please wait", "processing", "hold on", "one moment", "I'll process", "wait a moment"]:
            reply = reply.replace(phrase, "").replace(phrase.capitalize(), "")
        extracted = self.extract_fields(reply, conversation_history, candidate_fields)
        return {"response": reply, "action": "chat", "fields": extracted}

    def should_run_model(self, conversation_history, candidate_fields):
        if not candidate_fields or not all(f in candidate_fields and candidate_fields[f] not in (None, '', 0, 0.0, False) for f in self.required_fields):
            return False
        if not conversation_history:
            return False
        last_assistant = next((m for m in reversed(conversation_history) if m["role"] == "assistant"), None)
        if not last_assistant:
            return False
        confirmation_keywords = [
            "please confirm", "is this information correct", "is this correct", "can you confirm", "are these details correct", "please confirm so i can screen the tenant"
        ]
        return any(kw in last_assistant["content"].lower() for kw in confirmation_keywords)

# --- Intent Detection ---
def detect_intent(user_message, conversation_history=None):
    msg = user_message.lower()
    if any(word in msg for word in ["rent", "price", "how much", "estimate"]):
        return "rent_prediction"
    if any(word in msg for word in ["tenant", "screen", "applicant", "background"]):
        return "tenant_screening"
    if any(word in msg for word in ["maintenance", "repair", "fix", "upkeep"]):
        return "maintenance_prediction"
    # Default to rent prediction for now
    return "rent_prediction"

# --- Main Conversational Engine ---
def conversational_engine(conversation_history, user_message, last_candidate_fields=None):
    """
    Modular conversational engine for LandlordBuddy.
    Routes to the correct module handler based on detected intent.
    """
    intent = detect_intent(user_message, conversation_history)
    if intent == "rent_prediction":
        handler = RentPredictionHandler()
    elif intent == "tenant_screening":
        handler = TenantScreeningHandler()
    elif intent == "maintenance_prediction":
        handler = MaintenancePredictionHandler()
    else:
        handler = RentPredictionHandler()
    return handler.handle(conversation_history, user_message, last_candidate_fields)
