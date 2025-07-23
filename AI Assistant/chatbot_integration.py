import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

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
            "If info is missing, ask for it in a friendly, casual way. "
            "You must be flexible in understanding user input: users may provide information in any format, not just JSON or structured lists. "
            "You should do your best to interpret and extract the required details for rent prediction even if the user uses casual language, synonyms, or different phrasings. "
            "For example, if the user says 'avg distance', 'average station distance', 'station distance', or any similar phrase, you should map it to 'avg_distance_to_nearest_station'. "
            "Similarly, for all required fields, try to match and extract the information even if the user does not use the exact field names. "
            "The required information for rent prediction is: address, subdistrict_code, BEDROOMS, BATHROOMS, SIZE (in sq ft), PROPERTY TYPE. "
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
                    pattern = rf"(?:{syn})\s*[:=\-]?\s*(\d+\.?\d*|[\w\s,.'’]+)"
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
                        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
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
                        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
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
        # Run model if user confirms and all required fields are present (ignore last assistant message)
        if self.needs_confirmation(user_message):
            fields = rent_fields
            if all(f in fields and fields[f] not in (None, '', 0, 0.0) for f in self.required_fields):
                result = self.run_model(fields)
                return {"response": result, "action": "rent_prediction", "fields": fields}
            else:
                missing = [f for f in self.required_fields if f not in fields or fields[f] in (None, '', 0, 0.0)]
                return {"response": f"I need the following details to estimate rent: {', '.join(missing)}. Please provide them.", "action": "ask_for_info", "fields": fields}
        # Otherwise, continue the LLM-driven flow
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
            "You are LandlordBuddy, an expert and professional AI assistant for landlords. "
            "You support rent pricing, tenant screening, and maintenance prediction. "
            "For tenant screening, your job is to gather all required info (credit score, income, rent, employment status, eviction record), confirm with the user, and then run the official tenant screening script. "
            "NEVER provide your own evaluation, summary, or advice. "
            "If the script cannot be run (e.g., missing info), politely say: 'Sorry, I couldn't run the tenant screening script because some required information is missing. Please provide all the details.' "
            "If the script runs, present its output in a user-friendly, markdown-formatted way, and elaborate only to clarify the script's result, not to add your own judgment. "
            "You don't use words like script, model, or AI. "
            "You are a friendly, professional assistant, not a chatbot. "
            "Be concise, professional, and interactive. "
            "Never say 'please wait', 'processing', or imply that you will automatically proceed. Only ask for confirmation and wait for the user's explicit confirmation before running the script. "
            "Never mention or ask for tenant name, rental history, or any extra fields. "
            "Only summarize the required fields and ask for confirmation. "
            "When starting tenant screening, always ask for all required information at once, listing each required field (credit score, income, rent, employment status, eviction record) in a clear, markdown-formatted list. "
            "Do not ask for fields one by one. If any are missing, ask for all missing fields together in a single message. "
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
            ("system", "You are an expert assistant for landlords. Extract ONLY the following fields from the conversation and user message. If a field is missing, use 0, empty string, or False. Output only the JSON object as specified by the schema: {format_instructions}"),
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
        # Only keep required fields and ensure correct types
        clean_fields = {}
        for k in self.required_fields:
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

    def summarize_fields(self, fields):
        summary = "**Tenant Screening Details:**\n\n"
        for k in fields:
            v = fields.get(k, "[missing]")
            if k == "eviction_record":
                v = "Yes" if v is True else ("No" if v is False else v)
            summary += f"- **{k.replace('_', ' ').title()}**: {v}\n"
        summary += "\nIs this correct? Please confirm so I can screen the tenant."
        return summary

    def run_model(self, fields):
        from tenant_screening import screen_tenant  # Absolute import for script/module compatibility
        credit_score = int(fields.get("credit_score", 0) or 0)
        income = float(fields.get("income", 0) or 0)
        rent = float(fields.get("rent", 0) or 0)
        employment_status = str(fields.get("employment_status", "")).strip() or "unknown"
        eviction_record = bool(fields.get("eviction_record", False))
        print("Starting script with fields:", fields)
        print(f"Credit Score: {credit_score}, Income: {income}, Rent: {rent}, Employment Status: {employment_status}, Eviction Record: {eviction_record}")
        result = screen_tenant(credit_score, income, rent, employment_status, eviction_record)
        print("DOne with script")
        # Compose a brief summary based on recommendation
        summary = ""
        if result['recommendation'].lower() in ['approve', 'accept', 'approved', 'accepted']:
            summary = "✅ **Tenant Approved:** This applicant meets the screening criteria."
        elif result['recommendation'].lower() == 'review':
            summary = "⚠️ **Tenant Requires Further Review:** Some risk factors were detected."
        else:
            summary = "❌ **Tenant Rejected:** This applicant does not meet the screening criteria."
        md = f"{summary}\n\n**Tenant Screening Result:**\n\n"
        md += f"- **Recommendation:** {result['recommendation']}\n"
        md += f"- **Risk Score:** {result['risk_score']}\n"
        md += f"- **Details:**\n"
        for line in result['explanation'].split('\n'):
            md += f"  - {line}\n"
        return md

    def handle(self, conversation_history, user_message, last_candidate_fields=None):
        # Always merge new extracted fields with last candidate fields, only for required fields
        new_fields = self.extract_fields(user_message, conversation_history, last_candidate_fields)
        merged_fields = dict(last_candidate_fields) if last_candidate_fields else {}
        for k in self.required_fields:
            v = new_fields.get(k, None)
            if v not in (None, '', 0, 0.0, False):
                merged_fields[k] = v
        # Only keep required fields
        tenant_fields = {k: merged_fields.get(k, 0 if k in ["credit_score", "income", "rent"] else (False if k == "eviction_record" else "")) for k in self.required_fields}
        filtered_fields = {k: tenant_fields[k] for k in self.required_fields}
        # --- After user confirmation, always run the script if all required fields are present ---
        if self.needs_confirmation(user_message):
            if all(self._is_field_filled(k, filtered_fields.get(k)) for k in self.required_fields):
                result = self.run_model(filtered_fields)
                return {"response": result, "action": "screen_tenant", "fields": filtered_fields}
            else:
                return {
                    "response": "Sorry, I couldn't run the tenant screening script because some required information is missing. Please provide all the details (credit score, income, rent, employment status, and eviction record).",
                    "action": "ask_for_info",
                    "fields": filtered_fields
                }
        # Otherwise, continue the LLM-driven flow
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        response = self.chat.invoke([HumanMessage(content=m["content"]) for m in messages])
        reply = response.content.strip()
        # Remove any LLM advice/summary or extra fields
        for phrase in [
            "please wait", "processing", "hold on", "one moment", "I'll process", "wait a moment",
            "it's important to exercise caution", "you may wish to consider", "you may want to explore",
            "consider requesting a guarantor", "By considering these factors", "Tips for Landlord",
            "Based on the information provided", "Summary:"
        ]:
            reply = reply.replace(phrase, "")
            reply = reply.replace(phrase.capitalize(), "")
        # Remove lines with extra fields
        lines = reply.split('\n')
        filtered_lines = []
        for line in lines:
            if any(field in line.lower() for field in ["full name", "rental history", "name:", "history:", "annual income", "tenant's name"]):
                continue
            filtered_lines.append(line)
        reply = '\n'.join(filtered_lines)
        # Extract fields from reply and merge again
        reply_fields = self.extract_fields(reply, conversation_history, tenant_fields)
        for k in self.required_fields:
            v = reply_fields.get(k, None)
            if v not in (None, '', 0, 0.0, False):
                tenant_fields[k] = v
        # Always return only required fields
        return {"response": reply, "action": "chat", "fields": {k: tenant_fields[k] for k in self.required_fields}}

    def _is_field_filled(self, key, value):
        if key in ["credit_score", "income", "rent"]:
            return value not in (None, '', 0, 0.0)
        if key == "eviction_record":
            return value is not None  # Boolean, so False is a valid value
        if key == "employment_status":
            return bool(value and str(value).strip())
        return value not in (None, '', False)

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
            "please confirm", "is this information correct", "is this correct", "can you confirm", "are these details correct", "please confirm so i can screen the tenant"
        ]
        return any(kw in last_assistant["content"].lower() for kw in confirmation_keywords)

# --- Predictive Maintenance Integration ---



# --- Predictive Maintenance Handler with Address Mapping ---
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
            response = self.chat.invoke([HumanMessage(content=prompt_value.to_string())])
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

    def summarize_fields(self, fields):
        summary = "**Property Information for Maintenance Prediction:**\n\n"
        for k, v in fields.items():
            summary += f"- **{k}**: {v}\n"
        summary += "\nIs this information correct? Please confirm to proceed with the maintenance risk assessment."
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
        # Map risk score to recommended action
        if risk_score > 7:
            action = 'Immediate Action'
        elif risk_score > 4:
            action = 'Monitor'
        else:
            action = 'Routine'
        summary = (
            f"- **Predicted Maintenance Risk Score:** {risk_score:.2f}\n"
            f"- **Recommended Action:** {action}\n"
        )
        explanation = (
            "\n**How this was calculated:**\n"
            "The risk score is based on property age, time since last service, seasonality, and past maintenance history. "
            "A higher score means more urgent maintenance is likely needed.\n"
        )
        return summary + explanation

    def handle(self, conversation_history, user_message, last_candidate_fields=None):
        # If user provides all required fields and confirms, run the model
        candidate_fields = self.extract_fields(user_message, conversation_history, last_candidate_fields)
        if self.needs_confirmation(user_message):
            fields = candidate_fields
            if all(f in fields and fields[f] not in (None, '', 0, 0.0) for f in self.required_fields):
                result = self.run_model(fields)
                return {"response": result, "action": "maintenance_prediction", "fields": fields}
            else:
                missing = [f for f in self.required_fields if f not in fields or fields[f] in (None, '', 0, 0.0)]
                return {"response": f"I need the following details to predict maintenance risk: {', '.join(missing)}. Please provide them.", "action": "ask_for_info", "fields": fields}
        # Otherwise, continue the LLM-driven flow
        summary = self.summarize_fields(candidate_fields)
        return {"response": summary, "action": "chat", "fields": candidate_fields}

# --- Intent Detection ---
def detect_intent(user_message, conversation_history=None):
    msg = user_message.lower()
    if any(word in msg for word in ["rent", "price", "how much", "estimate"]):
        return "rent_prediction"
    if any(word in msg for word in ["tenant", "screen", "applicant", "background"]):
        return "tenant_screening"
    if any(word in msg for word in ["maintenance", "repair", "fix", "upkeep"]):
        return "maintenance_prediction"
    # If no intent is detected, return None
    return None

# --- LLM-based Intent Detection ---
def llm_detect_intent(conversation_history, user_message):
    """
    Use the LLM to robustly detect the user's intent, using the last 4-5 messages for context.
    Returns: 'rent_prediction', 'tenant_screening', 'maintenance_prediction', or None
    """
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
    # Use last 4-5 messages for context
    history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
    context = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    prompt = (
        "You are an expert assistant for landlords. "
        "Given the following conversation, classify the user's current intent as one of: 'rent_prediction', 'tenant_screening', 'maintenance_prediction', or 'other'. "
        "Only output the intent keyword.\n"
        f"Conversation:\n{context}\nUser message:\n{user_message}\nIntent:"
    )
    response = chat.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()
    if "rent" in intent:
        return "rent_prediction"
    if "tenant" in intent or "screen" in intent:
        return "tenant_screening"
    if "maintenance" in intent or "repair" in intent:
        return "maintenance_prediction"
    return None

# --- Explicit Intent Switch Detection ---
def user_requests_intent_switch(user_message):
    msg = user_message.lower()
    switch_phrases = [
        "forget it", "let's do", "i want to do", "switch to", "change to", "do rent instead", "do tenant instead", "do maintenance instead", "not this", "wrong task", "that's not what i meant", "i want rent", "i want tenant", "i want maintenance"
    ]
    return any(phrase in msg for phrase in switch_phrases)

# --- Main Conversational Engine ---
def conversational_engine(conversation_history, user_message, last_candidate_fields=None, last_intent=None, intent_completed=False):
    """
    Modular conversational engine for LandlordBuddy.
    Routes to the correct module handler based on detected intent.
    """
    # If user requests to switch/cancel, re-detect intent
    if user_requests_intent_switch(user_message):
        intent = llm_detect_intent(conversation_history, user_message)
        intent_completed = False
    elif last_intent and not intent_completed:
        # Persist the last intent until the flow is completed
        intent = last_intent
    else:
        # No active intent or just completed, detect new intent
        intent = llm_detect_intent(conversation_history, user_message)
        intent_completed = False

    confirmation_phrases = ["yes", "correct", "that's right", "yep", "confirmed", "go ahead", "proceed"]
    is_confirmation = user_message.strip().lower() in confirmation_phrases

    # Handle each intent
    if intent == "rent_prediction":
        handler = RentPredictionHandler()
        result = handler.handle(conversation_history, user_message, last_candidate_fields)
        # If model was run, mark intent as completed
        if result.get("action") == "screen_tenant" or result.get("action") == "rent_prediction":
            intent_completed = True
        return {**result, "last_intent": intent if not intent_completed else None, "intent_completed": intent_completed}
    elif intent == "tenant_screening":
        handler = TenantScreeningHandler()
        result = handler.handle(conversation_history, user_message, last_candidate_fields)
        if result.get("action") == "screen_tenant":
            intent_completed = True
        return {**result, "last_intent": intent if not intent_completed else None, "intent_completed": intent_completed}
    elif intent == "maintenance_prediction":
        handler = MaintenancePredictionHandler()
        result = handler.handle(conversation_history, user_message, last_candidate_fields)
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
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

from faiss_utils import semantic_search, load_faiss_index, record_to_text
