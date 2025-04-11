import base64
import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from bson import ObjectId

class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles ObjectId and datetime
    """
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

def to_json(data: Any) -> str:
    """
    Convert data to JSON string
    """
    return json.dumps(data, cls=JSONEncoder)

def from_json(json_str: str) -> Any:
    """
    Convert JSON string to data
    """
    return json.loads(json_str)

def is_valid_base64(s: str) -> bool:
    """
    Check if a string is valid base64
    """
    try:
        if not s:
            return False
        # Add padding if necessary
        s = s + '=' * (4 - len(s) % 4) if len(s) % 4 else s
        return base64.b64encode(base64.b64decode(s)).decode() == s
    except Exception:
        return False

def extract_car_features_from_text(text: str) -> Dict[str, Any]:
    """
    Extract car features from text using regex patterns
    """
    features = {}
    
    # Extract price
    price_match = re.search(r'(\d+(?:[\.,]\d+)?)\s*(?:triệu|tỷ|đồng|VND)', text, re.IGNORECASE)
    if price_match:
        price_str = price_match.group(1).replace(',', '.')
        price = float(price_str)
        # Convert to VND if in millions or billions
        if 'tỷ' in price_match.group(0).lower():
            price *= 1_000_000_000
        elif 'triệu' in price_match.group(0).lower():
            price *= 1_000_000
        features['price'] = price
    
    # Extract brand
    brands = ['Toyota', 'Honda', 'Ford', 'Mazda', 'Kia', 'Hyundai', 'BMW', 'Mercedes', 'Audi', 'Lexus']
    for brand in brands:
        if re.search(r'\b' + re.escape(brand) + r'\b', text, re.IGNORECASE):
            features['brand'] = brand
            break
    
    # Extract body type
    body_types = {
        'sedan': ['sedan'],
        'suv': ['suv', 'crossover'],
        'hatchback': ['hatchback', 'hatchbag'],
        'mpv': ['mpv', 'minivan'],
        'pickup': ['pickup', 'bán tải']
    }
    
    for body_type, keywords in body_types.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                features['body_type'] = body_type
                break
        if 'body_type' in features:
            break
    
    # Extract seating capacity
    seating_match = re.search(r'(\d+)\s*(?:chỗ|người|ghế|seats)', text, re.IGNORECASE)
    if seating_match:
        features['seating_capacity'] = int(seating_match.group(1))
    
    return features

def get_user_id_from_token(token: str) -> str:
    """
    Extract user_id from token
    In a real implementation, you would decode the JWT token
    """
    # This is a placeholder - in a real app, you'd decode the JWT
    return "user123" 