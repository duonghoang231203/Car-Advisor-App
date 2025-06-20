"""
Script để populate dữ liệu xe với thông tin nâng cao và đa dạng hơn
Bao gồm: điều hòa, các loại xe đa dạng, tính năng an toàn, công nghệ, v.v.
"""

import asyncio
import csv
import logging
import random
from typing import Dict, List, Any
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import mysql
from app.db.models import Car, CarSpecification
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.core.database import get_database_url
from app.db.enhanced_car_models import (
    EnhancedCar, EnhancedCarSpecification, CarDimensions, 
    CarFeatures, CarSafety, CarMediaTechnology, CarComfort
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dữ liệu mẫu cho các tính năng mở rộng
ENHANCED_VEHICLE_TYPES = {
    "passenger": {
        "sedan": ["compact_sedan", "midsize_sedan", "full_size_sedan", "luxury_sedan", "sport_sedan"],
        "suv": ["compact_suv", "midsize_suv", "full_size_suv", "luxury_suv", "sport_suv", "electric_suv"],
        "hatchback": ["subcompact_hatchback", "compact_hatchback", "sport_hatchback"],
        "coupe": ["sport_coupe", "luxury_coupe", "convertible_coupe"],
        "convertible": ["sport_convertible", "luxury_convertible"],
        "wagon": ["compact_wagon", "luxury_wagon", "sport_wagon"],
        "crossover": ["compact_crossover", "midsize_crossover", "luxury_crossover"]
    },
    "commercial": {
        "pickup_truck": ["compact_pickup", "midsize_pickup", "full_size_pickup", "heavy_duty_pickup"],
        "van": ["cargo_van", "passenger_van", "delivery_van"],
        "truck": ["box_truck", "flatbed_truck", "dump_truck", "tow_truck"],
        "bus": ["city_bus", "school_bus", "tour_bus", "shuttle_bus"]
    },
    "specialty": {
        "sports_car": ["roadster", "supercar", "hypercar", "track_car"],
        "electric": ["compact_electric", "luxury_electric", "performance_electric"],
        "hybrid": ["compact_hybrid", "luxury_hybrid", "sport_hybrid"],
        "off_road": ["jeep", "rock_crawler", "desert_runner", "expedition_vehicle"]
    }
}

ENHANCED_FEATURES = {
    "climate_control": {
        "air_conditioning": 0.95,
        "automatic_climate_control": 0.75,
        "dual_zone_climate": 0.45,
        "tri_zone_climate": 0.15,
        "quad_zone_climate": 0.05,
        "heated_seats_front": 0.60,
        "heated_seats_rear": 0.25,
        "cooled_seats_front": 0.30,
        "cooled_seats_rear": 0.10,
        "heated_steering_wheel": 0.35,
        "remote_start": 0.50
    },
    "safety_features": {
        "abs_brakes": 0.98,
        "electronic_stability_control": 0.95,
        "traction_control": 0.95,
        "adaptive_cruise_control": 0.40,
        "lane_departure_warning": 0.55,
        "lane_keeping_assist": 0.45,
        "blind_spot_monitoring": 0.65,
        "backup_camera": 0.85,
        "automatic_emergency_braking": 0.50,
        "forward_collision_warning": 0.60,
        "parking_sensors_rear": 0.70,
        "parking_sensors_front": 0.40
    },
    "technology": {
        "apple_carplay": 0.70,
        "android_auto": 0.70,
        "bluetooth_audio": 0.90,
        "wifi_hotspot": 0.30,
        "wireless_charging": 0.25,
        "usb_ports": 0.95,
        "touchscreen_size": [6.5, 7.0, 8.0, 8.4, 9.0, 10.1, 10.25, 11.6, 12.0, 12.3, 14.0],
        "built_in_navigation": 0.45,
        "voice_recognition": 0.55
    },
    "comfort": {
        "power_windows": 0.95,
        "power_door_locks": 0.95,
        "keyless_entry": 0.80,
        "push_button_start": 0.60,
        "power_seats_driver": 0.70,
        "power_seats_passenger": 0.40,
        "leather_steering_wheel": 0.50,
        "cruise_control": 0.85,
        "sunroof": 0.30,
        "panoramic_sunroof": 0.15
    },
    "lighting": {
        "led_headlights": 0.60,
        "led_taillights": 0.70,
        "led_daytime_running_lights": 0.80,
        "fog_lights": 0.45,
        "adaptive_headlights": 0.25,
        "ambient_lighting": 0.35
    }
}

MARKET_SEGMENTS = {
    "economy": ["affordable", "budget_friendly", "value_oriented", "entry_level"],
    "mainstream": ["popular", "family_oriented", "practical", "reliable"],
    "premium": ["upscale", "near_luxury", "premium_features", "refined"],
    "luxury": ["high_end", "exclusive", "premium_luxury", "ultra_luxury"],
    "sport": ["performance", "sport_oriented", "enthusiast", "high_performance"],
    "commercial": ["work_truck", "fleet", "commercial_grade", "heavy_duty"]
}

ENGINE_TYPES = {
    "gasoline": {
        "4_cylinder": ["2.0L I4", "2.4L I4", "2.5L I4", "1.8L I4 Turbo", "2.0L I4 Turbo"],
        "6_cylinder": ["3.5L V6", "3.6L V6", "2.7L V6 Turbo", "3.0L V6 Turbo"],
        "8_cylinder": ["5.0L V8", "5.7L V8", "6.2L V8", "4.0L V8 Turbo"]
    },
    "hybrid": {
        "mild_hybrid": ["2.0L I4 + Electric", "2.4L I4 + Electric"],
        "full_hybrid": ["2.5L I4 Hybrid", "3.5L V6 Hybrid"],
        "plug_in_hybrid": ["2.0L I4 PHEV", "3.6L V6 PHEV"]
    },
    "electric": {
        "single_motor": ["Single Motor Electric"],
        "dual_motor": ["Dual Motor Electric", "AWD Electric"],
        "tri_motor": ["Tri Motor Electric", "Performance Electric"]
    },
    "diesel": {
        "4_cylinder": ["2.0L I4 Diesel", "2.2L I4 Diesel"],
        "6_cylinder": ["3.0L V6 Diesel", "3.2L V6 Diesel"],
        "8_cylinder": ["6.6L V8 Diesel", "6.7L V8 Diesel"]
    }
}

def generate_enhanced_vehicle_data(car_data: Dict) -> Dict[str, Any]:
    """Tạo dữ liệu xe nâng cao dựa trên thông tin cơ bản"""
    
    # Xác định loại xe và phân khúc
    vehicle_style = car_data.get('vehicle_style', 'sedan').lower()
    market_category = car_data.get('market_category', 'mainstream').lower()
    
    # Tạo phân loại xe
    vehicle_classification = determine_vehicle_classification(vehicle_style, market_category)
    
    # Tạo thông số động cơ nâng cao
    engine_specs = generate_engine_specifications(car_data)
    
    # Tạo tính năng dựa trên phân khúc và năm
    features = generate_features_by_segment_and_year(market_category, car_data.get('year', 2020))
    
    # Tạo thông tin an toàn
    safety_features = generate_safety_features(market_category, car_data.get('year', 2020))
    
    # Tạo kích thước ước tính
    dimensions = estimate_vehicle_dimensions(vehicle_style)
    
    # Tạo điểm số đánh giá
    scores = calculate_vehicle_scores(features, safety_features, engine_specs)
    
    return {
        "vehicle_classification": vehicle_classification,
        "enhanced_engine_specs": engine_specs,
        "features": features,
        "safety": safety_features,
        "dimensions": dimensions,
        "scores": scores,
        "key_features": generate_key_features(features, safety_features, engine_specs)
    }

def determine_vehicle_classification(vehicle_style: str, market_category: str) -> Dict:
    """Xác định phân loại xe chi tiết"""
    
    # Xác định category chính
    if vehicle_style in ['pickup', 'truck', 'van']:
        category = 'commercial'
    elif vehicle_style in ['roadster', 'supercar']:
        category = 'specialty'
    else:
        category = 'passenger'
    
    # Xác định size
    size_mapping = {
        'compact': ['compact', 'subcompact'],
        'midsize': ['midsize', 'medium'],
        'full-size': ['large', 'full', 'full-size'],
        'luxury': ['luxury', 'premium']
    }
    
    size = 'midsize'  # default
    for size_key, keywords in size_mapping.items():
        if any(keyword in market_category.lower() for keyword in keywords):
            size = size_key
            break
    
    # Xác định segment
    segment = 'mainstream'
    if 'luxury' in market_category.lower() or 'premium' in market_category.lower():
        segment = 'luxury'
    elif 'performance' in market_category.lower() or 'sport' in market_category.lower():
        segment = 'sport'
    elif 'economy' in market_category.lower() or 'budget' in market_category.lower():
        segment = 'economy'
    
    return {
        'category': category,
        'type': vehicle_style,
        'size': size,
        'segment': segment,
        'usage': determine_usage_type(category, vehicle_style, segment)
    }

def determine_usage_type(category: str, vehicle_type: str, segment: str) -> str:
    """Xác định mục đích sử dụng xe"""
    if category == 'commercial':
        return 'commercial'
    elif segment == 'sport':
        return 'sport'
    elif vehicle_type in ['suv', 'crossover']:
        return 'family'
    elif vehicle_type in ['coupe', 'convertible']:
        return 'weekend'
    else:
        return 'daily'

def generate_engine_specifications(car_data: Dict) -> Dict:
    """Tạo thông số động cơ chi tiết"""
    fuel_type = car_data.get('engine_fuel_type', 'gasoline').lower()
    hp = car_data.get('engine_hp', 200)
    cylinders = car_data.get('engine_cylinders', 4)
    
    # Ước tính displacement dựa trên HP và cylinders
    displacement = estimate_displacement(hp, cylinders, fuel_type)
    
    # Ước tính torque
    torque = estimate_torque(hp, fuel_type)
    
    # Tạo mô tả động cơ
    engine_description = generate_engine_description(displacement, cylinders, fuel_type, hp)
    
    return {
        'engine_type': engine_description,
        'engine_displacement': displacement,
        'engine_hp': hp,
        'engine_torque': torque,
        'engine_cylinders': cylinders,
        'engine_fuel_type': fuel_type,
        'engine_aspiration': determine_aspiration(hp, displacement),
        'engine_configuration': determine_configuration(cylinders)
    }

def estimate_displacement(hp: int, cylinders: int, fuel_type: str) -> float:
    """Ước tính dung tích động cơ"""
    if fuel_type == 'electric':
        return 0.0
    
    # Công thức ước tính đơn giản
    if cylinders == 4:
        if hp < 150:
            return round(random.uniform(1.6, 2.0), 1)
        elif hp < 250:
            return round(random.uniform(2.0, 2.4), 1)
        else:
            return round(random.uniform(2.0, 2.5), 1)  # Turbo
    elif cylinders == 6:
        if hp < 250:
            return round(random.uniform(2.4, 3.0), 1)
        elif hp < 350:
            return round(random.uniform(3.0, 3.6), 1)
        else:
            return round(random.uniform(3.0, 4.0), 1)
    elif cylinders == 8:
        if hp < 400:
            return round(random.uniform(4.6, 5.7), 1)
        else:
            return round(random.uniform(5.7, 6.8), 1)
    else:
        return round(random.uniform(2.0, 3.0), 1)

def estimate_torque(hp: int, fuel_type: str) -> int:
    """Ước tính mô-men xoắn"""
    if fuel_type == 'electric':
        return int(hp * 0.8)  # Electric motors have high torque
    elif fuel_type == 'diesel':
        return int(hp * 1.3)  # Diesel engines have high torque
    else:
        return int(hp * 0.9)  # Gasoline engines

def determine_aspiration(hp: int, displacement: float) -> str:
    """Xác định loại hút khí"""
    if displacement == 0:
        return 'electric'
    
    # Ước tính dựa trên tỷ lệ HP/displacement
    hp_per_liter = hp / displacement if displacement > 0 else 0
    
    if hp_per_liter > 100:
        return 'turbocharged'
    elif hp_per_liter > 120:
        return 'supercharged'
    else:
        return 'naturally_aspirated'

def determine_configuration(cylinders: int) -> str:
    """Xác định cấu hình động cơ"""
    if cylinders <= 4:
        return 'inline'
    elif cylinders == 6:
        return random.choice(['inline', 'v-type'])
    else:
        return 'v-type'

def generate_engine_description(displacement: float, cylinders: int, fuel_type: str, hp: int) -> str:
    """Tạo mô tả động cơ"""
    if fuel_type == 'electric':
        if hp < 200:
            return 'Single Motor Electric'
        elif hp < 400:
            return 'Dual Motor Electric'
        else:
            return 'Tri Motor Electric'
    
    config = 'I' if cylinders <= 4 or (cylinders == 6 and random.random() < 0.3) else 'V'
    turbo = ' Turbo' if determine_aspiration(hp, displacement) == 'turbocharged' else ''
    
    return f"{displacement}L {config}{cylinders}{turbo}"

def generate_features_by_segment_and_year(market_category: str, year: int) -> Dict:
    """Tạo tính năng dựa trên phân khúc và năm sản xuất"""
    features = {}
    
    # Base probability dựa trên phân khúc
    segment_multiplier = {
        'economy': 0.6,
        'mainstream': 1.0,
        'premium': 1.4,
        'luxury': 1.8,
        'sport': 1.3
    }
    
    segment = 'mainstream'
    for seg in segment_multiplier.keys():
        if seg in market_category.lower():
            segment = seg
            break
    
    multiplier = segment_multiplier[segment]
    
    # Year progression (technology adoption over time)
    year_factor = min(1.0, (year - 2010) / 15)  # 2010 = 0, 2025 = 1
    
    # Generate features for each category
    for category, feature_probs in ENHANCED_FEATURES.items():
        features[category] = {}
        for feature, base_prob in feature_probs.items():
            if feature == 'touchscreen_size':
                # Special handling for touchscreen size
                if random.random() < (0.9 * multiplier * (0.5 + year_factor)):
                    sizes = feature_probs[feature]
                    # Larger screens in newer, more expensive cars
                    if segment in ['luxury', 'premium']:
                        features[category][feature] = random.choice(sizes[-4:])  # Larger sizes
                    else:
                        features[category][feature] = random.choice(sizes[:6])   # Smaller sizes
                else:
                    features[category][feature] = None
            elif feature == 'usb_ports':
                # Special handling for USB ports count
                if random.random() < (base_prob * multiplier):
                    if segment == 'luxury':
                        features[category][feature] = random.randint(4, 8)
                    elif segment in ['premium', 'sport']:
                        features[category][feature] = random.randint(2, 6)
                    else:
                        features[category][feature] = random.randint(1, 4)
                else:
                    features[category][feature] = 1  # At least 1 USB port
            else:
                # Boolean features
                adjusted_prob = min(0.95, base_prob * multiplier * (0.5 + year_factor))
                features[category][feature] = random.random() < adjusted_prob
    
    return features

def generate_safety_features(market_category: str, year: int) -> Dict:
    """Tạo tính năng an toàn"""
    safety = {}
    
    # Safety ratings (1-5 stars)
    if 'luxury' in market_category.lower():
        safety['nhtsa_overall_rating'] = random.randint(4, 5)
    elif 'premium' in market_category.lower():
        safety['nhtsa_overall_rating'] = random.randint(3, 5)
    else:
        safety['nhtsa_overall_rating'] = random.randint(3, 4)
    
    # Generate related ratings
    overall = safety['nhtsa_overall_rating']
    safety['nhtsa_front_crash'] = max(1, overall + random.randint(-1, 1))
    safety['nhtsa_side_crash'] = max(1, overall + random.randint(-1, 1))
    safety['nhtsa_rollover'] = max(1, overall + random.randint(-1, 1))
    
    # IIHS ratings
    iihs_ratings = ['Poor', 'Marginal', 'Acceptable', 'Good', 'Superior']
    base_rating = min(4, overall)  # Convert 5-star to IIHS scale
    safety['iihs_overall_rating'] = iihs_ratings[base_rating]
    
    # Airbag count based on segment
    if 'luxury' in market_category.lower():
        safety['airbags_count'] = random.randint(8, 12)
    elif 'premium' in market_category.lower():
        safety['airbags_count'] = random.randint(6, 10)
    else:
        safety['airbags_count'] = random.randint(4, 8)
    
    # Standard airbag types
    safety['front_airbags'] = True
    safety['side_airbags'] = random.random() < 0.9
    safety['curtain_airbags'] = random.random() < 0.8
    safety['knee_airbags'] = random.random() < 0.3
    
    return safety

def estimate_vehicle_dimensions(vehicle_style: str) -> Dict:
    """Ước tính kích thước xe"""
    # Base dimensions by vehicle type (in inches)
    dimension_ranges = {
        'sedan': {
            'length': (175, 200), 'width': (68, 74), 'height': (55, 60),
            'wheelbase': (105, 115), 'ground_clearance': (5.5, 7.0)
        },
        'suv': {
            'length': (180, 220), 'width': (70, 80), 'height': (65, 75),
            'wheelbase': (108, 125), 'ground_clearance': (7.0, 10.0)
        },
        'hatchback': {
            'length': (160, 180), 'width': (65, 70), 'height': (58, 65),
            'wheelbase': (98, 108), 'ground_clearance': (5.0, 6.5)
        },
        'coupe': {
            'length': (170, 190), 'width': (68, 74), 'height': (52, 58),
            'wheelbase': (100, 110), 'ground_clearance': (4.5, 6.0)
        },
        'pickup': {
            'length': (190, 240), 'width': (72, 82), 'height': (68, 78),
            'wheelbase': (120, 150), 'ground_clearance': (8.0, 12.0)
        }
    }
    
    # Default to sedan if style not found
    ranges = dimension_ranges.get(vehicle_style.lower(), dimension_ranges['sedan'])
    
    dimensions = {}
    for dim, (min_val, max_val) in ranges.items():
        dimensions[dim] = round(random.uniform(min_val, max_val), 1)
    
    # Calculate related dimensions
    if dimensions.get('length') and dimensions.get('width'):
        # Interior dimensions (estimated as percentage of exterior)
        dimensions['front_legroom'] = round(dimensions['length'] * 0.22, 1)
        dimensions['rear_legroom'] = round(dimensions['length'] * 0.18, 1)
        dimensions['front_shoulder_room'] = round(dimensions['width'] * 0.78, 1)
        dimensions['rear_shoulder_room'] = round(dimensions['width'] * 0.75, 1)
        
        # Cargo volume (estimated in cubic feet)
        if vehicle_style.lower() == 'suv':
            dimensions['cargo_volume'] = round(random.uniform(15, 35), 1)
            dimensions['cargo_volume_max'] = round(dimensions['cargo_volume'] * 2.5, 1)
        elif vehicle_style.lower() == 'sedan':
            dimensions['cargo_volume'] = round(random.uniform(12, 18), 1)
        elif vehicle_style.lower() == 'hatchback':
            dimensions['cargo_volume'] = round(random.uniform(18, 25), 1)
            dimensions['cargo_volume_max'] = round(dimensions['cargo_volume'] * 1.8, 1)
    
    return dimensions

def calculate_vehicle_scores(features: Dict, safety: Dict, engine_specs: Dict) -> Dict:
    """Tính điểm số đánh giá xe"""
    scores = {}
    
    # Safety score (0-10)
    safety_score = 0
    if safety.get('nhtsa_overall_rating'):
        safety_score += safety['nhtsa_overall_rating'] * 2  # 2-10 points
    
    # Add points for advanced safety features
    advanced_safety_features = [
        'adaptive_cruise_control', 'lane_keeping_assist', 'automatic_emergency_braking',
        'blind_spot_monitoring', 'surround_view_camera'
    ]
    
    safety_feature_count = 0
    for category in features.values():
        for feature in advanced_safety_features:
            if category.get(feature, False):
                safety_feature_count += 1
    
    safety_score = min(10, safety_score + (safety_feature_count * 0.2))
    scores['safety_score'] = round(safety_score, 1)
    
    # Technology score (0-10)
    tech_score = 0
    tech_features = []
    
    if features.get('technology'):
        tech_data = features['technology']
        if tech_data.get('apple_carplay') or tech_data.get('android_auto'):
            tech_score += 2
        if tech_data.get('touchscreen_size', 0) > 8:
            tech_score += 1.5
        if tech_data.get('wifi_hotspot'):
            tech_score += 1
        if tech_data.get('wireless_charging'):
            tech_score += 0.5
        if tech_data.get('built_in_navigation'):
            tech_score += 1
        if tech_data.get('voice_recognition'):
            tech_score += 1
        
        # Audio system
        if tech_data.get('speakers_count', 0) > 8:
            tech_score += 1
        if tech_data.get('subwoofer'):
            tech_score += 0.5
    
    scores['technology_score'] = round(min(10, tech_score), 1)
    
    # Comfort score (0-10)
    comfort_score = 0
    if features.get('comfort'):
        comfort_data = features['comfort']
        if comfort_data.get('power_seats_driver'):
            comfort_score += 1
        if comfort_data.get('power_seats_passenger'):
            comfort_score += 0.5
        if comfort_data.get('keyless_entry'):
            comfort_score += 0.5
        if comfort_data.get('push_button_start'):
            comfort_score += 0.5
        if comfort_data.get('sunroof') or comfort_data.get('panoramic_sunroof'):
            comfort_score += 1
    
    if features.get('climate_control'):
        climate_data = features['climate_control']
        if climate_data.get('automatic_climate_control'):
            comfort_score += 1
        if climate_data.get('heated_seats_front'):
            comfort_score += 1
        if climate_data.get('cooled_seats_front'):
            comfort_score += 1.5
        if climate_data.get('heated_steering_wheel'):
            comfort_score += 0.5
    
    scores['comfort_score'] = round(min(10, comfort_score), 1)
    
    # Fuel economy rating
    mpg_combined = engine_specs.get('mpg_combined', 25)
    if mpg_combined >= 40:
        scores['fuel_economy_rating'] = 'Excellent'
    elif mpg_combined >= 30:
        scores['fuel_economy_rating'] = 'Good'
    elif mpg_combined >= 25:
        scores['fuel_economy_rating'] = 'Average'
    else:
        scores['fuel_economy_rating'] = 'Below Average'
    
    return scores

def generate_key_features(features: Dict, safety: Dict, engine_specs: Dict) -> List[str]:
    """Tạo danh sách tính năng nổi bật"""
    key_features = []
    
    # Engine highlights
    if engine_specs.get('engine_fuel_type') == 'electric':
        key_features.append('All-Electric Powertrain')
    elif engine_specs.get('engine_fuel_type') == 'hybrid':
        key_features.append('Hybrid Technology')
    elif engine_specs.get('engine_aspiration') == 'turbocharged':
        key_features.append('Turbocharged Engine')
    
    # Safety highlights
    if safety.get('nhtsa_overall_rating', 0) >= 5:
        key_features.append('5-Star Safety Rating')
    
    # Check for advanced safety features
    advanced_safety = []
    for category in features.values():
        if category.get('adaptive_cruise_control'):
            advanced_safety.append('Adaptive Cruise Control')
        if category.get('automatic_emergency_braking'):
            advanced_safety.append('Automatic Emergency Braking')
        if category.get('lane_keeping_assist'):
            advanced_safety.append('Lane Keeping Assist')
    
    if advanced_safety:
        key_features.append('Advanced Driver Assistance')
    
    # Climate features
    if features.get('climate_control', {}).get('air_conditioning'):
        key_features.append('Air Conditioning')
    if features.get('climate_control', {}).get('heated_seats_front'):
        key_features.append('Heated Front Seats')
    
    # Technology features
    tech = features.get('technology', {})
    if tech.get('apple_carplay') and tech.get('android_auto'):
        key_features.append('Apple CarPlay & Android Auto')
    
    if tech.get('touchscreen_size', 0) >= 10:
        key_features.append('Large Touchscreen Display')
    
    # Comfort features
    comfort = features.get('comfort', {})
    if comfort.get('sunroof') or comfort.get('panoramic_sunroof'):
        key_features.append('Sunroof')
    
    if comfort.get('power_seats_driver'):
        key_features.append('Power Driver Seat')
    
    return key_features[:8]  # Limit to 8 key features

async def enhance_existing_cars():
    """Nâng cao dữ liệu xe hiện có"""
    logger.info("Starting to enhance existing car data...")
    
    async for session in mysql.get_session():
        try:
            # Get all cars with their specifications
            result = await session.execute(
                select(Car, CarSpecification)
                .join(CarSpecification, Car.id == CarSpecification.car_id, isouter=True)
            )
            cars_with_specs = result.fetchall()
            
            logger.info(f"Found {len(cars_with_specs)} cars to enhance")
            
            enhanced_count = 0
            for car, spec in cars_with_specs:
                try:
                    # Prepare car data for enhancement
                    car_data = {
                        'vehicle_style': spec.vehicle_style if spec else 'sedan',
                        'market_category': spec.market_category if spec else 'mainstream',
                        'year': car.year,
                        'engine_fuel_type': spec.engine_fuel_type if spec else 'gasoline',
                        'engine_hp': spec.engine_hp if spec else 200,
                        'engine_cylinders': spec.engine_cylinders if spec else 4
                    }
                    
                    # Generate enhanced data
                    enhanced_data = generate_enhanced_vehicle_data(car_data)
                    
                    # Update car with enhanced classification
                    classification = enhanced_data['vehicle_classification']
                    await session.execute(
                        update(Car)
                        .where(Car.id == car.id)
                        .values(
                            vehicle_type=classification['type'],
                            description=f"This is a {classification['segment']} {classification['type']} designed for {classification['usage']} use."
                        )
                    )
                    
                    # Update specifications with enhanced data
                    if spec:
                        enhanced_specs = enhanced_data['enhanced_engine_specs']
                        update_values = {}
                        
                        # Update only if current value is null or default
                        if not spec.engine_hp or spec.engine_hp == 0:
                            update_values['engine_hp'] = enhanced_specs['engine_hp']
                        if not spec.engine_cylinders or spec.engine_cylinders == 0:
                            update_values['engine_cylinders'] = enhanced_specs['engine_cylinders']
                        if not spec.engine_fuel_type:
                            update_values['engine_fuel_type'] = enhanced_specs['engine_fuel_type']
                        if not spec.transmission_type:
                            update_values['transmission_type'] = 'Automatic'  # Default
                        if not spec.driven_wheels:
                            update_values['driven_wheels'] = 'FWD'  # Default
                        if not spec.vehicle_style:
                            update_values['vehicle_style'] = classification['type']
                        if not spec.vehicle_size:
                            update_values['vehicle_size'] = classification['size']
                        if not spec.market_category:
                            update_values['market_category'] = classification['segment']
                        
                        # Add estimated MPG if missing
                        if not spec.city_mpg or spec.city_mpg == 0:
                            if enhanced_specs['engine_fuel_type'] == 'electric':
                                update_values['city_mpg'] = random.randint(100, 130)  # MPGe
                                update_values['highway_mpg'] = random.randint(90, 120)
                            elif enhanced_specs['engine_fuel_type'] == 'hybrid':
                                update_values['city_mpg'] = random.randint(35, 55)
                                update_values['highway_mpg'] = random.randint(30, 45)
                            else:
                                # Regular gasoline
                                if enhanced_specs['engine_cylinders'] == 4:
                                    update_values['city_mpg'] = random.randint(22, 35)
                                    update_values['highway_mpg'] = random.randint(28, 42)
                                elif enhanced_specs['engine_cylinders'] == 6:
                                    update_values['city_mpg'] = random.randint(18, 28)
                                    update_values['highway_mpg'] = random.randint(24, 35)
                                else:  # V8
                                    update_values['city_mpg'] = random.randint(12, 22)
                                    update_values['highway_mpg'] = random.randint(18, 30)
                        
                        if update_values:
                            await session.execute(
                                update(CarSpecification)
                                .where(CarSpecification.id == spec.id)
                                .values(**update_values)
                            )
                    
                    enhanced_count += 1
                    if enhanced_count % 100 == 0:
                        logger.info(f"Enhanced {enhanced_count} cars...")
                        await session.commit()
                
                except Exception as e:
                    logger.error(f"Error enhancing car ID {car.id}: {e}")
                    continue
            
            # Final commit
            await session.commit()
            logger.info(f"Successfully enhanced {enhanced_count} cars")
            
        except Exception as e:
            logger.error(f"Error enhancing cars: {e}")
            await session.rollback()
            raise

async def add_sample_enhanced_cars():
    """Thêm một số xe mẫu với dữ liệu nâng cao"""
    logger.info("Adding sample enhanced cars...")
    
    sample_cars = [
        {
            'name': 'Tesla Model S Plaid',
            'brand': 'Tesla',
            'model': 'Model S',
            'year': 2024,
            'price': 89990,
            'condition': 'new',
            'type': 'buy',
            'vehicle_type': 'sedan',
            'specs': {
                'engine_type': 'Tri Motor Electric',
                'engine_hp': 1020,
                'engine_torque': 1050,
                'engine_fuel_type': 'electric',
                'acceleration_0_60': 1.99,
                'top_speed': 200,
                'electric_range': 396,
                'seating_capacity': 5,
                'driven_wheels': 'AWD'
            },
            'features': {
                'air_conditioning': True,
                'automatic_climate_control': True,
                'tri_zone_climate': True,
                'heated_seats_front': True,
                'cooled_seats_front': True,
                'apple_carplay': False,  # Tesla has its own system
                'touchscreen_size': 17.0,
                'ota_updates': True,
                'autopilot': True
            }
        },
        {
            'name': 'Ford F-150 Lightning',
            'brand': 'Ford',
            'model': 'F-150',
            'year': 2024,
            'price': 54995,
            'condition': 'new',
            'type': 'buy',
            'vehicle_type': 'pickup',
            'specs': {
                'engine_type': 'Dual Motor Electric',
                'engine_hp': 452,
                'engine_torque': 775,
                'engine_fuel_type': 'electric',
                'electric_range': 320,
                'towing_capacity': 10000,
                'payload_capacity': 2000,
                'seating_capacity': 5,
                'driven_wheels': 'AWD'
            },
            'features': {
                'air_conditioning': True,
                'automatic_climate_control': True,
                'dual_zone_climate': True,
                'heated_seats_front': True,
                'apple_carplay': True,
                'android_auto': True,
                'touchscreen_size': 12.0,
                'backup_camera': True,
                'tow_hitch': True
            }
        },
        {
            'name': 'Porsche 911 Turbo S',
            'brand': 'Porsche',
            'model': '911',
            'year': 2024,
            'price': 207000,
            'condition': 'new',
            'type': 'buy',
            'vehicle_type': 'coupe',
            'specs': {
                'engine_type': '3.8L V6 Turbo',
                'engine_displacement': 3.8,
                'engine_hp': 640,
                'engine_torque': 590,
                'engine_cylinders': 6,
                'engine_fuel_type': 'gasoline',
                'engine_aspiration': 'turbocharged',
                'acceleration_0_60': 2.6,
                'top_speed': 205,
                'mpg_city': 18,
                'mpg_highway': 24,
                'seating_capacity': 4,
                'driven_wheels': 'AWD'
            },
            'features': {
                'air_conditioning': True,
                'automatic_climate_control': True,
                'dual_zone_climate': True,
                'heated_seats_front': True,
                'apple_carplay': True,
                'touchscreen_size': 10.9,
                'premium_audio': True,
                'sport_suspension': True
            }
        }
    ]
    
    # TODO: Implement actual insertion logic
    logger.info(f"Would add {len(sample_cars)} sample enhanced cars")

class EnhancedCarDataPopulator:
    def __init__(self, csv_file_path: str = "data/cars_data.csv"):
        self.csv_file_path = csv_file_path
        self.engine = create_engine(get_database_url())
        self.Session = sessionmaker(bind=self.engine)
        
        # Vehicle type mappings for enhanced classification
        self.vehicle_type_mapping = {
            'Sedan': 'sedan',
            'Coupe': 'coupe',
            'Convertible': 'convertible',
            'Wagon': 'wagon',
            'Hatchback': 'hatchback',
            '2dr Hatchback': 'hatchback',
            '4dr Hatchback': 'hatchback',
            'SUV': 'SUV',
            'Pickup': 'pickup',
            'Truck': 'truck',
            'Van': 'van',
            'Minivan': 'minivan',
            'Crossover': 'crossover'
        }
        
        # Market segment classification
        self.market_segments = {
            'economy': ['Hyundai', 'Kia', 'Nissan', 'Mazda', 'Mitsubishi', 'Suzuki'],
            'mainstream': ['Honda', 'Toyota', 'Ford', 'Chevrolet', 'Volkswagen', 'Subaru'],
            'premium': ['Acura', 'Infiniti', 'Lexus', 'Volvo', 'Saab'],
            'luxury': ['BMW', 'Mercedes-Benz', 'Audi', 'Jaguar', 'Land Rover', 'Cadillac'],
            'sport': ['Porsche', 'Ferrari', 'Lamborghini', 'Maserati', 'McLaren', 'Aston Martin']
        }
    
    def determine_vehicle_category(self, make: str, style: str, market_category: str) -> str:
        """Determine vehicle category based on make, style, and market category"""
        commercial_indicators = ['truck', 'van', 'pickup', 'commercial']
        specialty_indicators = ['sports', 'performance', 'exotic', 'luxury']
        
        style_lower = style.lower() if style else ''
        market_lower = market_category.lower() if market_category else ''
        
        if any(indicator in style_lower or indicator in market_lower for indicator in commercial_indicators):
            return 'commercial'
        elif any(indicator in style_lower or indicator in market_lower for indicator in specialty_indicators):
            return 'specialty'
        else:
            return 'passenger'
    
    def determine_market_segment(self, make: str, market_category: str, msrp: float) -> str:
        """Determine market segment based on make, category, and price"""
        for segment, brands in self.market_segments.items():
            if make in brands:
                return segment
        
        # Price-based classification if brand not in predefined segments
        if msrp and msrp > 0:
            if msrp >= 80000:
                return 'luxury'
            elif msrp >= 45000:
                return 'premium'
            elif msrp >= 25000:
                return 'mainstream'
            else:
                return 'economy'
        
        # Market category based classification
        if market_category:
            market_lower = market_category.lower()
            if 'luxury' in market_lower:
                return 'luxury'
            elif 'performance' in market_lower or 'sport' in market_lower:
                return 'sport'
            elif 'premium' in market_lower:
                return 'premium'
        
        return 'mainstream'
    
    def generate_enhanced_features(self, make: str, year: int, vehicle_style: str, market_segment: str) -> dict:
        """Generate comprehensive features based on vehicle characteristics"""
        
        # Base probabilities for features
        base_probs = {
            'has_air_conditioning': 0.95,  # USER'S MAIN CONCERN - 95% of cars have AC
            'power_windows': 0.85,
            'power_steering': 0.98,
            'abs_brakes': 0.95 if year >= 2000 else 0.70,
            'electronic_stability_control': 0.98 if year >= 2012 else 0.60,
            'traction_control': 0.90 if year >= 2000 else 0.40,
        }
        
        # Adjust probabilities based on market segment
        segment_multipliers = {
            'economy': 0.7,
            'mainstream': 1.0,
            'premium': 1.3,
            'luxury': 1.5,
            'sport': 1.4
        }
        
        multiplier = segment_multipliers.get(market_segment, 1.0)
        
        # Year-based feature availability
        year_factor = min(1.5, (year - 1990) / 20)  # Features become more common over time
        
        features = {}
        
        # Climate Control Features
        ac_prob = min(0.99, base_probs['has_air_conditioning'] * multiplier * year_factor)
        features['has_air_conditioning'] = random.random() < ac_prob
        
        if features['has_air_conditioning']:
            climate_types = ['manual', 'automatic', 'dual_zone']
            if market_segment in ['luxury', 'premium']:
                climate_types.extend(['tri_zone', 'quad_zone'])
            features['climate_control_type'] = random.choice(climate_types)
        
        features['heated_seats'] = random.random() < (0.3 * multiplier * year_factor)
        features['cooled_seats'] = random.random() < (0.15 * multiplier * year_factor) if market_segment in ['luxury', 'premium'] else False
        features['heated_steering_wheel'] = random.random() < (0.2 * multiplier * year_factor)
        
        # Lighting Features
        if year >= 2010:
            features['headlight_type'] = random.choice(['halogen', 'HID', 'LED'])
        else:
            features['headlight_type'] = 'halogen'
        
        features['fog_lights'] = random.random() < (0.6 * multiplier)
        features['daytime_running_lights'] = random.random() < (0.4 * year_factor)
        features['adaptive_headlights'] = random.random() < (0.2 * multiplier) if year >= 2005 else False
        
        # Windows and Roof
        features['power_windows'] = random.random() < (base_probs['power_windows'] * multiplier * year_factor)
        features['sunroof'] = random.random() < (0.3 * multiplier)
        features['panoramic_roof'] = random.random() < (0.15 * multiplier) if year >= 2005 else False
        features['convertible_top'] = vehicle_style == 'Convertible'
        
        # Exterior Features
        features['alloy_wheels'] = random.random() < (0.7 * multiplier)
        features['wheel_size'] = random.choice([16, 17, 18, 19, 20]) if features['alloy_wheels'] else random.choice([15, 16])
        features['roof_rails'] = random.random() < 0.4 if vehicle_style in ['SUV', 'Wagon'] else False
        features['tow_hitch'] = random.random() < 0.3 if vehicle_style in ['SUV', 'Pickup', 'Truck'] else False
        
        return features
    
    def generate_safety_features(self, year: int, market_segment: str) -> dict:
        """Generate comprehensive safety features"""
        safety = {}
        
        # Airbag systems
        safety['front_airbags'] = True  # Standard since 1998
        safety['side_airbags'] = random.random() < (0.8 if year >= 2000 else 0.3)
        safety['curtain_airbags'] = random.random() < (0.7 if year >= 2003 else 0.1)
        safety['knee_airbags'] = random.random() < (0.3 if year >= 2010 else 0)
        
        # Safety ratings
        if random.random() < 0.7:  # 70% have NHTSA ratings
            safety['nhtsa_overall_rating'] = round(random.uniform(3.0, 5.0), 1)
        
        safety['iihs_top_safety_pick'] = random.random() < 0.2 if year >= 2010 else False
        
        # Active safety
        safety['abs_brakes'] = True if year >= 2000 else random.random() < 0.7
        safety['electronic_stability_control'] = True if year >= 2012 else random.random() < 0.6
        safety['traction_control'] = random.random() < (0.9 if year >= 2000 else 0.4)
        safety['brake_assist'] = random.random() < (0.5 if year >= 2005 else 0.1)
        
        # Advanced driver assistance (more common in newer, premium vehicles)
        multiplier = 1.5 if market_segment in ['luxury', 'premium'] else 1.0
        year_factor = max(0, (year - 2010) / 10)
        
        safety['forward_collision_warning'] = random.random() < (0.3 * multiplier * year_factor)
        safety['automatic_emergency_braking'] = random.random() < (0.25 * multiplier * year_factor)
        safety['blind_spot_monitoring'] = random.random() < (0.4 * multiplier * year_factor)
        safety['lane_departure_warning'] = random.random() < (0.3 * multiplier * year_factor)
        safety['lane_keeping_assist'] = random.random() < (0.2 * multiplier * year_factor)
        safety['adaptive_cruise_control'] = random.random() < (0.3 * multiplier * year_factor)
        safety['parking_sensors'] = random.random() < (0.4 * multiplier)
        safety['backup_camera'] = True if year >= 2018 else random.random() < (0.5 * year_factor)
        
        # Security features
        safety['anti_theft_system'] = random.random() < 0.8
        safety['remote_start'] = random.random() < (0.3 * multiplier)
        safety['keyless_entry'] = random.random() < (0.7 * multiplier)
        safety['push_button_start'] = random.random() < (0.4 * multiplier * year_factor)
        
        return safety
    
    def generate_technology_features(self, year: int, market_segment: str) -> dict:
        """Generate technology and media features"""
        tech = {}
        
        multiplier = 1.5 if market_segment in ['luxury', 'premium'] else 1.0
        year_factor = max(0, (year - 2005) / 15)
        
        # Infotainment
        if year >= 2010:
            tech['touchscreen_size'] = random.choice([6.5, 7.0, 8.0, 8.4, 9.0, 10.1, 12.0])
            tech['infotainment_system'] = f"{random.choice(['Basic', 'Advanced', 'Premium'])} Infotainment"
        
        tech['gps_navigation'] = random.random() < (0.5 * multiplier * year_factor)
        tech['voice_control'] = random.random() < (0.4 * multiplier * year_factor)
        
        # Audio system
        if random.random() < (0.3 * multiplier):
            tech['audio_system_brand'] = random.choice(['Bose', 'Harman Kardon', 'Bang & Olufsen', 'JBL', 'Alpine'])
        
        tech['speaker_count'] = random.choice([4, 6, 8, 10, 12, 16])
        tech['premium_audio'] = random.random() < (0.3 * multiplier)
        tech['satellite_radio'] = random.random() < (0.6 * year_factor)
        
        # Connectivity
        tech['bluetooth'] = random.random() < (0.8 * year_factor) if year >= 2005 else False
        tech['wifi_hotspot'] = random.random() < (0.3 * year_factor) if year >= 2012 else False
        tech['apple_carplay'] = random.random() < (0.6 * year_factor) if year >= 2014 else False
        tech['android_auto'] = random.random() < (0.5 * year_factor) if year >= 2015 else False
        tech['usb_ports'] = random.choice([0, 1, 2, 3, 4]) if year >= 2008 else 0
        tech['wireless_charging'] = random.random() < (0.2 * multiplier * year_factor) if year >= 2015 else False
        
        # Advanced tech features
        tech['heads_up_display'] = random.random() < (0.15 * multiplier) if year >= 2010 else False
        tech['digital_instrument_cluster'] = random.random() < (0.3 * multiplier * year_factor) if year >= 2012 else False
        tech['ambient_lighting'] = random.random() < (0.25 * multiplier)
        
        return tech
    
    def generate_comfort_features(self, market_segment: str, year: int) -> dict:
        """Generate comfort and convenience features"""
        comfort = {}
        
        multiplier = 1.5 if market_segment in ['luxury', 'premium'] else 1.0
        year_factor = max(0.5, (year - 1990) / 20)
        
        # Seating
        if market_segment in ['luxury', 'premium']:
            comfort['seat_material'] = random.choice(['leather', 'leatherette', 'alcantara'])
        else:
            comfort['seat_material'] = random.choice(['cloth', 'vinyl', 'leatherette'])
        
        comfort['power_driver_seat'] = random.random() < (0.6 * multiplier * year_factor)
        comfort['power_passenger_seat'] = random.random() < (0.4 * multiplier * year_factor)
        comfort['memory_seats'] = random.random() < (0.3 * multiplier) if comfort['power_driver_seat'] else False
        comfort['lumbar_support'] = random.random() < (0.5 * multiplier)
        comfort['seat_ventilation'] = random.random() < (0.2 * multiplier) if market_segment in ['luxury', 'premium'] else False
        
        # Interior convenience
        comfort['power_steering'] = True if year >= 1995 else random.random() < 0.8
        comfort['cruise_control'] = random.random() < (0.7 * multiplier * year_factor)
        comfort['tilt_steering'] = random.random() < (0.8 * year_factor)
        comfort['telescoping_steering'] = random.random() < (0.6 * multiplier * year_factor)
        comfort['leather_steering_wheel'] = random.random() < (0.4 * multiplier)
        
        # Storage
        comfort['cup_holders'] = random.choice([2, 4, 6, 8])
        comfort['storage_compartments'] = random.choice([3, 4, 5, 6, 8])
        comfort['cargo_organizer'] = random.random() < (0.3 * multiplier)
        comfort['cargo_net'] = random.random() < (0.4 * multiplier)
        
        # Mirrors
        comfort['power_mirrors'] = random.random() < (0.8 * multiplier * year_factor)
        comfort['heated_mirrors'] = random.random() < (0.4 * multiplier) if year >= 2000 else False
        comfort['auto_dimming_mirrors'] = random.random() < (0.3 * multiplier)
        
        return comfort
    
    def generate_dimensions(self, vehicle_style: str) -> dict:
        """Generate realistic vehicle dimensions based on style"""
        dimensions = {}
        
        # Base dimensions by vehicle type (inches and pounds)
        dimension_ranges = {
            'Sedan': {'length': (180, 200), 'width': (70, 75), 'height': (55, 60), 'weight': (2800, 4000)},
            'Coupe': {'length': (175, 190), 'width': (70, 74), 'height': (52, 58), 'weight': (2600, 3800)},
            'SUV': {'length': (185, 210), 'width': (72, 80), 'height': (65, 75), 'weight': (3500, 6000)},
            'Truck': {'length': (210, 250), 'width': (75, 85), 'height': (70, 80), 'weight': (4000, 8000)},
            'Pickup': {'length': (200, 240), 'width': (72, 82), 'height': (68, 78), 'weight': (3800, 7000)},
            'Convertible': {'length': (175, 190), 'width': (70, 74), 'height': (52, 58), 'weight': (2800, 4200)},
            'Wagon': {'length': (185, 200), 'width': (70, 75), 'height': (58, 65), 'weight': (3000, 4200)},
            'Hatchback': {'length': (160, 180), 'width': (68, 72), 'height': (58, 65), 'weight': (2400, 3200)},
        }
        
        ranges = dimension_ranges.get(vehicle_style, dimension_ranges['Sedan'])
        
        # Exterior dimensions
        dimensions['length'] = round(random.uniform(*ranges['length']), 1)
        dimensions['width'] = round(random.uniform(*ranges['width']), 1)
        dimensions['height'] = round(random.uniform(*ranges['height']), 1)
        dimensions['wheelbase'] = round(dimensions['length'] * random.uniform(0.55, 0.65), 1)
        dimensions['ground_clearance'] = round(random.uniform(5.5, 9.5), 1)
        
        # Weight
        dimensions['curb_weight'] = random.randint(*ranges['weight'])
        dimensions['gross_weight'] = dimensions['curb_weight'] + random.randint(800, 1500)
        
        # Interior dimensions
        dimensions['front_headroom'] = round(random.uniform(38, 42), 1)
        dimensions['rear_headroom'] = round(random.uniform(36, 40), 1)
        dimensions['front_legroom'] = round(random.uniform(40, 45), 1)
        dimensions['rear_legroom'] = round(random.uniform(32, 40), 1)
        dimensions['front_shoulder_room'] = round(random.uniform(55, 60), 1)
        dimensions['rear_shoulder_room'] = round(random.uniform(53, 58), 1)
        
        # Storage
        if vehicle_style in ['Sedan', 'Coupe']:
            dimensions['cargo_volume'] = round(random.uniform(12, 18), 1)
        elif vehicle_style in ['SUV', 'Wagon']:
            dimensions['cargo_volume'] = round(random.uniform(25, 85), 1)
        elif vehicle_style in ['Truck', 'Pickup']:
            dimensions['cargo_volume'] = round(random.uniform(45, 65), 1)
        else:
            dimensions['cargo_volume'] = round(random.uniform(10, 25), 1)
        
        dimensions['passenger_volume'] = round(random.uniform(90, 120), 1)
        
        return dimensions
    
    def calculate_scores(self, safety_features: dict, tech_features: dict, comfort_features: dict) -> dict:
        """Calculate composite scores for safety, technology, and comfort"""
        scores = {}
        
        # Safety score (0-100)
        safety_items = [
            safety_features.get('side_airbags', False),
            safety_features.get('curtain_airbags', False),
            safety_features.get('abs_brakes', False),
            safety_features.get('electronic_stability_control', False),
            safety_features.get('forward_collision_warning', False),
            safety_features.get('automatic_emergency_braking', False),
            safety_features.get('blind_spot_monitoring', False),
            safety_features.get('lane_keeping_assist', False),
            safety_features.get('adaptive_cruise_control', False),
            safety_features.get('backup_camera', False)
        ]
        safety_score = (sum(safety_items) / len(safety_items)) * 100
        if safety_features.get('nhtsa_overall_rating'):
            safety_score = (safety_score + (safety_features['nhtsa_overall_rating'] / 5 * 100)) / 2
        scores['safety_score'] = round(safety_score, 1)
        
        # Technology score (0-100)
        tech_items = [
            tech_features.get('touchscreen_size', 0) > 0,
            tech_features.get('gps_navigation', False),
            tech_features.get('voice_control', False),
            tech_features.get('bluetooth', False),
            tech_features.get('apple_carplay', False),
            tech_features.get('android_auto', False),
            tech_features.get('premium_audio', False),
            tech_features.get('wifi_hotspot', False),
            tech_features.get('wireless_charging', False),
            tech_features.get('heads_up_display', False)
        ]
        scores['technology_score'] = round((sum(tech_items) / len(tech_items)) * 100, 1)
        
        # Comfort score (0-100)
        comfort_items = [
            comfort_features.get('power_driver_seat', False),
            comfort_features.get('memory_seats', False),
            comfort_features.get('lumbar_support', False),
            comfort_features.get('seat_material') in ['leather', 'leatherette'],
            comfort_features.get('cruise_control', False),
            comfort_features.get('power_mirrors', False),
            comfort_features.get('heated_mirrors', False),
            comfort_features.get('auto_dimming_mirrors', False),
            comfort_features.get('telescoping_steering', False),
            comfort_features.get('leather_steering_wheel', False)
        ]
        scores['comfort_score'] = round((sum(comfort_items) / len(comfort_items)) * 100, 1)
        
        return scores
    
    def generate_key_features(self, car_data: dict, features: dict, safety: dict, tech: dict, comfort: dict) -> list:
        """Generate a list of key features for the vehicle"""
        key_features = []
        
        # Air conditioning (user's main concern)
        if features.get('has_air_conditioning'):
            if features.get('climate_control_type') in ['dual_zone', 'tri_zone', 'quad_zone']:
                key_features.append(f"{features['climate_control_type'].replace('_', '-').title()} Climate Control")
            else:
                key_features.append("Air Conditioning")
        
        # Engine highlights
        if car_data.get('engine_hp') and car_data['engine_hp'] > 300:
            key_features.append(f"{car_data['engine_hp']}HP High Performance Engine")
        
        # Fuel efficiency
        if car_data.get('highway_mpg') and car_data['highway_mpg'] >= 35:
            key_features.append(f"Excellent Fuel Economy ({car_data['highway_mpg']} MPG Highway)")
        
        # Safety features
        if safety.get('automatic_emergency_braking'):
            key_features.append("Automatic Emergency Braking")
        if safety.get('blind_spot_monitoring'):
            key_features.append("Blind Spot Monitoring")
        
        # Technology features
        if tech.get('apple_carplay') and tech.get('android_auto'):
            key_features.append("Apple CarPlay & Android Auto")
        elif tech.get('apple_carplay'):
            key_features.append("Apple CarPlay")
        elif tech.get('android_auto'):
            key_features.append("Android Auto")
        
        if tech.get('premium_audio'):
            brand = tech.get('audio_system_brand', 'Premium')
            key_features.append(f"{brand} Audio System")
        
        # Comfort features
        if comfort.get('seat_material') == 'leather':
            key_features.append("Leather Seats")
        if comfort.get('memory_seats'):
            key_features.append("Memory Seats")
        
        # Roof features
        if features.get('panoramic_roof'):
            key_features.append("Panoramic Roof")
        elif features.get('sunroof'):
            key_features.append("Sunroof")
        
        # Drive type
        if car_data.get('driven_wheels') == 'all wheel drive':
            key_features.append("All-Wheel Drive")
        
        return key_features[:8]  # Limit to top 8 features
    
    def populate_enhanced_data(self, limit: int = None):
        """Main method to populate enhanced car data"""
        logger.info("Loading CSV data...")
        df = pd.read_csv(self.csv_file_path)
        
        if limit:
            df = df.head(limit)
        
        logger.info(f"Processing {len(df)} records...")
        
        session = self.Session()
        try:
            for index, row in df.iterrows():
                try:
                    # Determine vehicle classification
                    vehicle_style = row.get('Vehicle Style', 'Sedan')
                    enhanced_vehicle_type = self.vehicle_type_mapping.get(vehicle_style, 'sedan')
                    vehicle_category = self.determine_vehicle_category(
                        row['Make'], vehicle_style, row.get('Market Category', '')
                    )
                    market_segment = self.determine_market_segment(
                        row['Make'], row.get('Market Category', ''), row.get('MSRP', 0)
                    )
                    
                    # Create enhanced car record
                    car = EnhancedCar(
                        name=f"{row['Make']} {row['Model']}",
                        brand=row['Make'],
                        model=row['Model'],
                        year=int(row['Year']),
                        price=float(row.get('MSRP', 25000)),
                        condition='used' if row['Year'] < 2020 else 'new',
                        type='buy',
                        description=f"{row['Year']} {row['Make']} {row['Model']} - {vehicle_style}",
                        vehicle_category=vehicle_category,
                        vehicle_type=enhanced_vehicle_type,
                        market_segment=market_segment
                    )
                    
                    session.add(car)
                    session.flush()  # Get the car ID
                    
                    # Create specifications
                    specs = EnhancedCarSpecification(
                        car_id=car.id,
                        engine=f"{row.get('Engine HP', 'Unknown')}HP {row.get('Engine Cylinders', '')}Cyl",
                        engine_type='gasoline',  # Default, could be enhanced
                        engine_fuel_type=row.get('Engine Fuel Type', 'regular unleaded'),
                        engine_hp=int(row.get('Engine HP', 200)),
                        engine_cylinders=int(row.get('Engine Cylinders', 4)),
                        transmission=row.get('Transmission Type', 'Automatic'),
                        transmission_type=row.get('Transmission Type', 'automatic').lower(),
                        driven_wheels=row.get('Driven_Wheels', 'front wheel drive'),
                        number_of_doors=int(row.get('Number of Doors', 4)),
                        seating_capacity=random.choice([2, 4, 5, 7, 8]) if vehicle_style == 'SUV' else 5,
                        fuel_type=row.get('Engine Fuel Type', 'regular unleaded'),
                        city_mpg=int(row.get('city mpg', 25)),
                        highway_mpg=float(row.get('highway MPG', 30)),
                        combined_mpg=round((int(row.get('city mpg', 25)) + float(row.get('highway MPG', 30))) / 2, 1),
                        market_category=row.get('Market Category', ''),
                        vehicle_size=row.get('Vehicle Size', 'Midsize'),
                        vehicle_style=vehicle_style,
                        popularity=int(row.get('Popularity', 1000)),
                        msrp=float(row.get('MSRP', 25000))
                    )
                    session.add(specs)
                    
                    # Generate enhanced features
                    features_data = self.generate_enhanced_features(
                        row['Make'], int(row['Year']), vehicle_style, market_segment
                    )
                    features = CarFeatures(car_id=car.id, **features_data)
                    session.add(features)
                    
                    # Generate safety features
                    safety_data = self.generate_safety_features(int(row['Year']), market_segment)
                    safety = CarSafety(car_id=car.id, **safety_data)
                    session.add(safety)
                    
                    # Generate technology features
                    tech_data = self.generate_technology_features(int(row['Year']), market_segment)
                    technology = CarMediaTechnology(car_id=car.id, **tech_data)
                    session.add(technology)
                    
                    # Generate comfort features
                    comfort_data = self.generate_comfort_features(market_segment, int(row['Year']))
                    comfort = CarComfort(car_id=car.id, **comfort_data)
                    session.add(comfort)
                    
                    # Generate dimensions
                    dimensions_data = self.generate_dimensions(vehicle_style)
                    dimensions = CarDimensions(car_id=car.id, **dimensions_data)
                    session.add(dimensions)
                    
                    # Calculate scores and key features (would be added to car record if we had those fields)
                    scores = self.calculate_scores(safety_data, tech_data, comfort_data)
                    key_features = self.generate_key_features(row.to_dict(), features_data, safety_data, tech_data, comfort_data)
                    
                    if index % 100 == 0:
                        logger.info(f"Processed {index} records...")
                        session.commit()
                
                except Exception as e:
                    logger.error(f"Error processing row {index}: {e}")
                    continue
            
            session.commit()
            logger.info("Enhanced data population completed successfully!")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error during data population: {e}")
            raise
        finally:
            session.close()

if __name__ == "__main__":
    populator = EnhancedCarDataPopulator()
    # Start with a smaller subset for testing
    populator.populate_enhanced_data(limit=1000)  # Process first 1000 records 