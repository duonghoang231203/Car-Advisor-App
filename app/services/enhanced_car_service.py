from typing import Optional, Dict, Any, List
from sqlalchemy.orm import sessionmaker, selectinload
from sqlalchemy import create_engine, select
from app.core.database import get_database_url
from app.db.enhanced_car_models import (
    EnhancedCar, EnhancedCarSpecification, CarFeatures, CarSafety, 
    CarMediaTechnology, CarComfort, CarDimensions
)
# from app.models.enhanced_car import EnhancedCarResponse
from app.core.logging import logger

class EnhancedCarService:
    def __init__(self):
        self.engine = create_engine(get_database_url())
        self.Session = sessionmaker(bind=self.engine)
    
    async def get_enhanced_car_details(self, car_name: str, car_brand: str, car_year: int) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin chi tiết xe nâng cao dựa trên tên, hãng và năm
        """
        try:
            session = self.Session()
            
            # Tìm xe dựa trên brand, model và year
            query = select(EnhancedCar).options(
                selectinload(EnhancedCar.specifications),
                selectinload(EnhancedCar.features),
                selectinload(EnhancedCar.safety),
                selectinload(EnhancedCar.technology),
                selectinload(EnhancedCar.comfort),
                selectinload(EnhancedCar.dimensions)
            ).where(
                EnhancedCar.brand == car_brand,
                EnhancedCar.year == car_year
            )
            
            result = session.execute(query)
            enhanced_car = result.scalars().first()
            
            if not enhanced_car:
                logger.info(f"Enhanced car not found for {car_brand} {car_year}")
                return None
            
            # Chuyển đổi thành dictionary với tất cả thông tin chi tiết
            enhanced_data = self._convert_enhanced_car_to_dict(enhanced_car)
            session.close()
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error getting enhanced car details: {e}")
            return None
    
    def _convert_enhanced_car_to_dict(self, enhanced_car: EnhancedCar) -> Dict[str, Any]:
        """
        Chuyển đổi EnhancedCar object thành dictionary chi tiết
        """
        enhanced_data = {
            "enhanced_id": enhanced_car.id,
            "vehicle_category": enhanced_car.vehicle_category,
            "vehicle_type": enhanced_car.vehicle_type,
            "market_segment": enhanced_car.market_segment,
        }
        
        # Thông số kỹ thuật nâng cao
        if enhanced_car.specifications:
            specs = enhanced_car.specifications
            enhanced_data["enhanced_specifications"] = {
                "engine_type": specs.engine_type,
                "engine_displacement": specs.engine_displacement,
                "engine_torque": specs.engine_torque,
                "acceleration_0_60": specs.acceleration_0_60,
                "top_speed": specs.top_speed,
                "combined_mpg": specs.combined_mpg,
                "fuel_tank_capacity": specs.fuel_tank_capacity,
                "electric_range": specs.electric_range,
                "battery_capacity": specs.battery_capacity,
                "charging_time": specs.charging_time,
                "number_of_gears": specs.number_of_gears,
                "cargo_capacity": specs.cargo_capacity,
                "towing_capacity": specs.towing_capacity
            }
        
        # Tính năng khí hậu và tiện ích - GIẢI QUYẾT VẤN ĐỀ ĐIỀU HÒA
        if enhanced_car.features:
            features = enhanced_car.features
            enhanced_data["climate_comfort_features"] = {
                "has_air_conditioning": features.has_air_conditioning,
                "climate_control_type": features.climate_control_type,
                "heated_seats": features.heated_seats,
                "cooled_seats": features.cooled_seats,
                "heated_steering_wheel": features.heated_steering_wheel,
                "rear_climate_control": features.rear_climate_control,
                "headlight_type": features.headlight_type,
                "fog_lights": features.fog_lights,
                "daytime_running_lights": features.daytime_running_lights,
                "adaptive_headlights": features.adaptive_headlights,
                "power_windows": features.power_windows,
                "sunroof": features.sunroof,
                "moonroof": features.moonroof,
                "panoramic_roof": features.panoramic_roof,
                "convertible_top": features.convertible_top,
                "alloy_wheels": features.alloy_wheels,
                "wheel_size": features.wheel_size,
                "roof_rails": features.roof_rails,
                "tow_hitch": features.tow_hitch
            }
        
        # Tính năng an toàn
        if enhanced_car.safety:
            safety = enhanced_car.safety
            enhanced_data["safety_features"] = {
                "airbag_systems": {
                    "front_airbags": safety.front_airbags,
                    "side_airbags": safety.side_airbags,
                    "curtain_airbags": safety.curtain_airbags,
                    "knee_airbags": safety.knee_airbags,
                    "rear_airbags": safety.rear_airbags
                },
                "safety_ratings": {
                    "nhtsa_overall_rating": safety.nhtsa_overall_rating,
                    "iihs_top_safety_pick": safety.iihs_top_safety_pick
                },
                "active_safety": {
                    "abs_brakes": safety.abs_brakes,
                    "electronic_stability_control": safety.electronic_stability_control,
                    "traction_control": safety.traction_control,
                    "brake_assist": safety.brake_assist
                },
                "driver_assistance": {
                    "forward_collision_warning": safety.forward_collision_warning,
                    "automatic_emergency_braking": safety.automatic_emergency_braking,
                    "blind_spot_monitoring": safety.blind_spot_monitoring,
                    "lane_departure_warning": safety.lane_departure_warning,
                    "lane_keeping_assist": safety.lane_keeping_assist,
                    "adaptive_cruise_control": safety.adaptive_cruise_control,
                    "parking_sensors": safety.parking_sensors,
                    "backup_camera": safety.backup_camera,
                    "surround_view_camera": safety.surround_view_camera
                },
                "security": {
                    "anti_theft_system": safety.anti_theft_system,
                    "remote_start": safety.remote_start,
                    "keyless_entry": safety.keyless_entry,
                    "push_button_start": safety.push_button_start
                }
            }
        
        # Công nghệ và giải trí
        if enhanced_car.technology:
            tech = enhanced_car.technology
            enhanced_data["technology_features"] = {
                "infotainment": {
                    "touchscreen_size": tech.touchscreen_size,
                    "infotainment_system": tech.infotainment_system,
                    "gps_navigation": tech.gps_navigation,
                    "voice_control": tech.voice_control
                },
                "audio_system": {
                    "audio_system_brand": tech.audio_system_brand,
                    "speaker_count": tech.speaker_count,
                    "premium_audio": tech.premium_audio,
                    "satellite_radio": tech.satellite_radio
                },
                "connectivity": {
                    "bluetooth": tech.bluetooth,
                    "wifi_hotspot": tech.wifi_hotspot,
                    "apple_carplay": tech.apple_carplay,
                    "android_auto": tech.android_auto,
                    "usb_ports": tech.usb_ports,
                    "wireless_charging": tech.wireless_charging
                },
                "advanced_tech": {
                    "heads_up_display": tech.heads_up_display,
                    "digital_instrument_cluster": tech.digital_instrument_cluster,
                    "ambient_lighting": tech.ambient_lighting
                }
            }
        
        # Tiện nghi và thoải mái
        if enhanced_car.comfort:
            comfort = enhanced_car.comfort
            enhanced_data["comfort_features"] = {
                "seating": {
                    "seat_material": comfort.seat_material,
                    "power_driver_seat": comfort.power_driver_seat,
                    "power_passenger_seat": comfort.power_passenger_seat,
                    "memory_seats": comfort.memory_seats,
                    "lumbar_support": comfort.lumbar_support,
                    "seat_ventilation": comfort.seat_ventilation
                },
                "interior_convenience": {
                    "power_steering": comfort.power_steering,
                    "cruise_control": comfort.cruise_control,
                    "tilt_steering": comfort.tilt_steering,
                    "telescoping_steering": comfort.telescoping_steering,
                    "leather_steering_wheel": comfort.leather_steering_wheel
                },
                "storage": {
                    "cup_holders": comfort.cup_holders,
                    "storage_compartments": comfort.storage_compartments,
                    "cargo_organizer": comfort.cargo_organizer,
                    "cargo_net": comfort.cargo_net
                },
                "mirrors": {
                    "power_mirrors": comfort.power_mirrors,
                    "heated_mirrors": comfort.heated_mirrors,
                    "auto_dimming_mirrors": comfort.auto_dimming_mirrors
                }
            }
        
        # Kích thước và thông số vật lý
        if enhanced_car.dimensions:
            dims = enhanced_car.dimensions
            enhanced_data["vehicle_dimensions"] = {
                "exterior": {
                    "length": dims.length,
                    "width": dims.width,
                    "height": dims.height,
                    "wheelbase": dims.wheelbase,
                    "ground_clearance": dims.ground_clearance
                },
                "weight": {
                    "curb_weight": dims.curb_weight,
                    "gross_weight": dims.gross_weight
                },
                "interior_space": {
                    "front_headroom": dims.front_headroom,
                    "rear_headroom": dims.rear_headroom,
                    "front_legroom": dims.front_legroom,
                    "rear_legroom": dims.rear_legroom,
                    "front_shoulder_room": dims.front_shoulder_room,
                    "rear_shoulder_room": dims.rear_shoulder_room
                },
                "storage": {
                    "cargo_volume": dims.cargo_volume,
                    "passenger_volume": dims.passenger_volume
                }
            }
        
        # Tính toán điểm số tổng hợp
        enhanced_data["calculated_scores"] = self._calculate_feature_scores(enhanced_car)
        
        # Tạo danh sách tính năng nổi bật
        enhanced_data["key_features"] = self._generate_key_features(enhanced_car)
        
        return enhanced_data
    
    def _calculate_feature_scores(self, enhanced_car: EnhancedCar) -> Dict[str, float]:
        """
        Tính toán điểm số cho an toàn, công nghệ và tiện nghi
        """
        scores = {
            "safety_score": 0.0,
            "technology_score": 0.0,
            "comfort_score": 0.0,
            "overall_score": 0.0
        }
        
        # Tính điểm an toàn
        if enhanced_car.safety:
            safety = enhanced_car.safety
            safety_features = [
                safety.side_airbags, safety.curtain_airbags, safety.abs_brakes,
                safety.electronic_stability_control, safety.forward_collision_warning,
                safety.automatic_emergency_braking, safety.blind_spot_monitoring,
                safety.lane_keeping_assist, safety.adaptive_cruise_control,
                safety.backup_camera
            ]
            safety_score = (sum(safety_features) / len(safety_features)) * 100
            
            # Cộng điểm từ xếp hạng NHTSA nếu có
            if safety.nhtsa_overall_rating:
                safety_score = (safety_score + (safety.nhtsa_overall_rating / 5 * 100)) / 2
            
            scores["safety_score"] = round(safety_score, 1)
        
        # Tính điểm công nghệ
        if enhanced_car.technology:
            tech = enhanced_car.technology
            tech_features = [
                tech.touchscreen_size and tech.touchscreen_size > 0,
                tech.gps_navigation, tech.voice_control, tech.bluetooth,
                tech.apple_carplay, tech.android_auto, tech.premium_audio,
                tech.wifi_hotspot, tech.wireless_charging, tech.heads_up_display
            ]
            scores["technology_score"] = round((sum(tech_features) / len(tech_features)) * 100, 1)
        
        # Tính điểm tiện nghi
        if enhanced_car.comfort:
            comfort = enhanced_car.comfort
            comfort_features = [
                comfort.power_driver_seat, comfort.memory_seats, comfort.lumbar_support,
                comfort.seat_material and comfort.seat_material in ['leather', 'leatherette'],
                comfort.cruise_control, comfort.power_mirrors, comfort.heated_mirrors,
                comfort.auto_dimming_mirrors, comfort.telescoping_steering,
                comfort.leather_steering_wheel
            ]
            scores["comfort_score"] = round((sum(comfort_features) / len(comfort_features)) * 100, 1)
        
        # Tính điểm tổng thể
        scores["overall_score"] = round((scores["safety_score"] + scores["technology_score"] + scores["comfort_score"]) / 3, 1)
        
        return scores
    
    def _generate_key_features(self, enhanced_car: EnhancedCar) -> List[str]:
        """
        Tạo danh sách các tính năng nổi bật của xe
        """
        key_features = []
        
        # Tính năng điều hòa - VẤN ĐỀ CHÍNH CỦA NGƯỜI DÙNG
        if enhanced_car.features and enhanced_car.features.has_air_conditioning:
            if enhanced_car.features.climate_control_type:
                climate_type = enhanced_car.features.climate_control_type.replace('_', ' ').title()
                key_features.append(f"Điều hòa {climate_type}")
            else:
                key_features.append("Điều hòa không khí")
        
        # Tính năng động cơ
        if enhanced_car.specifications and enhanced_car.specifications.engine_hp:
            hp = enhanced_car.specifications.engine_hp
            if hp > 300:
                key_features.append(f"Động cơ mạnh mẽ {hp}HP")
            elif hp > 200:
                key_features.append(f"Động cơ {hp}HP")
        
        # Tiết kiệm nhiên liệu
        if enhanced_car.specifications and enhanced_car.specifications.combined_mpg:
            mpg = enhanced_car.specifications.combined_mpg
            if mpg >= 35:
                key_features.append(f"Tiết kiệm nhiên liệu {mpg} MPG")
        
        # Xe điện/hybrid
        if enhanced_car.specifications:
            if enhanced_car.specifications.engine_type == 'electric':
                if enhanced_car.specifications.electric_range:
                    key_features.append(f"Xe điện - {enhanced_car.specifications.electric_range} dặm")
                else:
                    key_features.append("Xe điện")
            elif enhanced_car.specifications.engine_type == 'hybrid':
                key_features.append("Xe hybrid")
        
        # Tính năng an toàn nổi bật
        if enhanced_car.safety:
            if enhanced_car.safety.automatic_emergency_braking:
                key_features.append("Phanh khẩn cấp tự động")
            if enhanced_car.safety.blind_spot_monitoring:
                key_features.append("Cảnh báo điểm mù")
            if enhanced_car.safety.adaptive_cruise_control:
                key_features.append("Cruise control thích ứng")
        
        # Tính năng công nghệ
        if enhanced_car.technology:
            if enhanced_car.technology.apple_carplay and enhanced_car.technology.android_auto:
                key_features.append("Apple CarPlay & Android Auto")
            if enhanced_car.technology.premium_audio:
                brand = enhanced_car.technology.audio_system_brand or "Premium"
                key_features.append(f"Âm thanh {brand}")
        
        # Tính năng tiện nghi
        if enhanced_car.comfort and enhanced_car.comfort.seat_material == 'leather':
            key_features.append("Ghế da")
        
        # Tính năng nóc
        if enhanced_car.features:
            if enhanced_car.features.panoramic_roof:
                key_features.append("Nóc panoramic")
            elif enhanced_car.features.sunroof:
                key_features.append("Cửa sổ trời")
        
        # Hệ dẫn động
        if enhanced_car.specifications and enhanced_car.specifications.driven_wheels == 'all_wheel_drive':
            key_features.append("Dẫn động 4 bánh")
        
        return key_features[:8]  # Giới hạn 8 tính năng nổi bật
    
    async def search_enhanced_cars(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Tìm kiếm xe với các bộ lọc nâng cao
        """
        try:
            session = self.Session()
            
            query = select(EnhancedCar).options(
                selectinload(EnhancedCar.specifications),
                selectinload(EnhancedCar.features),
                selectinload(EnhancedCar.safety),
                selectinload(EnhancedCar.technology),
                selectinload(EnhancedCar.comfort),
                selectinload(EnhancedCar.dimensions)
            )
            
            # Áp dụng các bộ lọc
            if filters.get('has_air_conditioning'):
                query = query.join(CarFeatures).where(CarFeatures.has_air_conditioning == True)
            
            if filters.get('vehicle_type'):
                query = query.where(EnhancedCar.vehicle_type == filters['vehicle_type'])
            
            if filters.get('market_segment'):
                query = query.where(EnhancedCar.market_segment == filters['market_segment'])
            
            result = session.execute(query)
            enhanced_cars = result.scalars().all()
            
            enhanced_data_list = []
            for car in enhanced_cars:
                enhanced_data = self._convert_enhanced_car_to_dict(car)
                enhanced_data_list.append(enhanced_data)
            
            session.close()
            return enhanced_data_list
            
        except Exception as e:
            logger.error(f"Error searching enhanced cars: {e}")
            return []

# Global instance
enhanced_car_service = EnhancedCarService() 