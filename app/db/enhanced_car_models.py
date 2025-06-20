from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey, Numeric, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

class EnhancedCar(Base):
    __tablename__ = "enhanced_cars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Basic Information
    name = Column(String(255), nullable=False)
    brand = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False)
    year = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    condition = Column("car_condition", String(50), nullable=False)  # new or used
    type = Column("car_type", String(50), nullable=False)  # buy or rent
    description = Column(Text, nullable=True)
    
    # Enhanced vehicle classification
    vehicle_category = Column(String(50), nullable=True)  # passenger, commercial, specialty
    vehicle_type = Column(String(50), nullable=True)  # sedan, SUV, truck, sports car, etc.
    market_segment = Column(String(50), nullable=True)  # economy, mainstream, premium, luxury, sport
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    specifications = relationship("EnhancedCarSpecification", back_populates="car", uselist=False)
    features = relationship("CarFeatures", back_populates="car", uselist=False)
    dimensions = relationship("CarDimensions", back_populates="car", uselist=False)
    safety = relationship("CarSafety", back_populates="car", uselist=False)
    technology = relationship("CarMediaTechnology", back_populates="car", uselist=False)
    comfort = relationship("CarComfort", back_populates="car", uselist=False)

class EnhancedCarSpecification(Base):
    __tablename__ = "enhanced_car_specifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey("enhanced_cars.id"), nullable=False)
    
    # Engine specifications
    engine = Column(String(100), nullable=True)
    engine_type = Column(String(50), nullable=True)  # gasoline, diesel, electric, hybrid
    engine_fuel_type = Column(String(50), nullable=True)
    engine_hp = Column(Integer, nullable=True)
    engine_torque = Column(Integer, nullable=True)  # lb-ft
    engine_cylinders = Column(Integer, nullable=True)
    engine_displacement = Column(Float, nullable=True)  # liters
    
    # Performance
    acceleration_0_60 = Column(Float, nullable=True)  # seconds
    top_speed = Column(Integer, nullable=True)  # mph
    
    # Fuel economy
    fuel_type = Column(String(50), nullable=True)
    city_mpg = Column(Integer, nullable=True)
    highway_mpg = Column(Float, nullable=True)
    combined_mpg = Column(Float, nullable=True)
    fuel_tank_capacity = Column(Float, nullable=True)  # gallons
    
    # Electric/Hybrid specific
    electric_range = Column(Integer, nullable=True)  # miles for electric/hybrid
    battery_capacity = Column(Float, nullable=True)  # kWh
    charging_time = Column(String(100), nullable=True)
    
    # Drivetrain
    transmission = Column(String(100), nullable=True)
    transmission_type = Column(String(50), nullable=True)
    driven_wheels = Column(String(50), nullable=True)
    number_of_gears = Column(Integer, nullable=True)
    
    # Basic vehicle info
    number_of_doors = Column(Integer, nullable=True)
    seating_capacity = Column(Integer, nullable=True)
    cargo_capacity = Column(Float, nullable=True)  # cubic feet
    towing_capacity = Column(Integer, nullable=True)  # lbs
    
    # Market data
    market_category = Column(String(100), nullable=True)
    vehicle_size = Column(String(50), nullable=True)
    vehicle_style = Column(String(50), nullable=True)
    popularity = Column(Integer, nullable=True)
    msrp = Column(Float, nullable=True)
    
    # Relationships
    car = relationship("EnhancedCar", back_populates="specifications")

class CarDimensions(Base):
    __tablename__ = "car_dimensions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey("enhanced_cars.id"), nullable=False)
    
    # Exterior dimensions (inches)
    length = Column(Float, nullable=True)
    width = Column(Float, nullable=True)
    height = Column(Float, nullable=True)
    wheelbase = Column(Float, nullable=True)
    ground_clearance = Column(Float, nullable=True)
    
    # Weight (lbs)
    curb_weight = Column(Integer, nullable=True)
    gross_weight = Column(Integer, nullable=True)
    
    # Interior dimensions (inches/cubic feet)
    front_headroom = Column(Float, nullable=True)
    rear_headroom = Column(Float, nullable=True)
    front_legroom = Column(Float, nullable=True)
    rear_legroom = Column(Float, nullable=True)
    front_shoulder_room = Column(Float, nullable=True)
    rear_shoulder_room = Column(Float, nullable=True)
    
    # Storage
    cargo_volume = Column(Float, nullable=True)
    passenger_volume = Column(Float, nullable=True)
    
    # Relationships
    car = relationship("EnhancedCar", back_populates="dimensions")

class CarFeatures(Base):
    __tablename__ = "car_features"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey("enhanced_cars.id"), nullable=False)
    
    # Climate Control - ADDRESSING USER'S MAIN CONCERN
    has_air_conditioning = Column(Boolean, default=False)
    climate_control_type = Column(String(50), nullable=True)  # manual, automatic, dual-zone, tri-zone
    heated_seats = Column(Boolean, default=False)
    cooled_seats = Column(Boolean, default=False)
    heated_steering_wheel = Column(Boolean, default=False)
    rear_climate_control = Column(Boolean, default=False)
    
    # Lighting
    headlight_type = Column(String(50), nullable=True)  # halogen, HID, LED
    fog_lights = Column(Boolean, default=False)
    daytime_running_lights = Column(Boolean, default=False)
    adaptive_headlights = Column(Boolean, default=False)
    
    # Windows and Roof
    power_windows = Column(Boolean, default=False)
    sunroof = Column(Boolean, default=False)
    moonroof = Column(Boolean, default=False)
    panoramic_roof = Column(Boolean, default=False)
    convertible_top = Column(Boolean, default=False)
    
    # Exterior features
    alloy_wheels = Column(Boolean, default=False)
    wheel_size = Column(Integer, nullable=True)  # inches
    roof_rails = Column(Boolean, default=False)
    running_boards = Column(Boolean, default=False)
    tow_hitch = Column(Boolean, default=False)
    
    # Relationships
    car = relationship("EnhancedCar", back_populates="features")

class CarSafety(Base):
    __tablename__ = "car_safety"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey("enhanced_cars.id"), nullable=False)
    
    # Airbag systems
    front_airbags = Column(Boolean, default=True)
    side_airbags = Column(Boolean, default=False)
    curtain_airbags = Column(Boolean, default=False)
    knee_airbags = Column(Boolean, default=False)
    rear_airbags = Column(Boolean, default=False)
    
    # Safety ratings
    nhtsa_overall_rating = Column(Float, nullable=True)  # 1-5 stars
    iihs_top_safety_pick = Column(Boolean, default=False)
    
    # Active safety features
    abs_brakes = Column(Boolean, default=True)
    electronic_stability_control = Column(Boolean, default=True)
    traction_control = Column(Boolean, default=True)
    brake_assist = Column(Boolean, default=False)
    
    # Driver assistance
    forward_collision_warning = Column(Boolean, default=False)
    automatic_emergency_braking = Column(Boolean, default=False)
    blind_spot_monitoring = Column(Boolean, default=False)
    lane_departure_warning = Column(Boolean, default=False)
    lane_keeping_assist = Column(Boolean, default=False)
    adaptive_cruise_control = Column(Boolean, default=False)
    parking_sensors = Column(Boolean, default=False)
    backup_camera = Column(Boolean, default=False)
    surround_view_camera = Column(Boolean, default=False)
    
    # Security
    anti_theft_system = Column(Boolean, default=False)
    remote_start = Column(Boolean, default=False)
    keyless_entry = Column(Boolean, default=False)
    push_button_start = Column(Boolean, default=False)
    
    # Relationships
    car = relationship("EnhancedCar", back_populates="safety")

class CarMediaTechnology(Base):
    __tablename__ = "car_media_technology"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey("enhanced_cars.id"), nullable=False)
    
    # Infotainment
    touchscreen_size = Column(Float, nullable=True)  # inches
    infotainment_system = Column(String(100), nullable=True)
    gps_navigation = Column(Boolean, default=False)
    voice_control = Column(Boolean, default=False)
    
    # Audio system
    audio_system_brand = Column(String(50), nullable=True)
    speaker_count = Column(Integer, nullable=True)
    premium_audio = Column(Boolean, default=False)
    satellite_radio = Column(Boolean, default=False)
    
    # Connectivity
    bluetooth = Column(Boolean, default=False)
    wifi_hotspot = Column(Boolean, default=False)
    apple_carplay = Column(Boolean, default=False)
    android_auto = Column(Boolean, default=False)
    usb_ports = Column(Integer, nullable=True)
    wireless_charging = Column(Boolean, default=False)
    
    # Other tech
    heads_up_display = Column(Boolean, default=False)
    digital_instrument_cluster = Column(Boolean, default=False)
    ambient_lighting = Column(Boolean, default=False)
    
    # Relationships
    car = relationship("EnhancedCar", back_populates="technology")

class CarComfort(Base):
    __tablename__ = "car_comfort"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey("enhanced_cars.id"), nullable=False)
    
    # Seating
    seat_material = Column(String(50), nullable=True)  # cloth, leather, vinyl
    power_driver_seat = Column(Boolean, default=False)
    power_passenger_seat = Column(Boolean, default=False)
    memory_seats = Column(Boolean, default=False)
    lumbar_support = Column(Boolean, default=False)
    seat_ventilation = Column(Boolean, default=False)
    
    # Interior convenience
    power_steering = Column(Boolean, default=True)
    cruise_control = Column(Boolean, default=False)
    tilt_steering = Column(Boolean, default=False)
    telescoping_steering = Column(Boolean, default=False)
    leather_steering_wheel = Column(Boolean, default=False)
    
    # Storage and convenience
    cup_holders = Column(Integer, nullable=True)
    storage_compartments = Column(Integer, nullable=True)
    cargo_organizer = Column(Boolean, default=False)
    cargo_net = Column(Boolean, default=False)
    
    # Mirrors and visibility
    power_mirrors = Column(Boolean, default=False)
    heated_mirrors = Column(Boolean, default=False)
    auto_dimming_mirrors = Column(Boolean, default=False)
    
    # Relationships
    car = relationship("EnhancedCar", back_populates="comfort")

class CarAvailableColors(Base):
    __tablename__ = "car_available_colors"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey("enhanced_cars.id"), nullable=False)
    
    # Exterior Colors
    exterior_colors = Column(JSON, nullable=True)  # List of available exterior colors
    
    # Interior Colors  
    interior_colors = Column(JSON, nullable=True)  # List of available interior colors
    
    # Trim Options
    interior_trim_options = Column(JSON, nullable=True)  # Wood, metal, carbon fiber, etc.

class CarMaintenanceInfo(Base):
    __tablename__ = "car_maintenance_info"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey("enhanced_cars.id"), nullable=False)
    
    # Service Intervals
    oil_change_interval = Column(Integer, nullable=True)  # miles
    air_filter_interval = Column(Integer, nullable=True)  # miles
    cabin_filter_interval = Column(Integer, nullable=True)  # miles
    brake_fluid_interval = Column(Integer, nullable=True)  # months
    coolant_interval = Column(Integer, nullable=True)  # miles
    transmission_service_interval = Column(Integer, nullable=True)  # miles
    
    # Warranty Information
    basic_warranty_years = Column(Integer, nullable=True)
    basic_warranty_miles = Column(Integer, nullable=True)
    powertrain_warranty_years = Column(Integer, nullable=True)
    powertrain_warranty_miles = Column(Integer, nullable=True)
    battery_warranty_years = Column(Integer, nullable=True)  # for electric vehicles
    battery_warranty_miles = Column(Integer, nullable=True)
    
    # Maintenance Costs (estimated annual)
    estimated_annual_maintenance_cost = Column(Float, nullable=True)
    estimated_annual_fuel_cost = Column(Float, nullable=True) 