from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey, Numeric
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Car(Base):
    __tablename__ = "cars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    brand = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False)
    year = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    condition = Column("car_condition", String(50), nullable=False)  # new or used
    type = Column("car_type", String(50), nullable=False)  # buy or rent
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    specifications = relationship("CarSpecification", back_populates="car", uselist=False)

class CarSpecification(Base):
    __tablename__ = "car_specifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey("cars.id"), nullable=False)
    engine = Column(String(100), nullable=True)
    transmission = Column(String(100), nullable=True)
    fuel_type = Column(String(50), nullable=True)
    mileage = Column(Numeric(10, 2), nullable=True)
    seating_capacity = Column(Integer, nullable=True)
    body_type = Column(String(50), nullable=True)  # This is the vehicle_style

    # Relationships
    car = relationship("Car", back_populates="specifications")
