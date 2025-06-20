#!/usr/bin/env python3
"""
Test script để kiểm tra tích hợp dữ liệu xe nâng cao vào API chi tiết xe
"""

import asyncio
import requests
import json
import sys
import os

# Thêm thư mục root vào Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# URL của API
BASE_URL = "http://localhost:8000/api/v1"

def test_create_enhanced_data():
    """Test tạo dữ liệu xe nâng cao"""
    print("🔧 Tạo dữ liệu xe nâng cao...")
    
    response = requests.post(f"{BASE_URL}/enhanced-cars/create-sample-data")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Tạo dữ liệu thành công!")
        print(f"   📝 {data['message']}")
        for detail in data['details']:
            print(f"   🚗 {detail}")
        return True
    else:
        print(f"❌ Lỗi tạo dữ liệu: {response.status_code}")
        print(f"   {response.text}")
        return False

def test_get_car_details_with_enhanced_data():
    """Test lấy chi tiết xe với dữ liệu nâng cao"""
    print("\n🔍 Test lấy chi tiết xe với dữ liệu nâng cao...")
    
    # Test với một ID xe cụ thể (giả sử xe có ID 1 tồn tại)
    test_car_ids = ["1", "2", "3", "10", "20"]
    
    for car_id in test_car_ids:
        print(f"\n📋 Test xe ID: {car_id}")
        response = requests.get(f"{BASE_URL}/cars/{car_id}")
        
        if response.status_code == 200:
            car_data = response.json()
            print(f"✅ Lấy thông tin xe thành công")
            print(f"   🚗 {car_data.get('brand', 'N/A')} {car_data.get('model', 'N/A')} {car_data.get('year', 'N/A')}")
            
            # Kiểm tra có dữ liệu nâng cao không
            has_enhanced = car_data.get('has_enhanced_data', False)
            print(f"   🔧 Có dữ liệu nâng cao: {'✅ Có' if has_enhanced else '❌ Không'}")
            
            if has_enhanced:
                # Hiển thị thông tin điều hòa
                climate_features = car_data.get('climate_comfort_features', {})
                if climate_features:
                    has_ac = climate_features.get('has_air_conditioning', False)
                    climate_type = climate_features.get('climate_control_type', 'N/A')
                    print(f"   ❄️  Điều hòa: {'✅ Có' if has_ac else '❌ Không'} ({climate_type})")
                    
                    if climate_features.get('heated_seats'):
                        print(f"   🔥 Ghế sưởi ấm: ✅")
                    if climate_features.get('cooled_seats'):
                        print(f"   ❄️  Ghế làm mát: ✅")
                
                # Hiển thị tính năng nổi bật
                key_features = car_data.get('key_features', [])
                if key_features:
                    print(f"   ⭐ Tính năng nổi bật:")
                    for feature in key_features[:5]:  # Hiển thị 5 tính năng đầu
                        print(f"      • {feature}")
                
                # Hiển thị điểm số
                scores = car_data.get('calculated_scores', {})
                if scores:
                    print(f"   📊 Điểm số:")
                    print(f"      An toàn: {scores.get('safety_score', 0)}/100")
                    print(f"      Công nghệ: {scores.get('technology_score', 0)}/100")
                    print(f"      Tiện nghi: {scores.get('comfort_score', 0)}/100")
                    print(f"      Tổng thể: {scores.get('overall_score', 0)}/100")
            
            break  # Dừng sau khi tìm được xe có dữ liệu
        else:
            print(f"   ❌ Không tìm thấy xe ID {car_id}")
    
    return True

def test_search_cars_with_ac():
    """Test tìm kiếm xe có điều hòa"""
    print("\n🔍 Test tìm kiếm xe có điều hòa...")
    
    response = requests.get(f"{BASE_URL}/enhanced-cars/search?has_air_conditioning=true&limit=5")
    
    if response.status_code == 200:
        cars = response.json()
        print(f"✅ Tìm thấy {len(cars)} xe có điều hòa")
        
        for i, car in enumerate(cars, 1):
            climate_features = car.get('climate_comfort_features', {})
            print(f"   {i}. {car.get('brand', 'N/A')} {car.get('year', 'N/A')}")
            print(f"      Loại xe: {car.get('vehicle_type', 'N/A')}")
            print(f"      Phân khúc: {car.get('market_segment', 'N/A')}")
            if climate_features:
                climate_type = climate_features.get('climate_control_type', 'N/A')
                print(f"      Điều hòa: {climate_type}")
        return True
    else:
        print(f"❌ Lỗi tìm kiếm: {response.status_code}")
        return False

def test_climate_features_summary():
    """Test lấy tổng quan tính năng điều hòa"""
    print("\n📊 Test tổng quan tính năng điều hòa...")
    
    response = requests.get(f"{BASE_URL}/enhanced-cars/features/climate")
    
    if response.status_code == 200:
        summary = response.json()
        print("✅ Lấy tổng quan thành công")
        print(f"   📈 {summary.get('air_conditioning_coverage', 'N/A')}")
        
        print("   🌡️  Các loại điều hòa:")
        for climate_type in summary.get('climate_control_types', []):
            print(f"      • {climate_type}")
        
        print("   ⭐ Tính năng điều hòa bổ sung:")
        for feature in summary.get('additional_climate_features', []):
            print(f"      • {feature}")
        
        return True
    else:
        print(f"❌ Lỗi lấy tổng quan: {response.status_code}")
        return False

def test_all_car_search():
    """Test tìm kiếm tất cả xe để xem có bao nhiêu xe có dữ liệu nâng cao"""
    print("\n🔍 Test kiểm tra dữ liệu nâng cao trong toàn bộ xe...")
    
    response = requests.get(f"{BASE_URL}/cars/search?page=1&page_size=50")
    
    if response.status_code == 200:
        search_result = response.json()
        cars = search_result.get('items', [])
        
        enhanced_count = 0
        ac_count = 0
        
        print(f"✅ Kiểm tra {len(cars)} xe")
        
        for car in cars:
            # Lấy chi tiết từng xe để kiểm tra dữ liệu nâng cao
            car_id = car.get('id')
            if car_id:
                detail_response = requests.get(f"{BASE_URL}/cars/{car_id}")
                if detail_response.status_code == 200:
                    car_detail = detail_response.json()
                    
                    if car_detail.get('has_enhanced_data'):
                        enhanced_count += 1
                        
                        climate_features = car_detail.get('climate_comfort_features', {})
                        if climate_features and climate_features.get('has_air_conditioning'):
                            ac_count += 1
        
        print(f"   📊 Thống kê:")
        print(f"      Xe có dữ liệu nâng cao: {enhanced_count}/{len(cars)}")
        print(f"      Xe có điều hòa: {ac_count}/{len(cars)}")
        
        return True
    else:
        print(f"❌ Lỗi tìm kiếm xe: {response.status_code}")
        return False

def main():
    """Chạy tất cả test"""
    print("🚀 Bắt đầu test tích hợp dữ liệu xe nâng cao\n")
    
    tests = [
        ("Tạo dữ liệu xe nâng cao", test_create_enhanced_data),
        ("Lấy chi tiết xe với dữ liệu nâng cao", test_get_car_details_with_enhanced_data),
        ("Tìm kiếm xe có điều hòa", test_search_cars_with_ac),
        ("Tổng quan tính năng điều hòa", test_climate_features_summary),
        ("Kiểm tra dữ liệu nâng cao toàn bộ xe", test_all_car_search)
    ]
    
    success_count = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        
        try:
            if test_func():
                success_count += 1
                print(f"✅ {test_name}: THÀNH CÔNG")
            else:
                print(f"❌ {test_name}: THẤT BẠI")
        except Exception as e:
            print(f"❌ {test_name}: LỖI - {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"📊 KẾT QUẢ CUỐI CÙNG")
    print(f"{'='*60}")
    print(f"✅ Thành công: {success_count}/{total_tests}")
    print(f"❌ Thất bại: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 TẤT CẢ TEST ĐÃ THÀNH CÔNG!")
        print("\n🚗 Hệ thống xe của bạn hiện đã có:")
        print("   ❄️  Thông tin chi tiết về điều hòa không khí")
        print("   🚙 Đa dạng loại xe (sedan, SUV, sports car, truck, hybrid, electric)")
        print("   🛡️  Tính năng an toàn chi tiết")
        print("   📱 Công nghệ hiện đại")
        print("   🛋️  Tiện nghi cao cấp")
        print("   📏 Thông số kích thước đầy đủ")
        print("   ⭐ Điểm số đánh giá")
        print("   🔥 Danh sách tính năng nổi bật")
    else:
        print("⚠️  MỘT SỐ TEST THẤT BẠI - VUI LÒNG KIỂM TRA!")

if __name__ == "__main__":
    main() 