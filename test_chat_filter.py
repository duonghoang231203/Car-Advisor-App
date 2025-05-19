import asyncio
from app.services.rag_service import rag_service

async def test_query_detection():
    # Test general phrases
    general_phrases = [
        "hello",
        "hi there",
        "how are you doing",
        "good morning",
        "thanks",
        "goodbye"
    ]
    
    # Test car-related phrases
    car_phrases = [
        "I need a car for my family",
        "What's the best SUV under $30,000?",
        "Compare Toyota Camry and Honda Accord",
        "fuel efficient vehicles",
        "luxury sedans with good performance",
        "Tell me about Tesla Model 3"
    ]
    
    print("Testing general phrases (should return False):")
    for phrase in general_phrases:
        result = await rag_service._is_car_related_query(phrase)
        print(f"'{phrase}': {result}")
    
    print("\nTesting car-related phrases (should return True):")
    for phrase in car_phrases:
        result = await rag_service._is_car_related_query(phrase)
        print(f"'{phrase}': {result}")

if __name__ == "__main__":
    asyncio.run(test_query_detection()) 