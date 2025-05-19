import asyncio
import logging
from app.services.rag_service import RAGService
from app.models.chat import Message

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_rag_query():
    # Initialize the RAG service
    rag = RAGService()

    # First query
    print("\n=== FIRST QUERY: What are some good Coupes ===")
    first_result = await rag.process_query('What are some good Coupes')

    # Print the first result
    print("\n=== FIRST RESPONSE ===")
    print(first_result['response'])

    print("\n=== FIRST SUGGESTIONS ===")
    for suggestion in first_result['suggestions']:
        print(f"- {suggestion['name']} (ID: {suggestion['id']})")

    print(f"\nTotal first suggestions: {len(first_result['suggestions'])}")

    # Create conversation history
    conversation_history = [
        Message(role="user", content="What are some good Coupes"),
        Message(role="assistant", content=first_result['response'])
    ]

    # Follow-up query asking for more options
    print("\n\n=== FOLLOW-UP QUERY: Show me more coupes ===")
    follow_up_result = await rag.process_query('Show me more coupes', conversation_history)

    # Print the follow-up result
    print("\n=== FOLLOW-UP RESPONSE ===")
    print(follow_up_result['response'])

    print("\n=== FOLLOW-UP SUGGESTIONS ===")
    for suggestion in follow_up_result['suggestions']:
        print(f"- {suggestion['name']} (ID: {suggestion['id']})")

    print(f"\nTotal follow-up suggestions: {len(follow_up_result['suggestions'])}")

if __name__ == "__main__":
    asyncio.run(test_rag_query())
