import os
import sys

# Add src to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.chatbot_service import ChatbotService
from src.config import Config

def main():
    """
    Script untuk inisialisasi dan testing sistem chatbot
    """
    print("=== Chatbot Unismuh Initialization ===")

    # Create chatbot service
    chatbot_service = ChatbotService()

    # Initialize system
    try:
        print("Initializing chatbot system...")
        chatbot_service.initialize(force_rebuild_index=True)
        print("✅ System initialized successfully!")

        # Test basic functionality
        print("\n=== Testing Basic Functionality ===")

        # Create test session
        session_id = chatbot_service.create_session()
        print(f"✅ Test session created: {session_id}")

        # Test queries
        test_queries = [
            "Apa saja fakultas yang ada di Universitas Muhammadiyah?",
            "Berapa biaya kuliah di universitas ini?",
            "Bagaimana cara mendaftar kuliah?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"User: {query}")

            result = chatbot_service.chat(session_id, query)

            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"Bot: {result['response']}")
                print(f"Retrieved docs: {result['retrieved_documents']}")
                print(f"Sources: {result['sources']}")

        # Show system stats
        print(f"\n=== System Stats ===")
        stats = chatbot_service.get_system_stats()
        print(f"Active sessions: {stats['active_sessions']}")
        print(f"System initialized: {stats['is_initialized']}")
        print(f"Index exists: {stats['index_exists']}")

        print("\n✅ All tests completed successfully!")
        print("\nSistem siap digunakan! Jalankan API dengan:")
        print("python -m uvicorn src.api.main:app --reload")

    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
