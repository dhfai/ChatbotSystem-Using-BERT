#!/usr/bin/env python3
"""
Script runner utama untuk sistem chatbot Unismuh
"""
import os
import sys
import argparse
import uvicorn

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def run_api(host="0.0.0.0", port=8000, reload=True):
    """Menjalankan API server"""
    print(f"🚀 Starting Chatbot API on {host}:{port}")
    print("📚 Dokumentasi API: http://localhost:8000/docs")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )

def initialize_system(force_rebuild=False):
    """Inisialisasi sistem chatbot"""
    try:
        from src.services.chatbot_service import ChatbotService

        print("🔧 Initializing chatbot system...")
        chatbot = ChatbotService()
        chatbot.initialize(force_rebuild_index=force_rebuild)
        print("✅ System initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return False

def test_system():
    """Test sistem chatbot"""
    try:
        from src.services.chatbot_service import ChatbotService

        print("🧪 Testing chatbot system...")
        chatbot = ChatbotService()

        # Initialize jika belum
        if not chatbot.is_initialized:
            chatbot.initialize()

        # Create test session
        session_id = chatbot.create_session()
        print(f"Session created: {session_id}")

        # Test query
        test_query = "Apa saja fakultas yang ada?"
        result = chatbot.chat(session_id, test_query)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        else:
            print(f"✅ Test successful!")
            print(f"Query: {test_query}")
            print(f"Response: {result['response'][:100]}...")
            return True

    except Exception as e:
        print(f"❌ Error testing system: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Chatbot Unismuh Runner")
    parser.add_argument("command", choices=["init", "api", "test"],
                       help="Command to run")
    parser.add_argument("--host", default="0.0.0.0",
                       help="API host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                       help="API port (default: 8000)")
    parser.add_argument("--no-reload", action="store_true",
                       help="Disable auto-reload")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="Force rebuild vector index")

    args = parser.parse_args()

    if args.command == "init":
        success = initialize_system(force_rebuild=args.force_rebuild)
        sys.exit(0 if success else 1)

    elif args.command == "api":
        # Initialize system first
        if not initialize_system():
            print("❌ Failed to initialize system. Exiting.")
            sys.exit(1)

        # Run API
        run_api(
            host=args.host,
            port=args.port,
            reload=not args.no_reload
        )

    elif args.command == "test":
        success = test_system()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
