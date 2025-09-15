from src.services.chatbot_service import ChatbotService

# Initialize chatbot service
chatbot = ChatbotService()
chatbot.initialize()

# Create a session
session_id = chatbot.create_session()

# Add conversation history
chatbot.session_manager.sessions[session_id]["history"] = [
    {'role': 'user', 'content': 'Jadi saya itu lulusan SMK jurusan TKJ, apakah di unismuh ada jurusan yang cocok dengan saya?'},
    {'role': 'assistant', 'content': 'Berdasarkan latar belakang TKJ Anda, jurusan yang cocok di Unismuh Makassar: 1. Program Studi Informatika - paling relevan dengan latar belakang TKJ, 2. Program Studi Teknik Elektro - juga sesuai dengan bidang teknologi'}
]

# Test query tentang teknologi pendidikan
query = 'kalau jurusan teknologi pendidikan apakah masih relevan atau cocok dengan latar pendidikan saya sewaktu SMK?'
print('Query:', query)
print()

response = chatbot.chat(session_id, query)
print('Response:', response['response'])
