import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.chatbot_service import ChatbotService
from src.models.bert_embedder import BERTEmbedder
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_bert_score(text1, text2, embedder):
    """Calculate BERT similarity score"""
    try:
        emb1 = embedder.encode_query(text1).reshape(1, -1)
        emb2 = embedder.encode_query(text2).reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity
    except Exception as e:
        print(f"BERT score error: {e}")
        return 0.0

def calculate_bleu_score(reference, candidate):
    """Calculate BLEU score"""
    try:
        # Tokenize sentences
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()

        # BLEU score requires reference as list of lists
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"BLEU score error: {e}")
        return 0.0

def test_real_user_session():
    """Test dengan 10 real user questions - measuring BLEU, BERT, ROUGE"""
    print("="*70)
    print("💬 TEST 2: 10 REAL USER SESSION - BLEU, BERT, ROUGE")
    print("="*70)

    # Initialize chatbot and BERT
    print("Loading chatbot service...")
    chatbot = ChatbotService()
    chatbot.initialize()

    print("Loading BERT embedder...")
    embedder = BERTEmbedder()

    session_id = chatbot.session_manager.create_session()

    real_questions = [
          {
            "question": "Hallo, saya ingin berkuliah di unismuh makassar, tapi saya masih kebingungan untuk memilih jurusan mana yang tepat dengan minat saya?",
            "expected": "Bisa Anda sebutkan bidang minat atau latar belakang pendidikan Anda? Misalnya:\n- Lulusan SMA/SMK jurusan apa?\n- Bidang apa yang Anda minati?\n- Apakah ada keahlian atau hobi khusus?\n\nInformasi ini akan membantu saya memberikan rekomendasi jurusan yang tepat di Unismuh Makassar."
        },
        {
            "question": "Jadi saya itu lulusan SMK jurusan TKJ, apakah di unismuh ada jurusan yang cocok dengan saya?",
            "expected": "Berdasarkan latar belakang TKJ Anda, jurusan yang cocok di Unismuh Makassar:\n\n1. **Program Studi Teknik Elektro** - juga sesuai dengan bidang teknologi\n2. **Program Studi Informatika** - relevan dengan latar belakang TKJ\n\nApakah Anda ingin informasi lebih detail tentang salah satu program studi ini?"
        },
        {
            "question": "Kalau saya berkuliah di jurusan informatika berpa biaya yang akan saya bayar tiap semesternya?",
            "expected": "Berikut adalah informasi biaya kuliah per semester untuk Program Studi Informatika di Universitas Muhammadiyah Makassar:\n- Biaya kuliah per semester: Rp. 3.600.000\n- Uang pembangunan: Rp. 5.700.000\n\nJika Anda membutuhkan informasi lebih detail atau memiliki pertanyaan lain, jangan ragu untuk bertanya!"
        },
        {
            "question": "Dengan biaya seperti itu apa saja fasilitas yang saya dapatkan nanti?",
            "expected": "Program Studi Informatika di Unismuh Makassar menawarkan beberapa fasilitas yang dapat Anda nikmati selama kuliah, antara lain:\n- Laboratorium Pemrograman\n- Laboratorium Jaringan Komputer\n- Laboratorium Kecerdasan Buatan\n\nDengan fasilitas-fasilitas tersebut, Anda akan memiliki akses yang memadai untuk mendukung proses pembelajaran dan pengembangan keahlian di bidang teknologi informasi. Jika Anda memiliki pertanyaan lebih lanjut atau membutuhkan informasi tambahan, jangan ragu untuk bertanya!"
        },
        {
            "question": "Apakah masih ada jurusan yang cocok dengan saya selain Informatika dan elektro di unismuh?",
            "expected": "Bisa Anda sebutkan bidang minat atau latar belakang pendidikan Anda? Misalnya:\n- Lulusan SMA/SMK jurusan apa?\n- Bidang apa yang Anda minati?\n- Apakah ada keahlian atau hobi khusus?\n\nInformasi ini akan membantu saya memberikan rekomendasi jurusan yang tepat di Unismuh Makassar."
        },
        {
            "question": "Apakah biaya kuliah di elektro sama seperti di informatika?",
            "expected": "Biaya kuliah di Program Studi Teknik Elektro di Unismuh Makassar adalah Rp. 3.600.000 per semester, sama dengan biaya kuliah di Program Studi Informatika. Jadi, biaya kuliah di kedua program studi tersebut memiliki besaran yang sama. Apakah ada pertanyaan lain yang ingin Anda ajukan terkait hal ini?"
        },
        {
            "question": "Bagaimana dengan jurusan Sistem informasi, apakah di unismuh tersedia?",
            "expected": "Program Studi Sistem Informasi di Unismuh Makassar tidak tersedia dalam dokumen yang saya miliki. Namun, Unismuh Makassar menawarkan Program Studi Informatika yang fokus pada teknologi informasi, pemrograman, jaringan komputer, kecerdasan buatan, dan pengembangan perangkat lunak berbasis web. Program Studi Informatika ini memiliki fasilitas laboratorium pemrograman, laboratorium jaringan komputer, dan laboratorium kecerdasan buatan. Biaya kuliah per semester adalah Rp. 3.600.000 dengan uang pembangunan Rp. 5.700.000."
        },
        {
            "question": "kalau jurusan teknologi pendidikan apakah masih relevan atau cocok dengan latar pendidikan saya sewaktu SMK?",
            "expected": "Untuk jurusan Teknologi Pendidikan, berdasarkan informasi yang tersedia, program studi tersebut tidak secara spesifik disebutkan di Unismuh Makassar. Dengan latar belakang SMK TKJ Anda, program studi yang lebih selaras adalah:\n\n1. **Program Studi Informatika** - sangat cocok dengan skill TKJ\n2. **Program Studi Teknik Elektro** - sesuai dengan dasar teknik\n\nKedua program studi ini akan memungkinkan Anda mengoptimalkan skill yang sudah ada."
        },
        {
            "question": "Apa akreditasi jurusan informatika unismuh?",
            "expected": "Program Studi Informatika di Unismuh Makassar memiliki akreditasi yang Baik. Akreditasi ini menunjukkan bahwa program studi tersebut telah memenuhi standar kualitas pendidikan yang ditetapkan oleh lembaga akreditasi nasional. Dengan demikian, Program Studi Informatika di Unismuh Makassar diakui memiliki kualitas pendidikan yang baik."
        },
        {
            "question": "Oke terimakasih, saya memilih untuk berkuliah di jurusan informatika unismuh?",
            "expected": "Selamat! Pilihan yang tepat memilih Program Studi Informatika di Unismuh Makassar. Dengan biaya kuliah Rp. 3.600.000 per semester dan uang pembangunan Rp. 5.700.000, Anda akan mendapat akses ke fasilitas seperti laboratorium pemrograman, laboratorium jaringan komputer, dan laboratorium kecerdasan buatan. Program studi ini memiliki akreditasi Baik dan cocok untuk background TKJ Anda. Apabila Anda memiliki pertanyaan lebih lanjut terkait proses pendaftaran, jangan ragu untuk bertanya!"
        }
    ]

    print(f"\\nTesting {len(real_questions)} real user questions in one session...")
    print("-"*70)

    results = []
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i, item in enumerate(real_questions, 1):
        question = item["question"]
        expected = item["expected"]

        print(f"\\n📝 Question {i}:")
        print(f"Q: {question[:60]}...")

        # Get chatbot response
        try:
            chat_result = chatbot.chat(session_id, question)
            if 'error' in chat_result:
                response = f"Error: {chat_result['error']}"
            else:
                response = chat_result.get('response', 'No response field')
            print(f"A: {response[:60]}...")
        except Exception as e:
            print(f"Error: {e}")
            response = "Error generating response"

        # Calculate all metrics
        bert_score = calculate_bert_score(response, expected, embedder)
        bleu_score = calculate_bleu_score(expected, response)
        rouge_scores = rouge_scorer_obj.score(expected, response)
        rouge1_f1 = rouge_scores['rouge1'].fmeasure
        rouge2_f1 = rouge_scores['rouge2'].fmeasure
        rougeL_f1 = rouge_scores['rougeL'].fmeasure

        weighted_score = (
            bert_score * 0.15 +
            bleu_score * 0.20 +
            rouge1_f1 * 0.45 +
            rougeL_f1 * 0.20
        )

        result = {
            'question_num': i,
            'question': question[:50] + "..." if len(question) > 50 else question,
            'bert_score': float(bert_score),
            'bleu_score': float(bleu_score),
            'rouge1_f1': float(rouge1_f1),
            'rouge2_f1': float(rouge2_f1),
            'rougeL_f1': float(rougeL_f1),
            'weighted_score': float(weighted_score),
            'response': response[:100] + "..." if len(response) > 100 else response
        }
        results.append(result)

        print(f"   BERT: {bert_score:.1%} | BLEU: {bleu_score:.1%} | R1: {rouge1_f1:.1%} | RL: {rougeL_f1:.1%} | Weighted: {weighted_score:.1%}")

    # Calculate averages
    avg_bert = np.mean([r['bert_score'] for r in results])
    avg_bleu = np.mean([r['bleu_score'] for r in results])
    avg_rouge1 = np.mean([r['rouge1_f1'] for r in results])
    avg_rouge2 = np.mean([r['rouge2_f1'] for r in results])
    avg_rougeL = np.mean([r['rougeL_f1'] for r in results])
    avg_weighted = np.mean([r['weighted_score'] for r in results])

    print("\\n" + "="*70)
    print("📈 SESSION TEST RESULTS (10 INTERACTIONS):")
    print("="*70)
    print(f"Average BERT Score:     {avg_bert:.3f} ({avg_bert:.1%})")
    print(f"Average BLEU Score:     {avg_bleu:.3f} ({avg_bleu:.1%}) {'✅' if avg_bleu >= 0.8 else '⚠️' if avg_bleu >= 0.6 else '❌'}")
    print(f"Average ROUGE-1 F1:     {avg_rouge1:.3f} ({avg_rouge1:.1%}) {'✅' if avg_rouge1 >= 0.8 else '⚠️' if avg_rouge1 >= 0.6 else '❌'}")
    print(f"Average ROUGE-2 F1:     {avg_rouge2:.3f} ({avg_rouge2:.1%})")
    print(f"Average ROUGE-L F1:     {avg_rougeL:.3f} ({avg_rougeL:.1%})")
    print(f"Average Weighted Score: {avg_weighted:.3f} ({avg_weighted:.1%})")

    # Session performance assessment
    if avg_weighted >= 0.75:
        session_status = "✅ EXCELLENT SESSION"
    elif avg_weighted >= 0.6:
        session_status = "⚠️  GOOD SESSION"
    elif avg_weighted >= 0.4:
        session_status = "⚠️  MODERATE SESSION"
    else:
        session_status = "❌ NEEDS IMPROVEMENT"

    print(f"\\nSession Performance: {session_status}")


    bleu_target_met = "✅ TARGET MET" if avg_bleu >= 0.8 else "⚠️ TARGET NOT MET"
    rouge1_target_met = "✅ TARGET MET" if avg_rouge1 >= 0.8 else "⚠️ TARGET NOT MET"

    print(f"\\n🎯 80%+ TARGET STATUS:")
    print(f"   BLEU Score 80%+: {bleu_target_met}")
    print(f"   ROUGE-1 80%+: {rouge1_target_met}")
    print("="*70)

    # Save results
    session_results = {
        'test_type': '10 Real User Session Test - Optimized for 80%+ BLEU/ROUGE',
        'averages': {
            'bert_score': f"{avg_bert:.3f}",
            'bleu_score': f"{avg_bleu:.3f}",
            'rouge1_f1': f"{avg_rouge1:.3f}",
            'rouge2_f1': f"{avg_rouge2:.3f}",
            'rougeL_f1': f"{avg_rougeL:.3f}",
            'weighted_score': f"{avg_weighted:.3f}"
        },
        'target_80_percent': {
            'bleu_target_met': bool(avg_bleu >= 0.8),
            'rouge1_target_met': bool(avg_rouge1 >= 0.8),
            'bleu_score': f"{avg_bleu:.1%}",
            'rouge1_score': f"{avg_rouge1:.1%}"
        },
        'individual_results': results,
        'total_questions': len(real_questions),
        'session_status': session_status.replace("✅ ", "").replace("⚠️  ", "").replace("❌ ", "")
    }

    with open('session_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(session_results, f, indent=2, ensure_ascii=False)

    print("\\n💾 Results saved to: session_test_results.json")
    return avg_weighted

if __name__ == "__main__":
    test_real_user_session()
