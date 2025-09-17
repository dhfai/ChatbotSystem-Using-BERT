"""
TEST 3: Two Different Questions Test
===================================
Test dengan 2 pertanyaan berbeda untuk mengukur BLEU, BERT, dan ROUGE scores
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.chatbot_service import ChatbotService
from src.models.bert_embedder import BERTEmbedder
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
import json
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

def calculate_rouge_scores(reference, candidate):
    """Calculate ROUGE scores"""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_f1': scores['rougeL'].fmeasure
        }
    except Exception as e:
        print(f"ROUGE score error: {e}")
        return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}

def test_two_questions():
    """Test dengan 2 pertanyaan berbeda dan ukur BLEU, BERT, ROUGE"""
    print("="*70)
    print("📝 TEST 3: TWO DIFFERENT QUESTIONS - BLEU, BERT, ROUGE")
    print("="*70)

    # Initialize components
    print("Loading chatbot service...")
    chatbot = ChatbotService()
    chatbot.initialize()  # Initialize the chatbot service

    print("Loading BERT embedder...")
    embedder = BERTEmbedder()

    # Create session for chatbot
    session_id = chatbot.session_manager.create_session()    # Dua pertanyaan yang berbeda dengan expected answers
    test_questions = [
        {
            "question": "Apa saja jurusan yang tersedia di Fakultas Teknik Unismuh Makassar?",
            "expected": "Fakultas Teknik Unismuh Makassar memiliki beberapa jurusan seperti Teknik Informatika, Teknik Sipil, Teknik Elektro, dan Teknik Mesin yang berkualitas dan terakreditasi."
        },
        {
            "question": "Bagaimana proses pendaftaran mahasiswa baru di Unismuh?",
            "expected": "Proses pendaftaran mahasiswa baru dapat dilakukan secara online melalui website resmi atau datang langsung ke kampus dengan membawa persyaratan lengkap seperti ijazah dan dokumen pendukung."
        }
    ]

    print(f"\\nTesting {len(test_questions)} different questions...")
    print("-"*70)

    all_results = []

    for i, item in enumerate(test_questions, 1):
        question = item["question"]
        expected = item["expected"]

        print(f"\\n🔍 QUESTION {i}:")
        print(f"Q: {question}")

        # Get chatbot response
        try:
            chat_result = chatbot.chat(session_id, question)
            if 'error' in chat_result:
                print(f"Chat error: {chat_result['error']}")
                response = f"Error: {chat_result['error']}"
            else:
                response = chat_result.get('response', 'No response field')
            print(f"A: {response}")
            print(f"Expected: {expected}")

        except Exception as e:
            print(f"Exception error: {e}")
            response = "Error: Could not generate response"

        print("\\n📊 CALCULATING SCORES...")

        # Calculate all scores
        bert_score = calculate_bert_score(response, expected, embedder)
        bleu_score = calculate_bleu_score(expected, response)
        rouge_scores = calculate_rouge_scores(expected, response)

        # Weighted overall score
        weighted_score = (
            bert_score * 0.4 +
            bleu_score * 0.2 +
            rouge_scores['rouge1_f1'] * 0.2 +
            rouge_scores['rougeL_f1'] * 0.2
        )

        result = {
            'question_num': i,
            'question': question,
            'response': response,
            'expected': expected,
            'bert_score': float(bert_score),
            'bleu_score': float(bleu_score),
            'rouge1_f1': float(rouge_scores['rouge1_f1']),
            'rouge2_f1': float(rouge_scores['rouge2_f1']),
            'rougeL_f1': float(rouge_scores['rougeL_f1']),
            'weighted_score': float(weighted_score)
        }
        all_results.append(result)

        # Display scores
        print(f"   BERT Score:   {bert_score:.3f} ({bert_score:.1%})")
        print(f"   BLEU Score:   {bleu_score:.3f} ({bleu_score:.1%})")
        print(f"   ROUGE-1 F1:   {rouge_scores['rouge1_f1']:.3f} ({rouge_scores['rouge1_f1']:.1%})")
        print(f"   ROUGE-2 F1:   {rouge_scores['rouge2_f1']:.3f} ({rouge_scores['rouge2_f1']:.1%})")
        print(f"   ROUGE-L F1:   {rouge_scores['rougeL_f1']:.3f} ({rouge_scores['rougeL_f1']:.1%})")
        print(f"   Weighted Score: {weighted_score:.3f} ({weighted_score:.1%})")
        print("-"*50)

    # Calculate averages
    avg_bert = sum(r['bert_score'] for r in all_results) / len(all_results)
    avg_bleu = sum(r['bleu_score'] for r in all_results) / len(all_results)
    avg_rouge1 = sum(r['rouge1_f1'] for r in all_results) / len(all_results)
    avg_rouge2 = sum(r['rouge2_f1'] for r in all_results) / len(all_results)
    avg_rougeL = sum(r['rougeL_f1'] for r in all_results) / len(all_results)
    avg_weighted = sum(r['weighted_score'] for r in all_results) / len(all_results)

    print("\\n" + "="*70)
    print("📈 FINAL RESULTS - TWO QUESTIONS TEST:")
    print("="*70)
    print(f"Average BERT Score:     {avg_bert:.3f} ({avg_bert:.1%})")
    print(f"Average BLEU Score:     {avg_bleu:.3f} ({avg_bleu:.1%})")
    print(f"Average ROUGE-1 F1:     {avg_rouge1:.3f} ({avg_rouge1:.1%})")
    print(f"Average ROUGE-2 F1:     {avg_rouge2:.3f} ({avg_rouge2:.1%})")
    print(f"Average ROUGE-L F1:     {avg_rougeL:.3f} ({avg_rougeL:.1%})")
    print(f"Average Weighted Score: {avg_weighted:.3f} ({avg_weighted:.1%})")

    # Overall assessment
    if avg_weighted >= 0.7:
        status = "✅ EXCELLENT"
    elif avg_weighted >= 0.5:
        status = "⚠️  GOOD"
    elif avg_weighted >= 0.3:
        status = "⚠️  MODERATE"
    else:
        status = "❌ NEEDS IMPROVEMENT"

    print(f"\\nOverall Performance: {status}")
    print("="*70)

    # Save detailed results
    final_results = {
        'test_type': 'Two Different Questions Test',
        'averages': {
            'bert_score': f"{avg_bert:.3f}",
            'bleu_score': f"{avg_bleu:.3f}",
            'rouge1_f1': f"{avg_rouge1:.3f}",
            'rouge2_f1': f"{avg_rouge2:.3f}",
            'rougeL_f1': f"{avg_rougeL:.3f}",
            'weighted_score': f"{avg_weighted:.3f}"
        },
        'individual_results': all_results,
        'overall_status': status.replace("✅ ", "").replace("⚠️  ", "").replace("❌ ", "")
    }

    with open('two_questions_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("\\n💾 Results saved to: two_questions_test_results.json")
    return avg_weighted

if __name__ == "__main__":
    test_two_questions()
