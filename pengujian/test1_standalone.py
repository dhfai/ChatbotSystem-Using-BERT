#!/usr/bin/env python3
"""
Run Test 1: Evaluasi 2 Pertanyaan - Dijalankan dari root folder
"""

import os
import sys
import time

# Add src to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import evaluation metrics libraries first
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    import nltk

    # Download required NLTK data
    try:
        import nltk.data
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt data...")
        nltk.download('punkt', quiet=True)

    print("✅ All evaluation libraries loaded successfully")

except ImportError as e:
    print(f"❌ Missing evaluation libraries: {e}")
    print("Please install: pip install nltk rouge-score bert-score")
    sys.exit(1)

# Import chatbot components
from src.services.chatbot_service import ChatbotService

def calculate_bleu_score(reference: str, candidate: str) -> float:
    """Calculate BLEU score between reference and candidate"""
    try:
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return 0.0

def calculate_bert_score(reference: str, candidate: str) -> dict:
    """Calculate BERT score between reference and candidate"""
    try:
        P, R, F1 = bert_score([candidate], [reference], lang='id', verbose=False)
        return {
            'precision': float(P[0]),
            'recall': float(R[0]),
            'f1': float(F1[0])
        }
    except Exception as e:
        print(f"Error calculating BERT score: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

def calculate_rouge_score(reference: str, candidate: str) -> dict:
    """Calculate ROUGE scores between reference and candidate"""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

def test_two_questions():
    """Test evaluasi untuk 2 pertanyaan dengan expected answers"""
    print("="*60)
    print("TEST 1: EVALUASI 2 PERTANYAAN")
    print("="*60)

    # Initialize chatbot
    print("Initializing chatbot service...")
    chatbot_service = ChatbotService()
    chatbot_service.initialize()
    session_id = chatbot_service.create_session()
    print(f"✅ Chatbot initialized with session: {session_id}")

    # Test questions dengan expected answers
    test_data = [
        {
            "question": "Apa saja fakultas yang ada di Universitas Muhammadiyah Makassar?",
            "expected_answer": "Universitas Muhammadiyah Makassar memiliki beberapa fakultas yaitu Fakultas Ekonomi dan Bisnis, Fakultas Teknik, dan Fakultas Keguruan dan Ilmu Pendidikan. Fakultas Ekonomi dan Bisnis fokus pada pengembangan ilmu ekonomi, bisnis, dan akuntansi berbasis nilai-nilai Islam. Fakultas Teknik didirikan pada tahun 1987 dengan Program Studi Teknik Pengairan. Fakultas Keguruan dan Ilmu Pendidikan merupakan salah satu fakultas tertua yang berdiri sejak awal pendirian universitas."
        },
        {
            "question": "Siapa dekan Fakultas Teknik Universitas Muhammadiyah Makassar?",
            "expected_answer": "Dekan Fakultas Teknik Universitas Muhammadiyah Makassar adalah Dr. Ir. Hj. Nurnawaty S.T. M.T. IPM. Fakultas ini juga memiliki wakil dekan yaitu Wakil Dekan I Ir. Muhammad Syafa'at S. Kuba S.T. M.T., Wakil Dekan II Dr. Ir. Andi Makbul Syamsuri S.T. M.T. IPM, dan Wakil Dekan III Dr. Abd. Rahman Bahtiar S.Ag. M.A."
        }
    ]

    results = []

    for i, test_item in enumerate(test_data, 1):
        print(f"\n{'-'*50}")
        print(f"EVALUATING QUESTION {i}")
        print(f"{'-'*50}")

        question = test_item["question"]
        expected = test_item["expected_answer"]

        print(f"Question: {question}")
        print(f"Expected: {expected[:100]}...")

        # Get chatbot response
        start_time = time.time()
        response = chatbot_service.chat(session_id, question)
        response_time = time.time() - start_time

        if "error" in response:
            print(f"❌ Error: {response['error']}")
            continue

        actual_answer = response.get('response', '')
        print(f"Actual: {actual_answer[:100]}...")
        print(f"Response time: {response_time:.2f}s")

        # Calculate all metrics
        print("\nCalculating evaluation metrics...")
        bleu_score_val = calculate_bleu_score(expected, actual_answer)
        bert_scores = calculate_bert_score(expected, actual_answer)
        rouge_scores = calculate_rouge_score(expected, actual_answer)

        result = {
            'question': question,
            'expected_answer': expected,
            'actual_answer': actual_answer,
            'response_time': response_time,
            'scores': {
                'bleu': bleu_score_val,
                'bert': bert_scores,
                'rouge': rouge_scores
            },
            'retrieved_docs': response.get('retrieved_documents', 0),
            'sources': response.get('sources', [])
        }

        results.append(result)

        # Print scores
        print(f"\n📊 EVALUATION SCORES:")
        print(f"  BLEU Score:     {bleu_score_val:.4f}")
        print(f"  BERT-F1 Score:  {bert_scores['f1']:.4f}")
        print(f"  BERT Precision: {bert_scores['precision']:.4f}")
        print(f"  BERT Recall:    {bert_scores['recall']:.4f}")
        print(f"  ROUGE-1:        {rouge_scores['rouge1']:.4f}")
        print(f"  ROUGE-2:        {rouge_scores['rouge2']:.4f}")
        print(f"  ROUGE-L:        {rouge_scores['rougeL']:.4f}")
        print(f"  Retrieved Docs: {response.get('retrieved_documents', 0)}")

    # Print summary
    print_summary(results)

    # Save results
    save_results(results)

    return results

def print_summary(results):
    """Print formatted summary of results"""
    print(f"\n{'='*70}")
    print(f"{'RINGKASAN HASIL EVALUASI 2 PERTANYAAN':^70}")
    print(f"{'='*70}")

    if not results:
        print("No results to summarize.")
        return

    # Calculate averages
    avg_bleu = sum(r['scores']['bleu'] for r in results) / len(results)
    avg_bert_f1 = sum(r['scores']['bert']['f1'] for r in results) / len(results)
    avg_rouge1 = sum(r['scores']['rouge']['rouge1'] for r in results) / len(results)
    avg_rougeL = sum(r['scores']['rouge']['rougeL'] for r in results) / len(results)
    avg_response_time = sum(r['response_time'] for r in results) / len(results)

    print(f"Total Questions: {len(results)}")
    print(f"Average Response Time: {avg_response_time:.2f}s")
    print()
    print(f"{'Metric':<15} {'Question 1':<12} {'Question 2':<12} {'Average':<12}")
    print(f"{'-'*55}")
    print(f"{'BLEU':<15} {results[0]['scores']['bleu']:<12.4f} {results[1]['scores']['bleu'] if len(results) > 1 else 0:<12.4f} {avg_bleu:<12.4f}")
    print(f"{'BERT-F1':<15} {results[0]['scores']['bert']['f1']:<12.4f} {results[1]['scores']['bert']['f1'] if len(results) > 1 else 0:<12.4f} {avg_bert_f1:<12.4f}")
    print(f"{'ROUGE-1':<15} {results[0]['scores']['rouge']['rouge1']:<12.4f} {results[1]['scores']['rouge']['rouge1'] if len(results) > 1 else 0:<12.4f} {avg_rouge1:<12.4f}")
    print(f"{'ROUGE-L':<15} {results[0]['scores']['rouge']['rougeL']:<12.4f} {results[1]['scores']['rouge']['rougeL'] if len(results) > 1 else 0:<12.4f} {avg_rougeL:<12.4f}")

    print(f"\n{'='*70}")

def save_results(results):
    """Save results to JSON file"""
    import json
    from datetime import datetime

    output_dir = "pengujian/output"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare final results
    final_results = {
        'test_type': 'two_questions_evaluation',
        'timestamp': datetime.now().isoformat(),
        'total_questions': len(results),
        'results': results,
        'summary': {
            'avg_bleu': sum(r['scores']['bleu'] for r in results) / len(results) if results else 0,
            'avg_bert_f1': sum(r['scores']['bert']['f1'] for r in results) / len(results) if results else 0,
            'avg_rouge1': sum(r['scores']['rouge']['rouge1'] for r in results) / len(results) if results else 0,
            'avg_rougeL': sum(r['scores']['rouge']['rougeL'] for r in results) / len(results) if results else 0,
            'avg_response_time': sum(r['response_time'] for r in results) / len(results) if results else 0
        }
    }

    filepath = os.path.join(output_dir, 'test1_two_questions_results.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"✅ Results saved to: {filepath}")

if __name__ == "__main__":
    try:
        results = test_two_questions()
        print("\n🎉 Test 1 completed successfully!")
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
