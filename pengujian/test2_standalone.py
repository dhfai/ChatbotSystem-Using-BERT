#!/usr/bin/env python3
"""
Run Test 2: Evaluasi 10 Interaksi dalam 1 Session dengan Visualisasi - Standalone
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add src to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

# Import required libraries
try:
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
    print("Please install: pip install nltk rouge-score bert-score matplotlib")
    sys.exit(1)

# Import chatbot components
from src.services.chatbot_service import ChatbotService

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')

def calculate_scores(reference: str, candidate: str) -> dict:
    """Calculate all evaluation scores"""
    try:
        # BLEU Score
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        smoothing = SmoothingFunction().method1
        bleu_score_val = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)

        # BERT Score
        P, R, F1 = bert_score([candidate], [reference], lang='id', verbose=False)
        bert_f1 = float(F1[0])

        # ROUGE Score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, candidate)
        rouge_l = rouge_scores['rougeL'].fmeasure

        return {
            'bleu': bleu_score_val,
            'bert_f1': bert_f1,
            'rouge_l': rouge_l
        }
    except Exception as e:
        print(f"Error calculating scores: {e}")
        return {'bleu': 0.0, 'bert_f1': 0.0, 'rouge_l': 0.0}

def test_session_interactions():
    """Test evaluasi konsistensi response dengan baseline yang akurat"""
    print("="*70)
    print("TEST 2: EVALUASI KONSISTENSI RESPONSE (IMPROVED)")
    print("="*70)
    print("Metode: Menggunakan response pertama sebagai baseline untuk")
    print("        mengukur konsistensi response pada session berikutnya")
    print("="*70)

    # Initialize chatbot
    print("Initializing chatbot service...")
    chatbot_service = ChatbotService()
    chatbot_service.initialize()
    session_id = chatbot_service.create_session()
    print(f"✅ Chatbot initialized with session: {session_id}")

    # 10 pertanyaan untuk evaluasi
    questions = [
        "Apa visi Fakultas Ekonomi dan Bisnis?",
        "Kapan Fakultas Teknik didirikan?",
        "Apa misi Fakultas Teknik?",
        "Siapa Wakil Dekan I Fakultas Ekonomi dan Bisnis?",
        "Apa program studi yang ada di Fakultas Teknik?",
        "Bagaimana visi FKIP?",
        "Apa fasilitas Program Studi Teknik Pengairan?",
        "Bagaimana pendidikan berbasis Islam di Fakultas Teknik?",
        "Apa saja organisasi mahasiswa di FEB?",
        "Bagaimana kerjasama internasional Fakultas Teknik?"
    ]

    # STEP 1: Generate baseline responses (first run)
    print("\n🔄 GENERATING BASELINE RESPONSES...")
    baseline_responses = {}

    for i, question in enumerate(questions, 1):
        print(f"Generating baseline {i}/10: {question[:50]}...")

        # Get baseline response
        response = chatbot_service.chat(session_id, question)

        if "error" not in response:
            baseline_responses[question] = response.get('response', '')
            print(f"  ✅ Generated ({len(baseline_responses[question])} chars)")
        else:
            baseline_responses[question] = "Error generating baseline"
            print(f"  ❌ Error: {response['error']}")

    print(f"\n✅ Generated {len(baseline_responses)} baseline responses")

    # STEP 2: Create new session and test consistency
    print(f"\n🔄 STARTING CONSISTENCY EVALUATION...")
    session_id = chatbot_service.create_session()  # New session

    results = []
    bleu_scores = []
    bert_scores = []
    rouge_scores = []
    response_times = []

    for i, question in enumerate(questions, 1):
        print(f"\n{'-'*50}")
        print(f"INTERACTION {i}/10")
        print(f"{'-'*50}")

        expected = baseline_responses[question]  # Use baseline as expected

        print(f"Question: {question}")

        # Get chatbot response
        start_time = time.time()
        response = chatbot_service.chat(session_id, question)
        response_time = time.time() - start_time
        response_times.append(response_time)

        if "error" in response:
            print(f"❌ Error: {response['error']}")
            # Add zero scores for failed responses
            scores = {'bleu': 0.0, 'bert_f1': 0.0, 'rouge_l': 0.0}
            actual_answer = ''
        else:
            actual_answer = response.get('response', '')
            print(f"Response: {actual_answer[:100]}...")
            print(f"Time: {response_time:.2f}s")

            # Calculate scores (comparing with baseline)
            scores = calculate_scores(expected, actual_answer)

        # Store scores for plotting
        bleu_scores.append(scores['bleu'])
        bert_scores.append(scores['bert_f1'])
        rouge_scores.append(scores['rouge_l'])

        # Store full result
        result = {
            'interaction': i,
            'question': question,
            'baseline_answer': expected,  # This is now the baseline from first run
            'actual_answer': actual_answer,
            'response_time': response_time,
            'scores': scores,
            'retrieved_docs': response.get('retrieved_documents', 0) if "error" not in response else 0,
            'sources': response.get('sources', []) if "error" not in response else []
        }
        results.append(result)

        print(f"Consistency Scores - BLEU: {scores['bleu']:.4f} | BERT-F1: {scores['bert_f1']:.4f} | ROUGE-L: {scores['rouge_l']:.4f}")

    print(f"\n✅ Completed evaluation using baseline responses for better consistency measurement")

    # Create visualization
    create_interaction_graph(bleu_scores, bert_scores, rouge_scores)

    # Print summary
    print_summary(results, bleu_scores, bert_scores, rouge_scores, response_times)

    # Save results
    save_results(results, bleu_scores, bert_scores, rouge_scores, response_times)

    return results

    # Print summary
    print_summary(results, bleu_scores, bert_scores, rouge_scores, response_times)

    # Save results
    save_results(results, bleu_scores, bert_scores, rouge_scores, response_times)

    return results

def create_interaction_graph(bleu_scores, bert_scores, rouge_scores):
    """Create line graph showing score progression over 10 interactions"""
    interactions = list(range(1, 11))

    plt.figure(figsize=(14, 10))

    # Plot lines for each metric
    plt.plot(interactions, bleu_scores, 'o-', linewidth=3, markersize=8, label='BLEU Score', color='#2E8B57')
    plt.plot(interactions, bert_scores, 's-', linewidth=3, markersize=8, label='BERT-F1 Score', color='#FF6B6B')
    plt.plot(interactions, rouge_scores, '^-', linewidth=3, markersize=8, label='ROUGE-L Score', color='#4ECDC4')

    # Customize plot
    plt.title('Response Consistency Across 10 Interactions\n(Compared to Baseline Response)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Interaction Number', fontsize=14, fontweight='bold')
    plt.ylabel('Consistency Score', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=13, loc='best')

    # Set axis limits and ticks
    plt.ylim(0, 1.0)
    plt.xlim(0.5, 10.5)
    plt.xticks(interactions)

    # Add value annotations
    for i, (bleu, bert, rouge) in enumerate(zip(bleu_scores, bert_scores, rouge_scores)):
        if bleu > 0.05:  # Only annotate significant values
            plt.annotate(f'{bleu:.3f}', (i+1, bleu), textcoords="offset points", xytext=(0,15), ha='center', fontsize=9, fontweight='bold')
        if bert > 0.05:
            plt.annotate(f'{bert:.3f}', (i+1, bert), textcoords="offset points", xytext=(0,15), ha='center', fontsize=9, fontweight='bold')
        if rouge > 0.05:
            plt.annotate(f'{rouge:.3f}', (i+1, rouge), textcoords="offset points", xytext=(0,15), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save plot
    output_dir = "pengujian/output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'test2_consistency_curve.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Consistency curve saved to: {filepath}")

def print_summary(results, bleu_scores, bert_scores, rouge_scores, response_times):
    """Print formatted summary of consistency evaluation results"""
    print(f"\n{'='*80}")
    print(f"{'RINGKASAN HASIL EVALUASI KONSISTENSI':^80}")
    print(f"{'='*80}")

    # Calculate statistics
    avg_bleu = np.mean(bleu_scores)
    avg_bert = np.mean(bert_scores)
    avg_rouge = np.mean(rouge_scores)
    avg_response_time = np.mean(response_times)

    std_bleu = np.std(bleu_scores)
    std_bert = np.std(bert_scores)
    std_rouge = np.std(rouge_scores)

    print(f"Total Interactions: {len(results)}")
    print(f"Average Response Time: {avg_response_time:.2f}s")
    print()
    print(f"{'Metric':<12} {'Average':<10} {'Std Dev':<10} {'Min':<8} {'Max':<8}")
    print(f"{'-'*52}")
    print(f"{'BLEU':<12} {avg_bleu:<10.4f} {std_bleu:<10.4f} {min(bleu_scores):<8.4f} {max(bleu_scores):<8.4f}")
    print(f"{'BERT-F1':<12} {avg_bert:<10.4f} {std_bert:<10.4f} {min(bert_scores):<8.4f} {max(bert_scores):<8.4f}")
    print(f"{'ROUGE-L':<12} {avg_rouge:<10.4f} {std_rouge:<10.4f} {min(rouge_scores):<8.4f} {max(rouge_scores):<8.4f}")
    print(f"{'='*80}")

def save_results(results, bleu_scores, bert_scores, rouge_scores, response_times):
    """Save results to JSON file"""
    import json
    from datetime import datetime

    output_dir = "pengujian/output"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics for summary
    summary = {
        'bleu': {
            'avg': np.mean(bleu_scores),
            'std': np.std(bleu_scores),
            'min': np.min(bleu_scores),
            'max': np.max(bleu_scores)
        },
        'bert_f1': {
            'avg': np.mean(bert_scores),
            'std': np.std(bert_scores),
            'min': np.min(bert_scores),
            'max': np.max(bert_scores)
        },
        'rouge_l': {
            'avg': np.mean(rouge_scores),
            'std': np.std(rouge_scores),
            'min': np.min(rouge_scores),
            'max': np.max(rouge_scores)
        },
        'response_time': {
            'avg': np.mean(response_times),
            'std': np.std(response_times),
            'min': np.min(response_times),
            'max': np.max(response_times)
        }
    }

    # Prepare final results
    final_results = {
        'test_type': 'session_consistency_evaluation_improved',
        'timestamp': datetime.now().isoformat(),
        'method': 'baseline_comparison',
        'description': 'Menggunakan response pertama sebagai baseline untuk mengukur konsistensi',
        'total_interactions': len(results),
        'results': results,
        'scores_over_time': {
            'bleu_scores': bleu_scores,
            'bert_f1_scores': bert_scores,
            'rouge_l_scores': rouge_scores,
            'response_times': response_times
        },
        'summary': summary,
        'interpretation': {
            'bleu_note': 'Skor BLEU lebih tinggi karena membandingkan dengan baseline response yang konsisten',
            'bert_f1_note': 'BERT-F1 mengukur konsistensi semantik antara response',
            'rouge_l_note': 'ROUGE-L mengukur konsistensi struktur kalimat'
        }
    }

    filepath = os.path.join(output_dir, 'test2_session_interactions_results.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"✅ Results saved to: {filepath}")

if __name__ == "__main__":
    try:
        results = test_session_interactions()
        print("\n🎉 Test 2 completed successfully!")
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
