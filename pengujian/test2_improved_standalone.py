#!/usr/bin/env python3
"""
Test 2 Improved: Evaluasi 10 Interaksi dengan Baseline Response yang Akurat
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
    print("Please install: pip install nltk rouge-score bert-score matplotlib seaborn")
    sys.exit(1)

# Import chatbot components
try:
    from src.services.chatbot_service import ChatbotService
    print("✅ Chatbot service imported successfully")
except ImportError as e:
    print(f"❌ Error importing chatbot service: {e}")
    sys.exit(1)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

def generate_baseline_responses():
    """Generate baseline responses from the chatbot system"""
    print("="*70)
    print("GENERATING BASELINE RESPONSES")
    print("="*70)

    # Initialize chatbot
    print("Initializing chatbot service...")
    chatbot_service = ChatbotService()
    chatbot_service.initialize()
    session_id = chatbot_service.create_session()
    print(f"✅ Chatbot initialized with session: {session_id}")

    # Questions to ask
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

    baseline_responses = {}

    print("\n🔄 Generating baseline responses...")
    for i, question in enumerate(questions, 1):
        print(f"Generating baseline {i}/10: {question[:50]}...")

        # Get response from chatbot
        response = chatbot_service.chat(session_id, question)

        if "error" not in response:
            baseline_responses[question] = {
                'response': response['response'],
                'sources': response.get('sources', []),
                'retrieved_docs': len(response.get('documents', [])),
                'response_time': response.get('response_time', 0)
            }
            print(f"  ✅ Generated ({len(response['response'])} chars)")
        else:
            print(f"  ❌ Error: {response['error']}")
            baseline_responses[question] = {
                'response': "Error generating baseline response",
                'sources': [],
                'retrieved_docs': 0,
                'response_time': 0
            }

    # Save baseline responses
    baseline_file = os.path.join(script_dir, "baseline_responses.json")
    with open(baseline_file, 'w', encoding='utf-8') as f:
        json.dump(baseline_responses, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Baseline responses saved to: {baseline_file}")
    return baseline_responses

def load_or_generate_baseline():
    """Load existing baseline or generate new one"""
    baseline_file = os.path.join(script_dir, "baseline_responses.json")

    if os.path.exists(baseline_file):
        print("📂 Loading existing baseline responses...")
        try:
            with open(baseline_file, 'r', encoding='utf-8') as f:
                baseline = json.load(f)
            print(f"✅ Loaded {len(baseline)} baseline responses")
            return baseline
        except Exception as e:
            print(f"❌ Error loading baseline: {e}")
            print("Generating new baseline...")
    else:
        print("📝 No existing baseline found. Generating new baseline...")

    return generate_baseline_responses()

def test_session_consistency():
    """Test evaluasi konsistensi response dalam multiple sessions"""
    print("="*70)
    print("TEST 2 IMPROVED: EVALUASI KONSISTENSI RESPONSE")
    print("="*70)

    # Load or generate baseline
    baseline_responses = load_or_generate_baseline()

    # Initialize chatbot for testing
    print("\nInitializing chatbot for consistency testing...")
    chatbot_service = ChatbotService()
    chatbot_service.initialize()

    # Test multiple sessions
    test_results = []
    questions = list(baseline_responses.keys())

    # Perform multiple runs for consistency testing
    num_runs = 3  # Test 3 times to check consistency
    all_runs_data = []

    for run in range(1, num_runs + 1):
        print(f"\n{'='*50}")
        print(f"CONSISTENCY RUN {run}/{num_runs}")
        print(f"{'='*50}")

        session_id = chatbot_service.create_session()
        run_results = []

        for i, question in enumerate(questions, 1):
            print(f"\n{'-'*40}")
            print(f"INTERACTION {i}/10 (Run {run})")
            print(f"{'-'*40}")

            baseline_response = baseline_responses[question]['response']

            print(f"Question: {question}")

            # Get current response
            start_time = time.time()
            response = chatbot_service.chat(session_id, question)
            response_time = time.time() - start_time

            if "error" not in response:
                current_response = response['response']

                # Calculate consistency scores (comparing with baseline)
                scores = calculate_scores(baseline_response, current_response)

                print(f"Response: {current_response[:100]}...")
                print(f"Consistency Scores - BLEU: {scores['bleu']:.3f}, BERT-F1: {scores['bert_f1']:.3f}, ROUGE-L: {scores['rouge_l']:.3f}")
                print(f"Response Time: {response_time:.2f}s")

                interaction_result = {
                    'run': run,
                    'interaction': i,
                    'question': question,
                    'baseline_response': baseline_response,
                    'current_response': current_response,
                    'response_time': response_time,
                    'scores': scores,
                    'retrieved_docs': len(response.get('documents', [])),
                    'sources': response.get('sources', [])
                }

                run_results.append(interaction_result)

            else:
                print(f"❌ Error: {response['error']}")
                interaction_result = {
                    'run': run,
                    'interaction': i,
                    'question': question,
                    'baseline_response': baseline_response,
                    'current_response': f"Error: {response['error']}",
                    'response_time': response_time,
                    'scores': {'bleu': 0.0, 'bert_f1': 0.0, 'rouge_l': 0.0},
                    'retrieved_docs': 0,
                    'sources': []
                }
                run_results.append(interaction_result)

        all_runs_data.extend(run_results)

    # Calculate statistics across all runs
    print(f"\n{'='*70}")
    print("CALCULATING CONSISTENCY STATISTICS")
    print(f"{'='*70}")

    # Group results by question for consistency analysis
    question_consistency = {}
    for result in all_runs_data:
        question = result['question']
        if question not in question_consistency:
            question_consistency[question] = []
        question_consistency[question].append(result['scores'])

    # Calculate consistency metrics
    consistency_stats = {}
    for question, score_list in question_consistency.items():
        bleu_scores = [s['bleu'] for s in score_list]
        bert_scores = [s['bert_f1'] for s in score_list]
        rouge_scores = [s['rouge_l'] for s in score_list]

        consistency_stats[question] = {
            'bleu': {
                'mean': np.mean(bleu_scores),
                'std': np.std(bleu_scores),
                'min': np.min(bleu_scores),
                'max': np.max(bleu_scores)
            },
            'bert_f1': {
                'mean': np.mean(bert_scores),
                'std': np.std(bert_scores),
                'min': np.min(bert_scores),
                'max': np.max(bert_scores)
            },
            'rouge_l': {
                'mean': np.mean(rouge_scores),
                'std': np.std(rouge_scores),
                'min': np.min(rouge_scores),
                'max': np.max(rouge_scores)
            }
        }

    # Overall consistency metrics
    all_bleu = [r['scores']['bleu'] for r in all_runs_data]
    all_bert = [r['scores']['bert_f1'] for r in all_runs_data]
    all_rouge = [r['scores']['rouge_l'] for r in all_runs_data]

    overall_stats = {
        'bleu': {
            'mean': np.mean(all_bleu),
            'std': np.std(all_bleu),
            'consistency_coefficient': 1 - (np.std(all_bleu) / max(np.mean(all_bleu), 0.001))
        },
        'bert_f1': {
            'mean': np.mean(all_bert),
            'std': np.std(all_bert),
            'consistency_coefficient': 1 - (np.std(all_bert) / max(np.mean(all_bert), 0.001))
        },
        'rouge_l': {
            'mean': np.mean(all_rouge),
            'std': np.std(all_rouge),
            'consistency_coefficient': 1 - (np.std(all_rouge) / max(np.mean(all_rouge), 0.001))
        }
    }

    print(f"\nOverall Consistency Statistics:")
    print(f"BLEU - Mean: {overall_stats['bleu']['mean']:.4f}, Std: {overall_stats['bleu']['std']:.4f}, Consistency: {overall_stats['bleu']['consistency_coefficient']:.4f}")
    print(f"BERT-F1 - Mean: {overall_stats['bert_f1']['mean']:.4f}, Std: {overall_stats['bert_f1']['std']:.4f}, Consistency: {overall_stats['bert_f1']['consistency_coefficient']:.4f}")
    print(f"ROUGE-L - Mean: {overall_stats['rouge_l']['mean']:.4f}, Std: {overall_stats['rouge_l']['std']:.4f}, Consistency: {overall_stats['rouge_l']['consistency_coefficient']:.4f}")

    # Create visualizations
    create_consistency_visualizations(all_runs_data, overall_stats, consistency_stats)

    # Save results
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    final_results = {
        'test_type': 'session_consistency_evaluation',
        'timestamp': datetime.now().isoformat(),
        'num_runs': num_runs,
        'num_questions': len(questions),
        'total_interactions': len(all_runs_data),
        'all_runs_data': all_runs_data,
        'consistency_stats': consistency_stats,
        'overall_stats': overall_stats,
        'execution_time': time.time() - time.time()  # Will be updated
    }

    results_file = os.path.join(output_dir, "test2_improved_consistency_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to: {results_file}")
    return final_results

def create_consistency_visualizations(all_runs_data, overall_stats, consistency_stats):
    """Create visualizations for consistency analysis"""
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Chatbot Response Consistency Analysis', fontsize=16, fontweight='bold')

    # 1. Consistency scores across runs
    runs = [r['run'] for r in all_runs_data]
    interactions = [r['interaction'] for r in all_runs_data]
    bleu_scores = [r['scores']['bleu'] for r in all_runs_data]
    bert_scores = [r['scores']['bert_f1'] for r in all_runs_data]
    rouge_scores = [r['scores']['rouge_l'] for r in all_runs_data]

    # Plot 1: Score trends across interactions
    interaction_means = {}
    for i in range(1, 11):
        interaction_data = [r for r in all_runs_data if r['interaction'] == i]
        if interaction_data:
            interaction_means[i] = {
                'bleu': np.mean([r['scores']['bleu'] for r in interaction_data]),
                'bert_f1': np.mean([r['scores']['bert_f1'] for r in interaction_data]),
                'rouge_l': np.mean([r['scores']['rouge_l'] for r in interaction_data])
            }

    axes[0,0].plot(list(interaction_means.keys()), [v['bleu'] for v in interaction_means.values()],
                   marker='o', label='BLEU', linewidth=2)
    axes[0,0].plot(list(interaction_means.keys()), [v['bert_f1'] for v in interaction_means.values()],
                   marker='s', label='BERT-F1', linewidth=2)
    axes[0,0].plot(list(interaction_means.keys()), [v['rouge_l'] for v in interaction_means.values()],
                   marker='^', label='ROUGE-L', linewidth=2)
    axes[0,0].set_title('Average Consistency Scores by Interaction')
    axes[0,0].set_xlabel('Interaction Number')
    axes[0,0].set_ylabel('Consistency Score')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Plot 2: Score distribution
    scores_data = []
    labels = []
    for score_list, label in [(bleu_scores, 'BLEU'), (bert_scores, 'BERT-F1'), (rouge_scores, 'ROUGE-L')]:
        scores_data.append(score_list)
        labels.append(label)

    axes[0,1].boxplot(scores_data, labels=labels)
    axes[0,1].set_title('Score Distribution Across All Runs')
    axes[0,1].set_ylabel('Score')
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: Consistency coefficient by metric
    consistency_coeffs = [overall_stats['bleu']['consistency_coefficient'],
                         overall_stats['bert_f1']['consistency_coefficient'],
                         overall_stats['rouge_l']['consistency_coefficient']]
    metric_names = ['BLEU', 'BERT-F1', 'ROUGE-L']

    bars = axes[1,0].bar(metric_names, consistency_coeffs, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1,0].set_title('Consistency Coefficient by Metric')
    axes[1,0].set_ylabel('Consistency Coefficient')
    axes[1,0].set_ylim(0, 1)
    axes[1,0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, coeff in zip(bars, consistency_coeffs):
        height = bar.get_height()
        axes[1,0].annotate(f'{coeff:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom')

    # Plot 4: Response time consistency
    response_times = [r['response_time'] for r in all_runs_data]
    run_times = {}
    for r in all_runs_data:
        if r['run'] not in run_times:
            run_times[r['run']] = []
        run_times[r['run']].append(r['response_time'])

    for run, times in run_times.items():
        axes[1,1].plot(range(1, len(times) + 1), times, marker='o', label=f'Run {run}', alpha=0.7)

    axes[1,1].set_title('Response Time Across Interactions')
    axes[1,1].set_xlabel('Interaction Number')
    axes[1,1].set_ylabel('Response Time (seconds)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    consistency_plot_path = os.path.join(output_dir, "test2_improved_consistency_analysis.png")
    plt.savefig(consistency_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Consistency visualization saved to: {consistency_plot_path}")

def main():
    """Main function"""
    start_time = time.time()

    try:
        print("🚀 Starting Test 2 Improved: Consistency Evaluation")
        results = test_session_consistency()

        execution_time = time.time() - start_time
        results['execution_time'] = execution_time

        # Update results file with execution time
        output_dir = os.path.join(script_dir, "output")
        results_file = os.path.join(output_dir, "test2_improved_consistency_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n🎉 Test 2 Improved completed successfully!")
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        print(f"📊 Consistency scores:")
        print(f"   BLEU: {results['overall_stats']['bleu']['mean']:.4f} (±{results['overall_stats']['bleu']['std']:.4f})")
        print(f"   BERT-F1: {results['overall_stats']['bert_f1']['mean']:.4f} (±{results['overall_stats']['bert_f1']['std']:.4f})")
        print(f"   ROUGE-L: {results['overall_stats']['rouge_l']['mean']:.4f} (±{results['overall_stats']['rouge_l']['std']:.4f})")

        return True

    except Exception as e:
        print(f"❌ Test 2 Improved failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
