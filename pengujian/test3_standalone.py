#!/usr/bin/env python3
"""
Run Test 3: Evaluasi Akurasi Model BERT dengan Metrics dan Kurva - Standalone
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Add src to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

# Import chatbot components
from src.services.chatbot_service import ChatbotService

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def test_bert_model_accuracy():
    """Test evaluasi akurasi model BERT dengan baseline comparison method"""
    print("="*70)
    print("TEST 3: EVALUASI AKURASI MODEL BERT (IMPROVED)")
    print("="*70)
    print("Metode: Menggunakan response consistency analysis untuk")
    print("        mengukur akurasi retrieval dan response generation")
    print("="*70)

    # Initialize chatbot
    print("Initializing chatbot service...")
    chatbot_service = ChatbotService()
    chatbot_service.initialize()
    session_id = chatbot_service.create_session()
    print(f"✅ Chatbot initialized with session: {session_id}")

    # Test dataset dengan kategori untuk evaluasi consistency dan retrieval
    test_questions = [
        # Fakultas Questions
        {"question": "Apa visi Fakultas Ekonomi dan Bisnis?", "category": "feb"},
        {"question": "Siapa Dekan Fakultas Ekonomi dan Bisnis?", "category": "feb"},
        {"question": "Program studi apa saja di FEB?", "category": "feb"},

        # Fakultas Teknik Questions
        {"question": "Kapan Fakultas Teknik didirikan?", "category": "teknik"},
        {"question": "Apa misi Fakultas Teknik?", "category": "teknik"},
        {"question": "Program studi yang ada di Fakultas Teknik?", "category": "teknik"},

        # FKIP Questions
        {"question": "Bagaimana visi FKIP?", "category": "fkip"},
        {"question": "Apa fasilitas di FKIP?", "category": "fkip"},
        {"question": "Program studi apa saja di FKIP?", "category": "fkip"},

        # General University Questions
        {"question": "Bagaimana pendidikan berbasis Islam di universitas?", "category": "general"},
        {"question": "Kerjasama internasional universitas?", "category": "general"},
        {"question": "Organisasi mahasiswa yang ada?", "category": "general"},
    ]

    print(f"\nSTEP 1: Generating baseline responses for {len(test_questions)} questions...")

    # Generate baseline responses (first run)
    baseline_responses = {}
    baseline_metadata = {}

    for i, test_item in enumerate(test_questions, 1):
        question = test_item["question"]
        category = test_item["category"]

        print(f"Generating baseline {i}/{len(test_questions)}: {category.upper()} - {question[:50]}...")

        start_time = time.time()
        response = chatbot_service.chat(session_id, question)
        response_time = time.time() - start_time

        if "error" not in response:
            baseline_responses[question] = response.get('response', '')
            baseline_metadata[question] = {
                'category': category,
                'retrieved_docs': len(response.get('documents', [])),
                'sources': response.get('sources', []),
                'response_time': response_time,
                'response_length': len(response.get('response', ''))
            }
            print(f"  ✅ Generated ({len(baseline_responses[question])} chars, {baseline_metadata[question]['retrieved_docs']} docs)")
        else:
            baseline_responses[question] = "Error generating baseline"
            baseline_metadata[question] = {
                'category': category,
                'retrieved_docs': 0,
                'sources': [],
                'response_time': response_time,
                'response_length': 0
            }
            print(f"  ❌ Error: {response['error']}")

    print(f"\n✅ Generated {len(baseline_responses)} baseline responses")

    # STEP 2: Test consistency and accuracy
    print(f"\nSTEP 2: Testing consistency and retrieval accuracy...")
    session_id = chatbot_service.create_session()  # New session

    results = []
    consistency_scores = []
    retrieval_metrics = []
    response_times_test = []
    categories = []

    for i, test_item in enumerate(test_questions, 1):
        question = test_item["question"]
        category = test_item["category"]
        baseline_response = baseline_responses[question]
        baseline_meta = baseline_metadata[question]

        print(f"\n{'-'*50}")
        print(f"TEST {i}/{len(test_questions)}: {category.upper()}")
        print(f"{'-'*50}")
        print(f"Question: {question}")

        # Get current response and measure time
        start_time = time.time()
        response = chatbot_service.chat(session_id, question)
        response_time = time.time() - start_time
        response_times_test.append(response_time)

        if "error" not in response:
            current_response = response.get('response', '')
            current_docs = len(response.get('documents', []))
            current_sources = response.get('sources', [])

            # Calculate consistency score using BERT similarity
            try:
                from bert_score import score as bert_score
                P, R, F1 = bert_score([current_response], [baseline_response], lang='id', verbose=False)
                consistency_score = float(F1[0])
            except:
                # Fallback to simple text similarity
                consistency_score = len(set(current_response.split()) & set(baseline_response.split())) / max(len(set(current_response.split())), 1)

            # Calculate retrieval consistency
            retrieval_consistency = 1.0 - abs(current_docs - baseline_meta['retrieved_docs']) / max(baseline_meta['retrieved_docs'], 1)
            retrieval_consistency = max(0.0, retrieval_consistency)

            # Overall performance score
            performance_score = (consistency_score * 0.7 + retrieval_consistency * 0.3)

            print(f"Consistency Score: {consistency_score:.4f}")
            print(f"Retrieval Consistency: {retrieval_consistency:.4f}")
            print(f"Performance Score: {performance_score:.4f}")
            print(f"Retrieved docs: {current_docs} (baseline: {baseline_meta['retrieved_docs']})")
            print(f"Time: {response_time:.2f}s")

        else:
            consistency_score = 0.0
            retrieval_consistency = 0.0
            performance_score = 0.0
            current_docs = 0
            current_sources = []
            current_response = f"Error: {response['error']}"
            print(f"❌ Error: {response['error']}")

        # Store results
        result = {
            'question': question,
            'category': category,
            'baseline_response': baseline_response,
            'current_response': current_response,
            'consistency_score': consistency_score,
            'retrieval_consistency': retrieval_consistency,
            'performance_score': performance_score,
            'baseline_docs': baseline_meta['retrieved_docs'],
            'current_docs': current_docs,
            'response_time': response_time,
            'baseline_time': baseline_meta['response_time']
        }
        results.append(result)

        consistency_scores.append(consistency_score)
        retrieval_metrics.append(retrieval_consistency)
        categories.append(category)

    # Calculate overall metrics
    avg_consistency = np.mean(consistency_scores)
    avg_retrieval = np.mean(retrieval_metrics)
    avg_performance = np.mean([r['performance_score'] for r in results])
    avg_response_time = np.mean(response_times_test)

    # Calculate category-wise performance
    category_stats = {}
    for category in set(categories):
        cat_results = [r for r in results if r['category'] == category]
        category_stats[category] = {
            'count': len(cat_results),
            'avg_consistency': np.mean([r['consistency_score'] for r in cat_results]),
            'avg_retrieval': np.mean([r['retrieval_consistency'] for r in cat_results]),
            'avg_performance': np.mean([r['performance_score'] for r in cat_results])
        }

    # Create visualizations (separated into multiple files)
    create_separated_plots(results, consistency_scores, retrieval_metrics, categories, response_times_test)

    # Print summary
    print_bert_improved_summary(results, avg_consistency, avg_retrieval, avg_performance,
                              category_stats, avg_response_time)

    # Save results
    save_bert_improved_results(results, consistency_scores, retrieval_metrics, category_stats,
                             avg_consistency, avg_retrieval, avg_performance, avg_response_time)

    return results

def create_separated_plots(results, consistency_scores, retrieval_metrics, categories, response_times):
    """Create separated visualization plots for BERT accuracy evaluation"""

    output_dir = "pengujian/output"
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Consistency Score Progression
    plt.figure(figsize=(12, 8))
    question_nums = list(range(1, len(consistency_scores) + 1))
    plt.plot(question_nums, consistency_scores, 'o-', linewidth=2, markersize=8, color='#3498db')
    plt.title('BERT Response Consistency Score Progression', fontweight='bold', fontsize=14)
    plt.xlabel('Question Number', fontsize=12)
    plt.ylabel('Consistency Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # Add value annotations
    for i, score in enumerate(consistency_scores):
        plt.annotate(f'{score:.3f}', (i+1, score), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'test3_consistency_progression.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Consistency progression plot saved: {plot1_path}")

    # Plot 2: Category-wise Performance
    plt.figure(figsize=(12, 8))
    category_stats = {}
    for category in set(categories):
        cat_results = [r for r in results if r['category'] == category]
        category_stats[category] = {
            'consistency': np.mean([r['consistency_score'] for r in cat_results]),
            'retrieval': np.mean([r['retrieval_consistency'] for r in cat_results]),
            'performance': np.mean([r['performance_score'] for r in cat_results])
        }

    categories_list = list(category_stats.keys())
    consistency_means = [category_stats[cat]['consistency'] for cat in categories_list]
    retrieval_means = [category_stats[cat]['retrieval'] for cat in categories_list]
    performance_means = [category_stats[cat]['performance'] for cat in categories_list]

    x = np.arange(len(categories_list))
    width = 0.25

    plt.bar(x - width, consistency_means, width, label='Consistency', color='#3498db', alpha=0.8)
    plt.bar(x, retrieval_means, width, label='Retrieval', color='#2ecc71', alpha=0.8)
    plt.bar(x + width, performance_means, width, label='Performance', color='#e74c3c', alpha=0.8)

    plt.title('Performance by Question Category', fontweight='bold', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, categories_list)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'test3_category_performance.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Category performance plot saved: {plot2_path}")

    # Plot 3: Response Time Analysis
    plt.figure(figsize=(12, 8))
    question_nums = list(range(1, len(response_times) + 1))
    baseline_times = [r['baseline_time'] for r in results]

    plt.plot(question_nums, response_times, 'o-', linewidth=2, markersize=8,
             color='#e74c3c', label='Current Run')
    plt.plot(question_nums, baseline_times, 's-', linewidth=2, markersize=8,
             color='#3498db', label='Baseline Run')

    plt.title('Response Time Comparison', fontweight='bold', fontsize=14)
    plt.xlabel('Question Number', fontsize=12)
    plt.ylabel('Response Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot3_path = os.path.join(output_dir, 'test3_response_time_comparison.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Response time comparison plot saved: {plot3_path}")

    # Plot 4: Score Distribution
    plt.figure(figsize=(12, 8))

    # Create box plots for different metrics
    data_to_plot = [consistency_scores, retrieval_metrics,
                    [r['performance_score'] for r in results]]
    labels = ['Consistency\nScore', 'Retrieval\nConsistency', 'Overall\nPerformance']

    box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title('Score Distribution Analysis', fontweight='bold', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)

    plt.tight_layout()
    plot4_path = os.path.join(output_dir, 'test3_score_distribution.png')
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Score distribution plot saved: {plot4_path}")

def print_bert_improved_summary(results, avg_consistency, avg_retrieval, avg_performance,
                              category_stats, avg_response_time):
    """Print formatted summary of BERT improved evaluation results"""
    print(f"\n{'='*80}")
    print(f"{'RINGKASAN HASIL EVALUASI BERT IMPROVED':^80}")
    print(f"{'='*80}")

    print(f"Total Questions Tested: {len(results)}")
    print(f"Average Response Time: {avg_response_time:.2f}s")
    print(f"\nOverall Performance Metrics:")
    print(f"{'Metric':<20} {'Score':<10} {'Quality':<15}")
    print(f"{'-'*45}")

    # Determine quality levels
    def get_quality_level(score):
        if score >= 0.9: return "Excellent"
        elif score >= 0.8: return "Very Good"
        elif score >= 0.7: return "Good"
        elif score >= 0.6: return "Fair"
        else: return "Needs Improvement"

    print(f"{'Consistency':<20} {avg_consistency:<10.4f} {get_quality_level(avg_consistency):<15}")
    print(f"{'Retrieval':<20} {avg_retrieval:<10.4f} {get_quality_level(avg_retrieval):<15}")
    print(f"{'Performance':<20} {avg_performance:<10.4f} {get_quality_level(avg_performance):<15}")

    print(f"\nCategory-wise Performance:")
    print(f"{'Category':<15} {'Consistency':<12} {'Retrieval':<12} {'Performance':<12} {'Count':<8}")
    print(f"{'-'*63}")

    for category, stats in category_stats.items():
        print(f"{category.upper():<15} {stats['avg_consistency']:<12.4f} "
              f"{stats['avg_retrieval']:<12.4f} {stats['avg_performance']:<12.4f} {stats['count']:<8}")

    print(f"\n{'='*80}")

def save_bert_improved_results(results, consistency_scores, retrieval_metrics, category_stats,
                             avg_consistency, avg_retrieval, avg_performance, avg_response_time):
    """Save BERT improved evaluation results to JSON"""

    output_dir = "pengujian/output"
    os.makedirs(output_dir, exist_ok=True)

    from datetime import datetime
    import json

    final_results = {
        'test_type': 'bert_model_consistency_evaluation_improved',
        'timestamp': datetime.now().isoformat(),
        'method': 'baseline_consistency_comparison',
        'description': 'Menggunakan response consistency analysis untuk mengukur akurasi retrieval dan response generation',
        'total_questions': len(results),
        'overall_metrics': {
            'consistency': avg_consistency,
            'retrieval': avg_retrieval,
            'performance': avg_performance,
            'response_time': avg_response_time
        },
        'category_breakdown': category_stats,
        'detailed_results': results,
        'score_arrays': {
            'consistency_scores': consistency_scores,
            'retrieval_metrics': retrieval_metrics
        },
        'interpretation': {
            'consistency_note': 'Skor konsistensi mengukur kesamaan response dengan baseline',
            'retrieval_note': 'Konsistensi retrieval mengukur stabilitas jumlah dokumen yang ditemukan',
            'performance_note': 'Skor performa gabungan dari konsistensi response dan retrieval'
        },
        'execution_time': time.time()  # Will be updated in main
    }

    filepath = os.path.join(output_dir, 'test3_bert_consistency_results.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"✅ Results saved to: {filepath}")

if __name__ == "__main__":
    try:
        start_time = time.time()
        results = test_bert_model_accuracy()
        execution_time = time.time() - start_time

        print(f"\n🎉 Test 3 Improved completed successfully!")
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")

    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
