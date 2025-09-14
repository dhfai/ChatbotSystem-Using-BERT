#!/usr/bin/env python3
"""
Evaluasi Model BERT dengan Classification Metrics (F1, Precision, Recall, Accuracy)
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from datetime import datetime

# Add src to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

# Import chatbot components
from src.services.chatbot_service import ChatbotService

# Configure matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def evaluate_bert_classification_metrics():
    """
    Evaluasi Model BERT dengan metrics klasifikasi lengkap
    """
    print("="*80)
    print("EVALUASI MODEL BERT - CLASSIFICATION METRICS")
    print("="*80)
    print("Metrics: F1 Score, Precision, Recall, Accuracy, Confusion Matrix")
    print("="*80)

    # Initialize chatbot
    print("Initializing chatbot service...")
    chatbot_service = ChatbotService()
    chatbot_service.initialize()
    session_id = chatbot_service.create_session()
    print(f"✅ Chatbot initialized with session: {session_id}")

    # Test dataset dengan ground truth labels
    test_data = [
        # Fakultas Ekonomi dan Bisnis (FEB) - Label: 0
        {"question": "Apa visi Fakultas Ekonomi dan Bisnis?", "true_category": "feb", "label": 0},
        {"question": "Siapa Dekan Fakultas Ekonomi dan Bisnis?", "true_category": "feb", "label": 0},
        {"question": "Program studi apa saja di FEB?", "true_category": "feb", "label": 0},
        {"question": "Biaya kuliah di FEB berapa?", "true_category": "feb", "label": 0},
        {"question": "Akreditasi program studi Manajemen FEB?", "true_category": "feb", "label": 0},

        # Fakultas Teknik (FT) - Label: 1
        {"question": "Kapan Fakultas Teknik didirikan?", "true_category": "teknik", "label": 1},
        {"question": "Apa misi Fakultas Teknik?", "true_category": "teknik", "label": 1},
        {"question": "Program studi yang ada di Fakultas Teknik?", "true_category": "teknik", "label": 1},
        {"question": "Fasilitas laboratorium di Fakultas Teknik?", "true_category": "teknik", "label": 1},
        {"question": "Dosen Fakultas Teknik yang berprestasi?", "true_category": "teknik", "label": 1},

        # Fakultas Keguruan dan Ilmu Pendidikan (FKIP) - Label: 2
        {"question": "Bagaimana visi FKIP?", "true_category": "fkip", "label": 2},
        {"question": "Apa fasilitas di FKIP?", "true_category": "fkip", "label": 2},
        {"question": "Program studi apa saja di FKIP?", "true_category": "fkip", "label": 2},
        {"question": "Syarat masuk program PGSD di FKIP?", "true_category": "fkip", "label": 2},
        {"question": "Organisasi mahasiswa di FKIP apa saja?", "true_category": "fkip", "label": 2},

        # General University Questions - Label: 3
        {"question": "Bagaimana pendidikan berbasis Islam di universitas?", "true_category": "general", "label": 3},
        {"question": "Kerjasama internasional universitas?", "true_category": "general", "label": 3},
        {"question": "Sejarah berdirinya Universitas Muhammadiyah Makassar?", "true_category": "general", "label": 3},
        {"question": "Cara mendaftar di universitas?", "true_category": "general", "label": 3},
        {"question": "Beasiswa yang tersedia di universitas?", "true_category": "general", "label": 3},
    ]

    # Category mapping
    category_mapping = {
        0: "feb",
        1: "teknik",
        2: "fkip",
        3: "general"
    }

    category_names = list(category_mapping.values())
    print(f"\nCategories: {category_names}")
    print(f"Total test questions: {len(test_data)}")

    # Collect responses and predictions
    print(f"\nSTEP 1: Collecting responses and making predictions...")

    true_labels = []
    predicted_labels = []
    response_details = []
    response_times = []

    for i, test_item in enumerate(test_data, 1):
        question = test_item["question"]
        true_category = test_item["true_category"]
        true_label = test_item["label"]

        print(f"\n[{i}/{len(test_data)}] Processing: {true_category.upper()}")
        print(f"Question: {question[:60]}...")

        # Get chatbot response
        start_time = time.time()
        response = chatbot_service.chat(session_id, question)
        response_time = time.time() - start_time
        response_times.append(response_time)

        if "error" not in response:
            response_text = response.get('response', '')
            retrieved_docs = response.get('documents', [])
            sources = response.get('sources', [])

            # Predict category based on response analysis
            predicted_label = predict_category_from_response(
                response_text, retrieved_docs, sources
            )

            predicted_category = category_mapping[predicted_label]

            print(f"  True: {true_category} | Predicted: {predicted_category}")
            print(f"  Retrieved docs: {len(retrieved_docs)} | Time: {response_time:.2f}s")

        else:
            response_text = f"Error: {response['error']}"
            predicted_label = 3  # Default to general
            retrieved_docs = []
            sources = []
            print(f"  ❌ Error: {response['error']}")

        # Store results
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

        response_details.append({
            'question': question,
            'true_category': true_category,
            'true_label': true_label,
            'predicted_category': category_mapping[predicted_label],
            'predicted_label': predicted_label,
            'response': response_text,
            'retrieved_docs_count': len(retrieved_docs),
            'sources_count': len(sources),
            'response_time': response_time,
            'correct': true_label == predicted_label
        })

    # Calculate classification metrics
    print(f"\nSTEP 2: Calculating classification metrics...")

    # Overall metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Classification report
    class_report = classification_report(
        true_labels, predicted_labels,
        target_names=category_names,
        output_dict=True,
        zero_division=0
    )

    print("✅ Classification metrics calculated")

    # Create results dictionary
    results = {
        'test_type': 'bert_classification_metrics_evaluation',
        'timestamp': datetime.now().isoformat(),
        'description': 'Evaluasi Model BERT dengan Classification Metrics (F1, Precision, Recall, Accuracy)',
        'total_questions': len(test_data),
        'categories': category_names,
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'avg_response_time': float(np.mean(response_times))
        },
        'per_class_metrics': {
            'precision': [float(p) for p in precision_per_class],
            'recall': [float(r) for r in recall_per_class],
            'f1_score': [float(f) for f in f1_per_class]
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'detailed_responses': response_details,
        'correct_predictions': sum(1 for r in response_details if r['correct']),
        'incorrect_predictions': sum(1 for r in response_details if not r['correct'])
    }

    # Print summary
    print_classification_summary(results)

    # Create visualizations
    create_classification_visualizations(results)

    # Save results
    save_classification_results(results)

    return results

def predict_category_from_response(response_text, retrieved_docs, sources):
    """
    Predict category based on response content analysis
    """
    response_lower = response_text.lower()

    # Category keywords
    feb_keywords = ['ekonomi', 'bisnis', 'feb', 'manajemen', 'akuntansi', 'dekan', 'keuangan', 'perbankan']
    teknik_keywords = ['teknik', 'ft', 'teknologi', 'listrik', 'telekomunikasi', 'komputer', 'iot', 'laboratorium teknik']
    fkip_keywords = ['fkip', 'pendidikan', 'keguruan', 'guru', 'pgsd', 'paud', 'matematika', 'bahasa', 'fisika', 'biologi']

    # Count keyword matches
    feb_count = sum(1 for word in feb_keywords if word in response_lower)
    teknik_count = sum(1 for word in teknik_keywords if word in response_lower)
    fkip_count = sum(1 for word in fkip_keywords if word in response_lower)

    # Determine category based on highest count
    counts = [feb_count, teknik_count, fkip_count]
    max_count = max(counts)

    if max_count == 0:
        return 3  # general
    else:
        return counts.index(max_count)  # Return index of highest count

def print_classification_summary(results):
    """Print formatted classification metrics summary"""
    print(f"\n{'='*80}")
    print(f"{'CLASSIFICATION METRICS SUMMARY':^80}")
    print(f"{'='*80}")

    overall = results['overall_metrics']

    print(f"Total Questions: {results['total_questions']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Incorrect Predictions: {results['incorrect_predictions']}")
    print(f"Average Response Time: {overall['avg_response_time']:.2f}s")

    print(f"\nOVERALL METRICS:")
    print(f"{'Metric':<15} {'Score':<10} {'Quality'}")
    print(f"{'-'*35}")
    print(f"{'Accuracy':<15} {overall['accuracy']:<10.4f} {get_quality_level(overall['accuracy'])}")
    print(f"{'Precision':<15} {overall['precision_macro']:<10.4f} {get_quality_level(overall['precision_macro'])}")
    print(f"{'Recall':<15} {overall['recall_macro']:<10.4f} {get_quality_level(overall['recall_macro'])}")
    print(f"{'F1-Score':<15} {overall['f1_macro']:<10.4f} {get_quality_level(overall['f1_macro'])}")

    print(f"\nPER-CLASS METRICS:")
    print(f"{'Category':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print(f"{'-'*52}")

    categories = results['categories']
    per_class = results['per_class_metrics']

    for i, category in enumerate(categories):
        precision = per_class['precision'][i] if i < len(per_class['precision']) else 0.0
        recall = per_class['recall'][i] if i < len(per_class['recall']) else 0.0
        f1 = per_class['f1_score'][i] if i < len(per_class['f1_score']) else 0.0

        print(f"{category.upper():<12} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")

    print(f"\n{'='*80}")

def get_quality_level(score):
    """Get quality level based on score"""
    if score >= 0.9: return "Excellent"
    elif score >= 0.8: return "Very Good"
    elif score >= 0.7: return "Good"
    elif score >= 0.6: return "Fair"
    else: return "Needs Improvement"

def create_classification_visualizations(results):
    """Create visualization plots for classification metrics"""
    output_dir = "pengujian/output"
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = np.array(results['confusion_matrix'])
    categories = results['categories']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[c.upper() for c in categories],
                yticklabels=[c.upper() for c in categories])

    plt.title('Confusion Matrix - BERT Classification', fontweight='bold', fontsize=14)
    plt.ylabel('True Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'bert_confusion_matrix.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {plot1_path}")

    # Plot 2: Per-class Metrics Comparison
    plt.figure(figsize=(12, 8))

    categories = results['categories']
    per_class = results['per_class_metrics']

    x = np.arange(len(categories))
    width = 0.25

    plt.bar(x - width, per_class['precision'], width, label='Precision', alpha=0.8, color='#3498db')
    plt.bar(x, per_class['recall'], width, label='Recall', alpha=0.8, color='#2ecc71')
    plt.bar(x + width, per_class['f1_score'], width, label='F1-Score', alpha=0.8, color='#e74c3c')

    plt.title('Per-Class Classification Metrics', fontweight='bold', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, [c.upper() for c in categories])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.1)

    # Add value labels on bars
    for i, category in enumerate(categories):
        plt.text(i-width, per_class['precision'][i]+0.01, f'{per_class["precision"][i]:.3f}',
                ha='center', va='bottom', fontsize=9)
        plt.text(i, per_class['recall'][i]+0.01, f'{per_class["recall"][i]:.3f}',
                ha='center', va='bottom', fontsize=9)
        plt.text(i+width, per_class['f1_score'][i]+0.01, f'{per_class["f1_score"][i]:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'bert_per_class_metrics.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Per-class metrics saved: {plot2_path}")

    # Plot 3: Overall Metrics Comparison
    plt.figure(figsize=(10, 6))

    overall = results['overall_metrics']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [overall['accuracy'], overall['precision_macro'],
              overall['recall_macro'], overall['f1_macro']]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    bars = plt.bar(metrics, scores, color=colors, alpha=0.8)

    plt.title('Overall Classification Metrics', fontweight='bold', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plot3_path = os.path.join(output_dir, 'bert_overall_metrics.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Overall metrics saved: {plot3_path}")

def save_classification_results(results):
    """Save classification results to JSON"""
    output_dir = "pengujian/output"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, 'bert_classification_metrics_results.json')

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Classification results saved to: {filepath}")

if __name__ == "__main__":
    try:
        start_time = time.time()
        print("Starting BERT Classification Metrics Evaluation...")

        results = evaluate_bert_classification_metrics()

        execution_time = time.time() - start_time
        print(f"\n🎉 BERT Classification Evaluation completed successfully!")
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
