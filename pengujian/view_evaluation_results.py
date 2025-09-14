#!/usr/bin/env python3
"""
Evaluation Results Viewer - Menampilkan dan menganalisis hasil evaluasi
"""

import os
import json
from datetime import datetime
import pandas as pd

def load_json_results(file_path):
    """Load JSON results file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def format_metrics(metrics):
    """Format metrics for display"""
    if isinstance(metrics, dict):
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"{key}: {value:.4f}")
            else:
                formatted.append(f"{key}: {value}")
        return " | ".join(formatted)
    return str(metrics)

def display_test_results():
    """Display results from all evaluation tests"""

    print("="*80)
    print("CHATBOT EVALUATION RESULTS SUMMARY".center(80))
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    output_dir = "pengujian/output"

    if not os.path.exists(output_dir):
        print("❌ Output directory not found. Please run evaluations first.")
        return False

    files = os.listdir(output_dir)
    json_files = [f for f in files if f.endswith('.json')]
    png_files = [f for f in files if f.endswith('.png')]

    print(f"📁 Output Directory: {os.path.abspath(output_dir)}")
    print(f"📊 Found {len(json_files)} result files and {len(png_files)} visualization files")

    # Test 1 Results
    print("\n" + "="*60)
    print("🔍 TEST 1: TWO QUESTIONS EVALUATION")
    print("="*60)

    test1_file = os.path.join(output_dir, "test1_two_questions_results.json")
    if os.path.exists(test1_file):
        results = load_json_results(test1_file)
        if results:
            print(f"Execution Time: {results.get('execution_time', 'N/A')} seconds")
            print(f"Total Questions: {len(results.get('detailed_results', []))}")

            if 'average_scores' in results:
                avg_scores = results['average_scores']
                print("\n📈 Average Scores:")
                for metric, score in avg_scores.items():
                    if isinstance(score, float):
                        print(f"  {metric.upper()}: {score:.4f}")

            if 'detailed_results' in results:
                print("\n📋 Question Results:")
                for i, result in enumerate(results['detailed_results'], 1):
                    print(f"  Q{i}: {result.get('question', 'N/A')[:50]}...")
                    metrics = result.get('scores', {})
                    print(f"      Scores: {format_metrics(metrics)}")
    else:
        print("❌ Test 1 results not found")

    # Test 2 Results
    print("\n" + "="*60)
    print("📊 TEST 2: SESSION INTERACTIONS EVALUATION")
    print("="*60)

    test2_file = os.path.join(output_dir, "test2_session_interactions_results.json")
    if os.path.exists(test2_file):
        results = load_json_results(test2_file)
        if results:
            print(f"Execution Time: {results.get('execution_time', 'N/A')} seconds")
            print(f"Total Interactions: {results.get('total_interactions', 'N/A')}")

            if 'final_scores' in results:
                final_scores = results['final_scores']
                print("\n📈 Final Session Scores:")
                for metric, score in final_scores.items():
                    if isinstance(score, float):
                        print(f"  {metric.upper()}: {score:.4f}")

            if 'interaction_scores' in results:
                interactions = results['interaction_scores']
                print(f"\n📋 Interaction Progression (showing first 5):")
                for i, interaction in enumerate(interactions[:5], 1):
                    print(f"  Interaction {i}: BERT-F1: {interaction.get('bert_f1', 'N/A'):.4f}")
    else:
        print("❌ Test 2 results not found")

    # Test 3 Results
    print("\n" + "="*60)
    print("🤖 TEST 3: BERT MODEL ACCURACY EVALUATION")
    print("="*60)

    test3_file = os.path.join(output_dir, "test3_bert_accuracy_results.json")
    if os.path.exists(test3_file):
        results = load_json_results(test3_file)
        if results:
            print(f"Execution Time: {results.get('execution_time', 'N/A')} seconds")
            print(f"Total Test Samples: {results.get('total_samples', 'N/A')}")

            if 'overall_metrics' in results:
                metrics = results['overall_metrics']
                print("\n📈 Overall Performance Metrics:")
                print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
                print(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
                print(f"  F1-Score: {metrics.get('f1', 'N/A'):.4f}")

            if 'category_breakdown' in results:
                print("\n📊 Category Performance:")
                breakdown = results['category_breakdown']
                for category, metrics in breakdown.items():
                    if isinstance(metrics, dict) and 'f1-score' in metrics:
                        print(f"  {category}: F1={metrics['f1-score']:.4f}, Support={metrics.get('support', 'N/A')}")
    else:
        print("❌ Test 3 results not found")

    # Visualization Files
    print("\n" + "="*60)
    print("📊 GENERATED VISUALIZATIONS")
    print("="*60)

    if png_files:
        for png_file in sorted(png_files):
            file_path = os.path.join(output_dir, png_file)
            file_size = os.path.getsize(file_path)
            print(f"  📈 {png_file} ({file_size:,} bytes)")
    else:
        print("❌ No visualization files found")

    # Overall Assessment
    print("\n" + "="*60)
    print("🎯 OVERALL SYSTEM ASSESSMENT")
    print("="*60)

    assessment_score = 0
    total_checks = 0

    # Check if all result files exist
    required_files = [
        "test1_two_questions_results.json",
        "test2_session_interactions_results.json",
        "test3_bert_accuracy_results.json"
    ]

    files_exist = 0
    for file in required_files:
        if os.path.exists(os.path.join(output_dir, file)):
            files_exist += 1

    print(f"📋 Evaluation Completeness: {files_exist}/{len(required_files)} tests completed")

    if files_exist == len(required_files):
        print("✅ All evaluations successfully completed")
        print("✅ System is ready for production assessment")
        print("✅ Review individual metrics for detailed insights")
    else:
        print("⚠️  Some evaluations are missing - run complete evaluation suite")

    print("\n" + "="*80)
    print("END OF EVALUATION SUMMARY")
    print("="*80)

    return True

def main():
    """Main function"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        display_test_results()
    except Exception as e:
        print(f"❌ Error displaying results: {e}")
        return False

    return True

if __name__ == "__main__":
    main()
