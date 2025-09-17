"""
TEST 1: BERT Model Performance Test
==================================
Simple test to check BERT model accuracy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.bert_embedder import BERTEmbedder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

def test_bert_model():
    """Test BERT model with basic similarity calculations"""
    print("="*60)
    print("🧠 BERT MODEL PERFORMANCE TEST")
    print("="*60)

    # Initialize BERT
    print("Loading BERT model...")
    embedder = BERTEmbedder()

    # Test questions and expected similar responses
    test_pairs = [
        ("Apa itu jurusan Teknik Informatika?",
         "Teknik Informatika adalah program studi yang mempelajari teknologi komputer"),
        ("Berapa biaya kuliah di Unismuh?",
         "Biaya kuliah di Universitas Muhammadiyah Makassar terjangkau"),
        ("Bagaimana cara daftar kuliah?",
         "Pendaftaran kuliah dapat dilakukan secara online"),
        ("Apa saja jurusan di fakultas teknik?",
         "Fakultas Teknik memiliki berbagai program studi teknologi"),
        ("Dimana lokasi kampus Unismuh?",
         "Kampus Universitas Muhammadiyah Makassar terletak di Makassar")
    ]

    similarities = []

    print(f"\\nTesting {len(test_pairs)} question pairs...")
    print("-"*60)

    for i, (q1, q2) in enumerate(test_pairs, 1):
        # Get embeddings
        emb1 = embedder.encode_query(q1).reshape(1, -1)
        emb2 = embedder.encode_query(q2).reshape(1, -1)

        # Calculate similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        similarities.append(similarity)

        print(f"Test {i}: {similarity:.1%}")
        print(f"  Q1: {q1[:50]}...")
        print(f"  Q2: {q2[:50]}...")
        print()

    # Results
    avg_similarity = np.mean(similarities)

    print("="*60)
    print("📊 BERT MODEL RESULTS:")
    print(f"Average Similarity: {avg_similarity:.1%}")
    print(f"Model Status: {'✅ GOOD' if avg_similarity >= 0.7 else '⚠️  NEEDS IMPROVEMENT' if avg_similarity >= 0.5 else '❌ POOR'}")
    print("="*60)

    # Save results
    results = {
        'bert_model_accuracy': f"{avg_similarity:.1%}",
        'individual_scores': [f"{s:.1%}" for s in similarities],
        'status': 'GOOD' if avg_similarity >= 0.7 else 'MODERATE' if avg_similarity >= 0.5 else 'POOR'
    }

    with open('bert_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return avg_similarity

if __name__ == "__main__":
    test_bert_model()
