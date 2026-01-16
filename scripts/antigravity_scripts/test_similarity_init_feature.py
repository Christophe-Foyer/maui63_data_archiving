
import torch
import numpy as np
from maui63_data_archiving.detection.core import FewShotDetector

def test_similarity_init():
    print("Initializing detector (this will load the model)...")
    # We load the model just to instantiate the class legally, 
    # but we will bypass compute_embeddings for this specific logic test.
    detector = FewShotDetector()
    
    # Mock data
    dim = 16
    # 5 positive examples
    pos_embeddings = torch.randn(5, dim)
    # Normalize them because real embeddings are normalized
    pos_embeddings = pos_embeddings / pos_embeddings.norm(dim=-1, keepdim=True)
    
    # Initialize
    print("Testing initialize_with_similarity...")
    detector.initialize_with_similarity(pos_embeddings)
    
    # Create query embeddings
    # 1. Exact match to the mean (should change to be close to 1.0)
    # 2. Orthogonal (should be close to 0.0)
    # 3. Opposite (should be close to -1.0)
    
    mean_embed = pos_embeddings.mean(dim=0)
    mean_embed = mean_embed / mean_embed.norm(dim=-1, keepdim=True)
    
    orthogonal = torch.randn(1, dim)
    # Make orthogonal to mean
    orthogonal = orthogonal - (orthogonal @ mean_embed.T) * mean_embed
    orthogonal = orthogonal / orthogonal.norm(dim=-1, keepdim=True)
    
    opposite = -mean_embed
    
    queries = torch.cat([mean_embed.unsqueeze(0), orthogonal, opposite.unsqueeze(0)], dim=0)
    
    # Predict
    scores = detector.predict(queries)
    print(f"Similarity Scores: {scores}")
    
    # Checks
    if not np.allclose(scores[0], 1.0, atol=1e-5):
        print("WARNING: Expected score ~1.0 for mean embedding.")
    else:
        print("Check passed: Mean embedding score is ~1.0")
        
    if not np.abs(scores[1]) < 1e-5:
        print("WARNING: Expected score ~0.0 for orthogonal embedding.")
    else:
        print("Check passed: Orthogonal embedding score is ~0.0")

    if not np.allclose(scores[2], -1.0, atol=1e-5):
        print("WARNING: Expected score ~-1.0 for opposite embedding.")
    else:
        print("Check passed: Opposite embedding score is ~-1.0")

    # Now verify fallback to Classifier after training
    print("\nTesting transition to Logistic Regression...")
    neg_embeddings = torch.randn(5, dim)
    neg_embeddings = neg_embeddings / neg_embeddings.norm(dim=-1, keepdim=True)
    
    # Indices for training (assuming we concatenated pos and neg)
    all_embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
    pos_indices = list(range(5))
    neg_indices = list(range(5, 10))
    
    detector.train(all_embeddings, pos_indices, neg_indices)
    
    probs = detector.predict(queries)
    print(f"Classifier Probabilities: {probs}")
    
    if np.all((probs >= 0) & (probs <= 1)):
        print("Check passed: Probabilities are in [0, 1]")
    else:
        print("ERROR: Probabilities out of range!")

if __name__ == "__main__":
    test_similarity_init()
