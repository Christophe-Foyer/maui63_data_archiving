
import sys
import os
import torch
from transformers import AutoModel

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from maui63_data_archiving.ml.classification import DinoV3Classifier

def inspect():
    try:
        classifier = DinoV3Classifier(device="cpu")
        print(classifier)
        print("\nModel structure:")
        print(classifier.model)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
