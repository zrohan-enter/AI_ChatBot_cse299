from typing import List, Optional
from transformers import pipeline
from scipy.spatial.distance import cosine

class NLPProcessor:
    """Handles NLP-based question matching using sentence embeddings."""
   
    def __init__(self):
        self.nlp = pipeline("feature-extraction", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")

    def find_best_match(self, user_question: str, questions: List[str]) -> Optional[str]:
        """Find the best matching question using sentence embeddings and cosine similarity."""
        try:
            user_embedding = self.nlp(user_question)[0][0]  # Get embedding for user question
            best_score, best_question = 0, None
            for q in questions:
                q_embedding = self.nlp(q)[0][0]  # Get embedding for each question
                score = 1 - cosine(user_embedding, q_embedding)
                if score > best_score and score > 0.85:  # Threshold for similarity
                    best_score, best_question = score, q
            print(f"Debug: Best match for '{user_question}': '{best_question}' with score {best_score}")
            return best_question
        except Exception as e:
            print(f"Bot: Error in NLP matching: {e}")
            return None