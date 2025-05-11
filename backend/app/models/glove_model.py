import numpy as np
import pickle
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
class GloveEmbeddingHandler:
    def __init__(self, embedding_path: str):
        self.embeddings = {}
        self.vocab=[]
        self.embedding_dim = 100
        self.loaded_embeddings(embedding_path)
    def load_embeddings(self, embedding_path: str):
        """Load Glove embeddings from file"""
        print(f"Loading GloVe embeddings from {embedding_path}")
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings[word] = vector
                self.vocab.append(word)
        print(f"Loaded {len(self.embeddings)} word embeddings")
    def get_embedding(self, word: str)-> np.ndarray:
        """Get embedding for a single word"""
        return self.embeddings.get(word.lower(), np.zeros(self.embedding_dim))
    def create_embedding_matrix(self, tokenizer) -> np.ndarray:
        """ Creating embedding matrix for use in neural network"""
        vocab_size = len(tokenizer.word_index) + 1
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        for word, i in tokenizer.word_index.items():
            embedding_vector = self.embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        return embedding_matrix
    def find_similar_words(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Find words most similar to the given word using cosine similarity"""
        if word.lower() not in self.embeddings:
            return []
        
        word_vector = self.embeddings[word.lower()].reshape(1, -1)
        similarities = []
        
        for vocab_word, vector in self.embeddings.items():
            if vocab_word == word.lower():
                continue
            
            sim = cosine_similarity(word_vector, vector.reshape(1, -1))[0][0]
            similarities.append((vocab_word, sim))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def get_average_embedding(self, text: str) -> np.ndarray:
        """Get average embedding for a text (sentence or paragraph)"""
        words = text.lower().split()
        embeddings = []
        
        for word in words:
            if word in self.embeddings:
                embeddings.append(self.embeddings[word])
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using average embeddings"""
        emb1 = self.get_average_embedding(text1).reshape(1, -1)
        emb2 = self.get_average_embedding(text2).reshape(1, -1)
        
        return cosine_similarity(emb1, emb2)[0][0]
    
    def save_embedding_matrix(self, matrix: np.ndarray, filepath: str):
        """Save embedding matrix for later use"""
        np.save(filepath, matrix)
    
    def load_embedding_matrix(self, filepath: str) -> np.ndarray:
        """Load saved embedding matrix"""
        return np.load(filepath)