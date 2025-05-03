from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

class LSTMModelWrapper:
    def __init__(self, total_words, max_seq_len, embedding_matrix=None):
        self.model = self._build_model(total_words, max_seq_len, embedding_matrix)
    
    def _build_model(self, total_words, max_seq_len, embedding_matrix):
        model = Sequential([
            Embedding(input_dim=total_words,
                     output_dim=100,
                     input_length=max_seq_len-1,
                     weights=[embedding_matrix] if embedding_matrix is not None else None,
                     trainable=embedding_matrix is None),
            LSTM(150),
            Dense(total_words, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model