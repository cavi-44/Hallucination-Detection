import numpy as np
#Based on theory presented in documents:
#Paper #1: "Efficient Estimation of Word Representations in Vector Space" - Mikolov, Chen, Corrado, Dean 2013
#Paper #2: "Distributed Representations of Words and Phrases and their Compositionality" - Mikolov,  Sutskever, Chen, Corrado, Dean 2013


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))
class Word2VecSGNS:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        #division by emdedding_dim to stabilize training (avoiding death gradients of sigmoid function)
        self.target_word_embeddings = np.random.uniform(-1, 1, (self.vocab_size, self.embedding_dim)) / embedding_dim

        #the Output (initially all zeros)
        self.context_word_embeddings = np.zeros((self.vocab_size, self.embedding_dim))

    def train_step(self, target_idx, context_idx, negative_indices):
        v_target = self.target_word_embeddings[target_idx] #shape: (embedding_dim)
        v_context = self.context_word_embeddings[context_idx] #shape: (embedding_dim)
        #pulling multiple negative samples since negative_indices is a vector of size K
        v_negative = self.context_word_embeddings[negative_indices] #shape: (K, embedding_dim)

        #dot products and sigmoid
        dot_tc = np.dot(v_target, v_context)
        dot_tn = np.dot(v_negative, v_target)

        sigmoid_tc = sigmoid(dot_tc)
        sigmoid_tn = sigmoid(dot_tn)

        #we add a tiny value to prevent model from achieving fe. -inf on log(0)
        eps = 1e-10

        #Calculating Loss
        #Original formula (Paper #2) maximizes likelihood, but we will minimize loss (essentially the same)
        #We also add epsilon for stability and use formula: sigmoid(-x) == 1 - sigmoid(x)
        loss = -np.log(sigmoid_tc + eps) - np.sum(np.log(1 - sigmoid_tn + eps))


        #Computing gradients:
        #derivatives:
        der_tc = sigmoid_tc - 1.0
        der_tn = sigmoid_tn

        grad_context = der_tc * v_target #shape: (embedding) - gradient for positive context embeddings
        grad_negative = np.outer(der_tn, v_target) #shape: (K, embedding) - gradients (each for K rows) for negative context embeddings
        grad_target = der_tc * v_context + np.dot(der_tn, v_negative)  #shape: (embedding) - gradient for target word
                                                                        #sum of errors of respective negative samples

        self.target_word_embeddings[target_idx] -= self.learning_rate * grad_target
        self.context_word_embeddings[context_idx] -= self.learning_rate * grad_context
        self.context_word_embeddings[negative_indices] -= self.learning_rate * grad_negative

        return loss



#ELEMENT COPY-PASTED FROM AI - START
def evaluate(model, word2idx, target_word, top_k=10):
    # 1. Get the ID of the word we want to test
    if target_word not in word2idx:
        print(f"'{target_word}' not found in vocabulary.")
        return

    idx2word = {i: w for w, i in word2idx.items()}
    target_id = word2idx[target_word]

    # 2. Extract the vector for our target word
    # Shape: (dim,)
    target_vec = model.target_word_embeddings[target_id]

    # 3. Get ALL vectors to compare against
    # Shape: (vocab_size, dim)
    all_vectors = model.target_word_embeddings

    # 4. Calculate Cosine Similarity: (A · B) / (||A|| * ||B||)
    # This is the "Confidence" score we discussed
    dot_product = np.dot(all_vectors, target_vec)
    norms = np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(target_vec)
    similarity = dot_product / (norms + 1e-9)  # Add epsilon to avoid divide by zero

    # 5. Sort by highest similarity and skip the first one (which is the word itself)
    nearest_indices = np.argsort(similarity)[::-1][1:top_k + 1]

    print(f"\nWords most similar to '{target_word}':")
    for idx in nearest_indices:
        print(f" -> {idx2word[idx]}: {similarity[idx]:.4f}")
def save_model(model, word2idx, file_name="word2vec.txt"):
    print(f"Saving model {file_name}")

    #extracting weights
    embeddings = model.target_word_embeddings
    vocab_size, dim = embeddings.shape

    idx2word = {i: w for w, i in word2idx.items()}

    with open(file_name, 'w', encoding='utf-8') as f:
        # first line of word2vec: vocab_size dimension
        f.write(f"{vocab_size} {dim}\n")
        for i in range(vocab_size): #writting each word with its vector
            word = idx2word[i]
            vector_str = " ".join(map(str, embeddings[i]))
            f.write(f"{word} {vector_str}\n")

    print("Model saved successfully in standard Word2Vec format.")
#ELEMENT COPY-PASTED FROM AI - END