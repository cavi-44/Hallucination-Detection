import numpy as np
#Based on theory presentent in documents:
#Paper #1: "Efficient Estimation of Word Representations in Vector Space" - Mikolow, Chen, Corrado, Dean 2013
#Paper #2: "Distributed Representations of Words and Phrases and their Compositionality" - Mikolow,  Sutskever, Chen, Corrado, Dean 2013
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

        #we add a really small value to prevent model from achieving fe. -inf on log(0)
        eps = 1e-10

        #Calculating Loss
        #Original formula (Paper #2) maximizes likelihood, but we will minimize loss (essentially the same)
        #We also add epsilon for stability and use formula: sigmoid(-x) == 1 - sigmoid(x)
        loss = -np.log(sigmoid_tc + eps) - np.sum(np.log(1 - sigmoid_tn + eps))


        #Computing gradients:
        #derivatives:
        der_tc = 1.0 - sigmoid_tc
        der_tn = sigmoid_tn

        grad_context = der_tc * v_target #shape: (embedding) - gradient for positive context embeddings
        grad_negative = np.outer(der_tn, v_target) #shape: (K, embedding) - gradients (each for K rows) for negative context embeddings
        grad_target = der_tc * v_context + np.dot(der_tn, v_negative)  #shape: (embedding) - gradient for target word
                                                                        #sum of errors of respective negative samples

        self.target_word_embeddings[target_idx] -= self.learning_rate * grad_target
        self.context_word_embeddings[context_idx] -= self.learning_rate * grad_context
        self.context_word_embeddings[negative_indices] -= self.learning_rate * grad_negative

        return loss

