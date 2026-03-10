import word2vec
import word2vec as w2v
import data
import numpy as np
import time # Added for tracking speed


def train(corpus, vocab_size, unigram_table, epochs=5, window_size=2, k_neg=5, dim=10, patience_limit=5, decay=0.5, non_reducable_epoch_count=0):
    model = w2v.Word2VecSGNS(vocab_size, embedding_dim=dim)
    log_interval = 100000 # logging

    best_loss = float('inf')
    patience_counter = 0 #for adaptive lr
    for epoch in range(epochs):
        epoch_loss = 0
        interval_loss = 0  # accumulator for the specific interval
        start_time = time.time() #things for logs really


        for i, target_idx in enumerate(corpus):
            #dynamic window
            actual_window = np.random.randint(1, window_size + 1)
            start = max(0, i - actual_window)
            end = min(len(corpus), i + actual_window + 1)

            for j in range(start, end):
                if i == j: continue #word shouldn't predict itself
                context_idx = corpus[j]

                # Negative sampling from unigram O(1)
                neg_indices = unigram_table[np.random.randint(0, len(unigram_table), size=k_neg)]

                loss = model.train_step(target_idx, context_idx, neg_indices)
                epoch_loss += loss
                interval_loss += loss

            #for logs and reduce lr on plateau
            if i > 0 and i % log_interval == 0:
                avg_interval_loss = interval_loss / log_interval
                percent_done = (i / len(corpus)) * 100
                elapsed_time = time.time() - start_time
                words_per_sec = i / elapsed_time

                if avg_interval_loss < best_loss:
                    best_loss = avg_interval_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > patience_limit and non_reducable_epoch_count >= 0:
                    model.learning_rate *= decay
                    patience_counter = 0
                    print("Adjusting lr on plateau")


                print(f"Epoch {epoch + 1}/{epochs} | Step: {i}/{len(corpus)} ({percent_done:.1f}%) | "
                         f"Speed: {words_per_sec:.0f} words/sec | Loss: {avg_interval_loss:.4f}")

                interval_loss = 0 #reset
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss / len(corpus):.4f}")
    return model



data.download_text8()
text = data.load_dataset("text8") #you might want to change path if it doesn't work
corpus, vocab_size, word2idx, unigram_table = data.prepare_data(text)

print(f"Vocabulary size: {vocab_size}")
print(f"Total words in training corpus (after subsampling): {len(corpus)}")

print("Training:")
trained_model = train(corpus, vocab_size, unigram_table, epochs=1, window_size=8, k_neg=8, dim=150, patience_limit=3, decay=0.5, non_reducable_epoch_count=0)
word2vec.evaluate(trained_model, word2idx, "king")
word2vec.evaluate(trained_model, word2idx, "apple")
word2vec.evaluate(trained_model, word2idx, "bank")
word2vec.evaluate(trained_model, word2idx, "river")
word2vec.evaluate(trained_model, word2idx, "crime")
word2vec.save_model(trained_model, word2idx, "model.txt")