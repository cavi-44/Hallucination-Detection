# Hallucination Detector 
### Phase 1 — Semantic Embedding Engine

A specialized implementation of **Word2Vec (Skip-gram with Negative Sampling, SGNS)** designed as the foundational semantic layer for automated hallucination detection.

---

# 1. Project Context

This repository is dedicated to the development of a **Hallucination Detection framework**.

The current implementation completes **Phase 1**, which focuses on constructing a **custom Word2Vec engine from first principles**.

By mapping words into a continuous **100-dimensional vector space**, the system establishes the mathematical foundation required to identify:

- **Out-of-distribution tokens**
- **Semantically inconsistent phrases**
- **Potential hallucinations in AI-generated text**

The embeddings produced by this engine will later serve as the core representation layer for detecting semantic anomalies in generated outputs.

---

# 2. Core Implementation: Word2Vec (SGNS)

The current engine implements **Skip-gram with Negative Sampling (SGNS)** entirely in **NumPy** to ensure full transparency and control over the underlying gradient computations.

SGNS improves training efficiency by learning to distinguish **real context pairs from randomly sampled negative pairs**, rather than computing probabilities across the entire vocabulary. :contentReference[oaicite:0]{index=0}

### Architecture
- **Model:** Skip-gram
- Optimized for **semantic relatedness** rather than syntactic prediction.

### Optimization
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Backpropagation:** Fully derived and implemented manually.

### Learning Dynamics

#### Adaptive Learning Rate
Implements a **Reduce-on-Plateau scheduler** that monitors loss stabilization and dynamically lowers the learning rate when convergence slows.

#### Curriculum Learning
Supports **non-stationary hyperparameters**, including:

- Dynamic context window sizes
- Adaptive negative sample scaling

This enables progressively more complex training regimes as the model stabilizes.

### Data Pipeline

The preprocessing pipeline includes:

- **Subsampling of high-frequency tokens**
- **O(1) negative sampling** using a precomputed unigram table
- **text8 dataset compatibility**

---

# 3. Current Capabilities

The model currently processes the **text8 corpus** to generate high-density semantic embeddings.

These embeddings enable several analytical operations:

### Semantic Similarity Analysis

Measure conceptual similarity using **cosine distance** between word vectors.

Example:

cosine(v_dog, v_cat) > cosine(v_dog, v_car)


### Neighborhood Clustering

Identify clusters of semantically related words that form **coherent contextual regions**.

Example cluster:

king, queen, prince, monarch, crown


### Vector Arithmetic (Analogical Reasoning)

Evaluate embedding quality through vector analogies:

v_king - v_man + v_woman ≈ v_queen


This property emerges when embeddings correctly capture **semantic relationships between words**.

---

# 4. Technical Specifications

| Component | Status | Implementation Detail |
|-----------|--------|-----------------------|
| Data Ingestion | ✅ Completed | Subsampling, Unigram Table, text8 Support |
| SGNS Engine | ✅ Completed | Custom Gradient Descent, Sigmoid Clipping |
| LR Scheduler | ✅ Completed | Adaptive "Reduce on Plateau" |
| Detection Layer | ⏳ In Progress | Cosine Similarity-based Inconsistency Scoring |

---

# 5. Future Roadmap: Hallucination Detection

With the **Word2Vec engine completed**, the next development phase focuses on building the hallucination detection layer.

### Contextual Scoring

Implement a **sliding window analysis** that measures the average semantic coherence of generated sentences.

Concept:

Sentence Semantic Density = mean cosine similarity within context window


Low coherence signals **potential hallucinations or factual inconsistencies**.

### Threshold Calibration

Define **Hallucination Zones**, where semantic similarity to the global sentence context drops below a calibrated threshold.

Example:

if similarity < τ:
flag_token_as_hallucination()

### Evaluation

Benchmark the system using hallucination detection datasets such as:

- **HaluEval**
- Other factual consistency benchmarks

Goal:

- Measure **precision**
- Measure **recall**
- Optimize **semantic anomaly thresholds**

---

# 6. Usage

Train the embedding engine and export the trained model:

```bash
python main.py
```
Sidenote: As it is a version in development, the parameters must be hardcoded in function call.


# 7. Project Status

Phase 1: Semantic Embedding Engine — Completed

Next milestone:

Phase 2 — Semantic Inconsistency Detection

# License

MIT License

# Author

cavi-44

