# Automated Evaluation of Conversational AI

### DA5401 — Kaggle Data Challenge 2025  
**Author:** Basavaraj A Naduvinamani (DA25C005)

## Overview
Automated model to evaluate conversational AI responses and score them between 0 and 10 using prompt-response context and metric vectors. Achieved **2.671 RMSE** on private leaderboard using a **Hybrid Weighted Ensemble**.

## Method
- Combined Prompt + System Prompt + Response using `[SEP]`
- TF-IDF (50k vocab, 1–3 n-grams) + SVD (100 components)
- Text length features  
- Ensemble: MLP (0.50) + Ridge (0.30) + SVR (0.20)
- 5-Fold CV, AdamW optimizer, output clipped to [0, 10]

## Results
| Model | RMSE |
|------|------|
| Final Ensemble | **2.671** |

## Key Takeaways
- LSA features outperform raw transformer embeddings for this dataset
- Ensemble improves robustness on noisy multilingual data
