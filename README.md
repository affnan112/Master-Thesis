# Benchmarking Activation-Based CNN Filter Pruning

**A Comparison of Class-Agnostic and Class-Aware Scoring Strategies**

> **Core Research Question:** Does incorporating class label information into filter scoring produce better structured pruning outcomes than treating all training samples equally?

This repository contains the official codebase for my Master's Thesis in Statistics and Data Science at Hasselt University. It implements a fully deterministic, two-stage structured filter pruning pipeline for Convolutional Neural Networks (CNNs). It benchmarks three activation-based scoring criteria (Average Activation, APoZ, and Entropy) under both **class-agnostic** and **class-aware** paradigms.

---

## 📌 Project Overview

Neural network pruning removes redundant structures from trained models to reduce computational costs without altering the inference pipeline. While most filter pruning methods score importance by averaging activations across an entire dataset (class-agnostic), this project implements a **class-aware** discriminability metric. 

The class-aware criterion scores each filter based on its inter-class discriminability:

$$s_f^{disc} = |s_f^{(0)} - s_f^{(1)}|$$

By testing these strategies on a custom VGG-style binary classifier (Cats vs. Dogs), this pipeline evaluates at what compression thresholds class-awareness actually impacts network accuracy, filter selection stability, and per-class performance balance.

---

## 🚀 Key Findings

* **The "Dead Filter" Phenomenon:** An initial cleaning phase identified massive over-parameterization. Removing permanently inactive filters resulted in a **50.3% parameter reduction** while slightly *improving* test accuracy (96.70% to 97.99%).
* **The Null Result at Standard Compression:** Under standard pruning profiles (up to 70% in the deepest layers), class-aware and class-agnostic methods produce statistically indistinguishable accuracy outcomes. Fine-tuning compensates for differing filter selections when the model retains sufficient capacity.
* **Structured vs. Random Pruning:** Structured activation-based pruning consistently and significantly outperforms random filter removal at high compression ratios.
* **Extreme Compression Thresholds:** In single-layer stress tests (up to 95% removal), the choice of metric becomes critical. Class-aware Entropy scoring failed severely at extreme constraints (16 pp accuracy drop), while Class-aware Average Activation proved the most resilient.
* **100% Determinism:** All activation-based scoring methods tested are fully deterministic across random seeds, ensuring perfect filter selection reproducibility.

---

## 🛠️ Methodology & Pipeline

The project operates on a custom four-block VGG-style CNN trained on a balanced 8,000-image dataset. The compression pipeline follows a strict **Prune $\rightarrow$ Inherit Weights $\rightarrow$ Fine-Tune** sequence.

### Compression Profiles Tested

| Profile | Conv 1 | Conv 2 | Conv 3 | Conv 4 |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | 0% | 5% | 10% | 25% |
| **Medium** | 0% | 10% | 25% | 50% |
| **High** | 0% | 20% | 40% | 70% |

### Pruning Metrics Evaluated
1. **Average Activation:** Ranks filters by mean absolute output volume.
2. **APoZ (Average Percentage of Zeros):** Ranks filters by the lowest frequency of zero-value activations (least "dead").
3. **Activation Entropy:** Ranks filters by the informational richness (Shannon entropy) of their activation distributions.

---

## 💻 Tech Stack

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras 3
* **Data Processing & Math:** NumPy, Pandas
* **Visualization:** Matplotlib

---

