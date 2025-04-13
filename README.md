# 🕵️‍♂️ Synthetic Fraud Demo

**The Identity That Never Existed: Exposing Synthetic Fraud with Adversarial AI**

This interactive demo shows how synthetic identities can bypass traditional fraud models—and how adversarial AI can retrain those models to be more robust. Built by Caroline Keddie for demonstrating AI assurance to financial institutions.

---

## 🎯 What It Does

- Trains a **GAN (Generative Adversarial Network)** to create synthetic identity features (e.g., credit score & age)
- Uses a **baseline XGBoost fraud model** to detect synthetic vs real identities
- Shows how synthetic identities **slip through undetected**
- Retrains the model with synthetic data labeled as fraud
- Shows improved performance in detecting adversarial examples
- Visualizes the improvement in a **bar chart** comparing accuracy

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-org/advai-synthetic-fraud-demo.git
cd advai-synthetic-fraud-demo
