import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# GAN CONFIGURATION
# ----------------------------

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_gan(epochs=1000, latent_dim=10, output_dim=2):
    generator = Generator(latent_dim, output_dim)
    discriminator = Discriminator(output_dim)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

    real_data = torch.tensor(np.random.normal(0, 1, (1000, output_dim)), dtype=torch.float32)

    for _ in range(epochs):
        z = torch.randn(64, latent_dim)
        fake_data = generator(z)

        real_labels = torch.ones(64, 1)
        fake_labels = torch.zeros(64, 1)

        real_loss = criterion(discriminator(real_data[:64]), real_labels)
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        z = torch.randn(64, latent_dim)
        fake_data = generator(z)
        g_loss = criterion(discriminator(fake_data), real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    return generator

def train_xgboost(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("Advai: Synthetic Identity Fraud Demo")

st.sidebar.header("Controls")
latent_dim = st.sidebar.slider("Latent Dim (GAN)", 2, 20, 10)
samples_to_generate = st.sidebar.slider("Synthetic IDs to Generate", 50, 500, 100)
evaluate_model = st.sidebar.checkbox("Run Evaluation on Synthetic Identities")
show_comparison = st.sidebar.checkbox("Show Real vs Synthetic Comparison")
retrain_with_synthetic = st.sidebar.checkbox("Retrain Model with Synthetic IDs (Marked as Fraud)")

# Generate synthetic data
generator = train_gan(latent_dim=latent_dim)
z = torch.randn(samples_to_generate, latent_dim)
generated_data = generator(z).detach().numpy()
df_gen = pd.DataFrame(generated_data, columns=["credit score", "age"])

st.subheader("Generated Synthetic Identities")
st.dataframe(df_gen.head())

fig, ax = plt.subplots()
ax.scatter(df_gen["credit score"], df_gen["age"], alpha=0.6, label="Synthetic IDs", color='orange')
ax.set_title("GAN Generated Synthetic Identity Features")
ax.set_xlabel("Credit Score")
ax.set_ylabel("Age")
st.pyplot(fig)

# Real data
X_real = np.random.randn(1000, 2)
y_real = np.random.binomial(1, 0.1, 1000)

# Show comparison
if show_comparison:
    df_real = pd.DataFrame(X_real, columns=["credit score", "age"])
    fig2, ax2 = plt.subplots()
    ax2.scatter(df_real["credit score"], df_real["age"], alpha=0.4, label="Real IDs", color='blue')
    ax2.scatter(df_gen["credit score"], df_gen["age"], alpha=0.6, label="Synthetic IDs", color='orange')
    ax2.set_title("Real vs Synthetic Identity Feature Space")
    ax2.set_xlabel("Credit Score")
    ax2.set_ylabel("Age")
    ax2.legend()
    st.pyplot(fig2)

# Model Evaluation
if evaluate_model:
    st.subheader("XGBoost Model Evaluation")

    # Baseline model
    baseline_model = train_xgboost(X_real, y_real)
    preds_fake_baseline = baseline_model.predict(df_gen.values)
    y_pred_real_baseline = baseline_model.predict(X_real)

    acc_real_before = accuracy_score(y_real, y_pred_real_baseline)
    acc_synth_before = accuracy_score(np.ones(len(df_gen)), preds_fake_baseline)

    st.write("❌ Synthetic IDs not flagged by baseline model:")
    st.dataframe(df_gen[preds_fake_baseline == 0].head())

    st.write("📊 Baseline Report (on real data):")
    st.json(classification_report(y_real, y_pred_real_baseline, output_dict=True))

    if retrain_with_synthetic:
        st.subheader("Retrained Model (Synthetic IDs = Fraud)")

        X_combined = np.vstack([X_real, df_gen.values])
        y_combined = np.concatenate([y_real, np.ones(len(df_gen))])

        retrained_model = train_xgboost(X_combined, y_combined)
        preds_real_after = retrained_model.predict(X_real)
        preds_synth_after = retrained_model.predict(df_gen.values)

        acc_real_after = accuracy_score(y_real, preds_real_after)
        acc_synth_after = accuracy_score(np.ones(len(df_gen)), preds_synth_after)

        st.write("✅ Synthetic IDs flagged after retraining:")
        st.dataframe(df_gen[preds_synth_after == 1].head())

        st.write("📈 Retrained Model Report (on real data):")
        st.json(classification_report(y_real, preds_real_after, output_dict=True))

        # Bar Chart: Accuracy Comparison
        st.subheader("🚀 Model Performance Comparison")

        labels = ["Real (Before)", "Real (After)", "Synthetic (After)"]
        accuracies = [acc_real_before, acc_real_after, acc_synth_after]

        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(labels, accuracies, color=['blue', 'green', 'orange'])
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("Accuracy")
        ax_bar.set_title("Accuracy Before vs After Retraining")
        for i, v in enumerate(accuracies):
            ax_bar.text(i, v + 0.02, f"{v:.2f}", ha='center')
        st.pyplot(fig_bar)

st.sidebar.markdown("---")
st.sidebar.markdown("👁️‍🗨️ *This tool simulates and defends against synthetic fraud.*")

