"""Eksperimen Tugas Analisis 1 - kalimat Bahasa Indonesia + contoh manual d_k=2.
Memakai fungsi dari skrip template lab_pekan_05_transformer_self_attention.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
from lab_pekan_05_transformer_self_attention import (
    scaled_dot_product_attention, plot_attention_heatmap, softmax,
)

np.random.seed(42)

# ---------------- Bagian 1: contoh manual (3 kata, d_k = 2) ----------------
print("=== BAGIAN 1: CONTOH MANUAL ===")
kata = ["Saya", "makan", "nasi"]
Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
d_k = 2
S = Q @ K.T
S_scaled = S / np.sqrt(d_k)
A = softmax(S_scaled)
out = A @ V
np.set_printoptions(precision=4, suppress=True)
print("Q K^T =\n", S)
print("sqrt(d_k) =", round(np.sqrt(d_k), 4))
print("Q K^T / sqrt(d_k) =\n", S_scaled)
print("Softmax per baris =\n", A)
print("Jumlah tiap baris =", A.sum(axis=1))
print("Output = A V =\n", out)
o2, a2 = scaled_dot_product_attention(Q, K, V)
assert np.allclose(A, a2) and np.allclose(out, o2)

# ---------------- Bagian 2: kalimat Indonesia ----------------
print("\n=== BAGIAN 3: KALIMAT INDONESIA ===")
tokens = ["Kucing", "itu", "tidak", "menyeberang", "jalan", "karena", "dia", "sangat", "lelah"]
n, d = len(tokens), 16
emb = np.random.randn(n, d) * 0.1
Wq = np.random.randn(d, d) * 0.2
Wk = np.random.randn(d, d) * 0.2
Wv = np.random.randn(d, d) * 0.2
Qs, Ks, Vs = emb @ Wq, emb @ Wk, emb @ Wv
Qs = Qs * 12.0  # skala agar skor Q.K^T cukup besar dan softmax tidak seragam

i_dia, i_kucing, i_jalan, i_lelah = (tokens.index(t) for t in ["dia", "Kucing", "jalan", "lelah"])
# Bias semantik sintetis (meniru model terlatih), sama seperti template:
Ks[i_kucing] = Qs[i_dia] * 1.8 + np.random.randn(d) * 0.05
Ks[i_jalan] = Qs[i_dia] * 0.3 + np.random.randn(d) * 0.05
Ks[i_lelah] = Qs[i_dia] * 1.2 + np.random.randn(d) * 0.05

_, W = scaled_dot_product_attention(Qs, Ks, Vs)
print("Bobot perhatian query 'dia':")
for idx in np.argsort(W[i_dia])[::-1]:
    print(f"  {tokens[idx]:<12} {W[i_dia, idx]:.4f}")
plot_attention_heatmap(tokens, W, "heatmap_kalimat_indonesia.png")
