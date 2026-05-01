# Entanglement — Wukong Cup 2026 (3rd Prize)

This repository contains our solution for the **2026 CIC “Wukong Cup” Quantum Computing Competition (University Track)**, where our team ranked **3rd globally**.

---

## 🧠 Overview

We worked on two core problems:

- **Q1:** Design and implementation of Grover’s oracle and diffusion operator circuits  
- **Q2:** Constrained Variational Quantum Circuit (VQC) for ordinal regression under strict resource limits  

The work was implemented using **pyQPanda3** and focuses on **efficient quantum circuit design under NISQ constraints**.

---

## 🔬 Key Insight

> Under strict constraints (limited qubits, ≤100 parameters, no classical dimensionality reduction),  
> **structured entanglement (Tree Tensor Network)** outperforms unstructured circuit designs.

This challenges the common assumption that increasing qubits or entanglement improves performance.

---

## ⚛️ Q1 — Grover’s Algorithm (Circuit Design)

We designed quantum circuits for:

- Oracle construction for target state |1010⟩  
- Diffusion operator for amplitude amplification  

### Highlights:
- Used multi-controlled Z (MCZ) gates with X-gate transformations  
- Implemented phase marking and amplitude amplification explicitly  
- Derived full mathematical formulation of oracle and diffusion operators  

📄 See full report: `Grovers_report.pdf`

---

## ⚙️ Q2 — Variational Quantum Model (TTN Architecture)

We developed a **4-qubit Variational Quantum Circuit (VQC)** for ordinal regression.

### Key Design:

- **Feature Reduction:** 18 → 4 features using statistical filtering  
- **Encoding:** Angle encoding with data re-uploading  
- **Architecture:** Tree Tensor Network (TTN)  
- **Parameters:** 33 (within ≤100 constraint)  

---

## 🌳 TTN Architecture (Core Contribution)

Instead of standard topologies:

- ❌ Ring (4 CNOTs)  
- ❌ Linear chain  

We used:

- ✅ **Hierarchical TTN (3 CNOTs)**  
  - Local pairing (q0–q1, q2–q3)  
  - Global compression (q1–q2)

### Why TTN?

- Fewer gates (efficient for NISQ)
- Better gradient behavior (reduced barren plateaus)
- Matches domain structure of data

---

## 📊 Experimental Results

### Hardware Ablation Study

| Model | Qubits | MAE |
|------|--------|------|
| 1-Qubit | 1 | ~0.809 |
| 2-Qubit | 2 | ~0.816 |
| 3-Qubit | 3 | ~0.809 |
| 4-Qubit Linear | 4 | ~0.810 |
| 4-Qubit Ring | 4 | ~0.810 |
| **4-Qubit TTN (ours)** | 4 | **0.7943** |

### Key Finding:

- All **unstructured topologies converge to ~0.81 MAE**
- Only **TTN breaks the performance barrier**

👉 **Structure > Entanglement > Qubit Count**

---

## 📈 Model Details

- Loss: MAE  
- Optimizer: Adam (lr = 0.01)  
- Output constraint: [1,5] using sigmoid scaling  
- Training: 100 epochs  

Score:

27.71 / 30


---

## 🧪 Additional Exploration

We also explored:

- 2-qubit dense encoding model (high efficiency, lower expressivity)  
- Trade-off between parameter count and model capacity  

---

## 🚀 Research Direction

This work opens up directions in:

- Structured ansatz design for NISQ systems  
- Entanglement topology vs expressivity trade-offs  
- Barren plateau mitigation via tensor networks  
- Resource-constrained quantum learning  

---

## 🛠️ Tech Stack

- pyQPanda3  
- Variational Quantum Circuits (VQC)  
- Tree Tensor Networks (TTN)  

---

## 👥 Team

Developed by **Team Entanglement**:

- Aditya Raj (Team Leader)  
- Oishik Kar  
- Noble Agyeman-Bobie  

---

## 📄 Reports

- Q1: Grover’s Algorithm → `Grovers_report.pdf`  
- Q2: TTN-based VQC → `Q2-Report.pdf`

---

## 📌 License

This project is licensed under the MIT License.