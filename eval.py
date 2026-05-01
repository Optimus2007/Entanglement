"""
QML Psychological Stress Predictor — Evaluation Script
=======================================================
Loads pre-trained weights from train.py, runs inference on
eval.csv, and computes the Mean Absolute Error (MAE).

Usage:
    python eval.py
"""

import numpy as np
import pandas as pd
from pyvqnet.tensor import QTensor, tensor
from pyvqnet.nn import Module, Linear
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import probs_measure
import pyqpanda3.core as pq

# ──────────────────────────────────────────────
# Configuration (must match train.py exactly)
# ──────────────────────────────────────────────
FEATURES = ['Palpitations', 'Act_Conflict', 'Sleep_Issues', 'Subj_Conf']
TARGET   = 'Recent_Stress'
N_QUBITS = 4
N_LAYERS = 2
N_PARAMS = N_LAYERS * N_QUBITS * 2  # 16


# ──────────────────────────────────────────────
# Preprocessing (identical to train.py)
# ──────────────────────────────────────────────
def preprocess(df):
    """
    Extract the 4 selected features and normalize
    from Likert [1, 5] to [0, pi] for angle encoding.
    """
    df = df.copy()
    X = df[FEATURES].values.astype(np.float32)
    X = (X - 1.0) / 4.0 * np.pi
    y = df[TARGET].values.astype(np.float32) if TARGET in df.columns else None
    return X, y


# ──────────────────────────────────────────────
# Quantum Circuit (identical to train.py)
# ──────────────────────────────────────────────
def vqc_circuit(input_data, weights):
    """
    4-qubit VQC with TTN entanglement.

    Per layer:
      1. Angle Encoding (Data Re-uploading): RY(x_i) on each qubit
      2. TTN Entanglement (3 CNOTs):
         - Level 1: CNOT(q0,q1), CNOT(q2,q3) — local domain pairing
         - Level 2: CNOT(q1,q2) — global cross-domain compression
      3. Trainable Ansatz: RY(theta) + RZ(theta) on each qubit
    """
    machine = pq.CPUQVM()
    qlist   = list(range(N_QUBITS))
    circuit = pq.QCircuit()

    param_idx = 0
    for layer in range(N_LAYERS):

        # Step 1: Angle Encoding (Data Re-uploading)
        for i in range(N_QUBITS):
            circuit << pq.RY(qlist[i], input_data[i])

        # Step 2: TTN Entanglement
        circuit << pq.CNOT(qlist[0], qlist[1])  # physical domain
        circuit << pq.CNOT(qlist[2], qlist[3])  # psychosocial domain
        circuit << pq.CNOT(qlist[1], qlist[2])  # cross-domain

        # Step 3: Trainable Ansatz
        for i in range(N_QUBITS):
            circuit << pq.RY(qlist[i], weights[param_idx])
            param_idx += 1
        for i in range(N_QUBITS):
            circuit << pq.RZ(qlist[i], weights[param_idx])
            param_idx += 1

    prog = pq.QProg()
    prog << circuit

    try:
        return probs_measure(machine, prog, qlist)
    except AttributeError:
        try:
            return probs_measure(prog, machine, qlist)
        except Exception:
            res = pq.prob_run_dict(prog, qlist, -1)
            return [res.get(format(i, f'0{N_QUBITS}b'), 0.0)
                    for i in range(2 ** N_QUBITS)]


# ──────────────────────────────────────────────
# Model (identical to train.py)
# ──────────────────────────────────────────────
class StressQNN(Module):
    def __init__(self):
        super().__init__()
        self.qlayer = QuantumLayer(
            qprog_with_measure=vqc_circuit,
            para_num=N_PARAMS,
            diff_method="parameter_shift",
            delta=0.01
        )
        self.fc = Linear(2 ** N_QUBITS, 1)

    def forward(self, x):
        q_out = self.qlayer(x)
        out   = self.fc(q_out)
        out   = 1.0 + 4.0 * tensor.sigmoid(out)
        return out.reshape([-1])


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate():
    # Load evaluation data
    print("Loading eval.csv...")
    eval_df = pd.read_csv('eval.csv')
    X_eval, y_eval = preprocess(eval_df)

    # Build model and load pre-trained weights
    print("Loading pre-trained weights...")
    model = StressQNN()
    saved_weights = np.load('best_model_weights.npy', allow_pickle=True).item()

    for k, v in model.state_dict().items():
        model.state_dict()[k][:] = QTensor(saved_weights[k])

    # Run inference
    model.eval()
    eval_x = QTensor(X_eval)
    predictions = model(eval_x)
    pred_np = predictions.to_numpy().flatten()

    # Compute MAE
    if y_eval is not None:
        mae = float(np.mean(np.abs(pred_np - y_eval)))
        score = 30 * np.exp(-0.1 * mae)
        print(f"MAE: {mae:.4f}")
        print(f"Score: {score:.2f}/30")
    else:
        print("No target column found; printing predictions only.")

    # Print predictions
    print(f"\nPredictions ({len(pred_np)} samples):")
    for i, p in enumerate(pred_np):
        print(f"  Sample {i+1:3d}: {p:.4f}")


if __name__ == "__main__":
    evaluate()
