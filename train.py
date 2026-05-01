"""
QML Psychological Stress Predictor
===================================
4-qubit Variational Quantum Circuit with Tree Tensor Network (TTN)
entanglement for hierarchical ordinal regression.

Architecture:
  - Angle Encoding with Data Re-uploading (2 layers)
  - TTN Entanglement: 3 CNOTs/layer (25% fewer than ring topology)
    Level 1: local domain pairing (physical + psychosocial)
    Level 2: global cross-domain compression
  - Trainable Ansatz: RY + RZ per qubit per layer

Parameters:
  Quantum  : 2 layers x 4 qubits x 2 rotations = 16
  Classical: FC layer 16 -> 1 = 17 (weights + bias)
  Total    : 33 parameters (well under the 100-param limit)

References:
  [1] Grant et al., "Hierarchical quantum classifiers", npj Quantum Inf. 4, 65 (2018)
  [2] Perez-Salinas et al., "Data re-uploading for a universal quantum classifier", Quantum 4, 226 (2020)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyvqnet.tensor import QTensor, tensor
from pyvqnet.nn import Module, Linear
from pyvqnet.optim import Adam
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.qnn.pq3.measure import probs_measure
import pyqpanda3.core as pq

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
FEATURES = ['Palpitations', 'Act_Conflict', 'Sleep_Issues', 'Subj_Conf']
TARGET   = 'Recent_Stress'
N_QUBITS = 4
N_LAYERS = 2
N_PARAMS = N_LAYERS * N_QUBITS * 2  # 16 trainable quantum params

EPOCHS     = 100
BATCH_SIZE = 32
LR         = 0.01


# ──────────────────────────────────────────────
# Preprocessing
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
# Custom MAE Loss
# ──────────────────────────────────────────────
class MAELoss(Module):
    """Mean Absolute Error loss — directly optimizes the competition metric."""
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        return tensor.mean(tensor.abs(preds - targets))


# ──────────────────────────────────────────────
# Quantum Circuit: Angle Encoding + TTN
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

        # Step 2: TTN Entanglement (Grant et al., 2018)
        # Level 1 — local pairwise entanglement
        circuit << pq.CNOT(qlist[0], qlist[1])  # physical domain
        circuit << pq.CNOT(qlist[2], qlist[3])  # psychosocial domain
        # Level 2 — global hierarchical compression
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

    # Robust measurement — handles pyqpanda3 API variations
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
# Model
# ──────────────────────────────────────────────
class StressQNN(Module):
    """
    Hybrid quantum-classical model.
    Quantum layer outputs 2^4 = 16 probabilities,
    FC layer maps to a single stress prediction in [1, 5].
    """
    def __init__(self):
        super().__init__()
        self.qlayer = QuantumLayer(
            qprog_with_measure=vqc_circuit,
            para_num=N_PARAMS,
            diff_method="parameter_shift",
            delta=0.01
        )
        self.fc = Linear(2 ** N_QUBITS, 1)  # 16 -> 1

    def forward(self, x):
        q_out = self.qlayer(x)
        out   = self.fc(q_out)
        out   = 1.0 + 4.0 * tensor.sigmoid(out)  # clamp to [1, 5]
        return out.reshape([-1])


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_model():
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    eval_df  = pd.read_csv('eval.csv')

    X_train, y_train = preprocess(train_df)
    X_eval,  y_eval  = preprocess(eval_df)

    model     = StressQNN()
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = MAELoss()

    num_batches = len(X_train) // BATCH_SIZE

    best_val_mae = float('inf')
    best_score   = 0.0
    best_weights = None

    train_history = []
    val_history   = []

    fc_params = 2 ** N_QUBITS + 1
    print(f"Quantum params: {N_PARAMS} | FC params: {fc_params} | Total: {N_PARAMS + fc_params}")
    print(f"Architecture: Angle Encoding + TTN Entanglement | Layers: {N_LAYERS}")
    print("-" * 60)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        # Shuffle training data
        indices      = np.random.permutation(len(X_train))
        X_train_shuf = X_train[indices]
        y_train_shuf = y_train[indices]

        for b in range(num_batches):
            start = b * BATCH_SIZE
            end   = start + BATCH_SIZE

            batch_x = QTensor(X_train_shuf[start:end])
            batch_y = QTensor(y_train_shuf[start:end])

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer._step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        eval_x     = QTensor(X_eval)
        eval_preds = model(eval_x)

        avg_train_loss = epoch_loss / num_batches
        val_mae  = float(np.mean(np.abs(eval_preds.to_numpy() - y_eval)))
        val_score = 30 * np.exp(-0.1 * val_mae)

        train_history.append(avg_train_loss)
        val_history.append(val_mae)

        # Save best weights
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_score   = val_score
            best_weights = {k: v.to_numpy().copy()
                           for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f} | "
                  f"Score: {val_score:.2f}/30")

    print("-" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Best Validation MAE: {best_val_mae:.4f} -> Score: {best_score:.2f}/30")

    # Save weights
    np.save("best_model_weights.npy", best_weights)
    print("Saved: best_model_weights.npy")

    # Save training curve
    best_epoch = val_history.index(best_val_mae)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_history,
             label='Train MAE', color='blue', linewidth=1.5)
    plt.plot(range(1, EPOCHS + 1), val_history,
             label='Validation MAE', color='red', linewidth=1.5)
    plt.axvline(x=best_epoch + 1, color='gray', linestyle='--',
                label=f'Best Epoch ({best_epoch + 1})')
    plt.title('QML Stress Predictor: Train vs Validation (TTN Ansatz)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('loss_curve_ttn.png', dpi=300)
    print("Saved: loss_curve_ttn.png")


if __name__ == "__main__":
    train_model()
