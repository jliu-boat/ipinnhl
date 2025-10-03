import torch
from models import PINN

def init_loss(model, U: torch.Tensor, n_qubits: int):
    """
    Initial condition loss for PINN.
    
    Args:
        model: PINN model
        U: unitary operator (2^n x 2^n complex torch.Tensor)
        n_qubits: number of qubits
    Returns:
        loss (torch scalar)
    """
    device = next(model.parameters()).device
    dim = 2 ** n_qubits

    # --- True initial state ---
    ket0 = torch.zeros(dim, dtype=torch.complex64, device=device)
    ket0[0] = 1.0  # |00..0>
    psi_true = U @ ket0
    psi_true = psi_true / torch.linalg.norm(psi_true)

    # --- Predicted initial state (t=0) ---
    # Build computational basis
    basis = torch.zeros((dim, n_qubits), dtype=torch.float32, device=device)
    for i in range(dim):
        bits = list(map(int, format(i, f"0{n_qubits}b")))
        basis[i] = torch.tensor(bits, dtype=torch.float32, device=device)

    t_col = torch.zeros((dim, 1), dtype=torch.float32, device=device)  # t=0
    inputs = torch.cat([basis, t_col], dim=1)

    outputs = model(inputs)  # [dim, 2]
    psi_pred = outputs[:, 0] + 1j * outputs[:, 1]
    psi_pred = psi_pred / torch.linalg.norm(psi_pred)

    # --- Overlap ---
    overlap = torch.dot(torch.conj(psi_true), psi_pred)
    loss = -torch.abs(overlap) ** 2
    return loss.real


# ----------------- Unit test -----------------
if __name__ == "__main__":
    n_qubits = 2
    n_hidden = 32
    n_layer = 3

    model = PINN(n_qubits, n_hidden, n_layer)

    # Random unitary for test (QR decomposition)
    import numpy as np
    from scipy.linalg import qr

    dim = 2 ** n_qubits
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Q, _ = qr(A)
    U = torch.tensor(Q, dtype=torch.complex64)

    loss_val = init_loss(model, U, n_qubits)
    print("Init loss:", loss_val.item())
