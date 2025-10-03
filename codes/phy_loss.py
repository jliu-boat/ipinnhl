import torch
from models import PINN

def physics_loss(model, H: torch.Tensor, t: float, n_qubits: int):
    """
    Physics-informed loss enforcing Schrödinger equation:
        L = || i dPsi/dt - H Psi ||^2
    """
    device = next(model.parameters()).device
    dim = 2 ** n_qubits

    # Build computational basis as 0/1 encoding
    basis = torch.zeros((dim, n_qubits), dtype=torch.float32, device=device)
    for i in range(dim):
        bits = list(map(int, format(i, f"0{n_qubits}b")))
        basis[i] = torch.tensor(bits, dtype=torch.float32, device=device)

    time_col = torch.full((dim, 1), t, dtype=torch.float32, device=device)
    inputs = torch.cat([basis, time_col], dim=1)  # shape [dim, n_qubits+1]
    inputs.requires_grad_(True)

    # Forward pass
    outputs = model(inputs)  # shape [dim, 2]
    psi = outputs[:, 0] + 1j * outputs[:, 1]  # complex wavefunction

    # Normalize
    norm = torch.sqrt(torch.sum(torch.abs(psi) ** 2))
    psi = psi / norm

    # Derivative wrt time
    dpsi_dt = torch.autograd.grad(
        psi,
        inputs,
        grad_outputs=torch.ones_like(psi),
        retain_graph=True,
        create_graph=True,
    )[0][:, -1]  # derivative wrt last input (time)

    # Residual: i dPsi/dt - H Psi
    Hpsi = torch.matmul(H, psi)
    residual = 1j * dpsi_dt - Hpsi

    loss = torch.mean(torch.abs(residual) ** 2)
    return loss

def data_loss(model, samples, t: float, n_qubits: int):
    """
    Negative log-likelihood loss from sampled measurement data.
    
    Args:
        model: PINN model
        samples: list of sampled bitstrings (e.g., ['010', '111'])
        t: time value
        n_qubits: number of qubits
    Returns:
        loss (torch scalar)
    """
    device = next(model.parameters()).device
    losses = []

    for bitstring in samples:
        # Convert bitstring -> tensor input
        bits = torch.tensor([int(b) for b in bitstring],
                            dtype=torch.float32, device=device).unsqueeze(0)
        time_col = torch.tensor([[t]], dtype=torch.float32, device=device)
        x = torch.cat([bits, time_col], dim=1)  # [1, n+1]

        # Forward pass
        out = model(x)  # [1,2]
        prob = torch.sum(out**2, dim=1)  # real^2 + imag^2

        # Numerical stability (avoid log(0))
        prob = torch.clamp(prob, min=1e-12)

        losses.append(-torch.log(prob))

    return torch.mean(torch.cat(losses))


# ----------------- Unit test -----------------
if __name__ == "__main__":
    # 1 qubit example
    n_qubits = 1
    n_hidden = 16
    n_layer = 2

    model = PINN(n_qubits, n_hidden, n_layer)

    # Hamiltonian: Z
    H = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    # Pick a time
    t = 0.5

    loss_val = physics_loss(model, H, t, n_qubits)
    print("Physics loss (1 qubit, H=Z, t=0.5):", loss_val.item())


