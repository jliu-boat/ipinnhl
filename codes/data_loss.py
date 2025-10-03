import torch
import numpy as np
from models import PINN

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
    n_qubits = 2
    n_hidden = 32
    n_layer = 3
    t = 0.5

    model = PINN(n_qubits, n_hidden, n_layer)

    # Fake samples (normally from data_gen.py)
    samples = ["00", "01", "11", "10"]

    loss_val = data_loss(model, samples, t, n_qubits)
    print("Data loss:", loss_val.item())
