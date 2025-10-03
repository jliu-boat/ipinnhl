import torch
import torch.nn as nn
import torch.optim as optim

from models import PINN
from phy_loss import physics_loss
from data_loss import data_loss
from init_loss import init_loss


def get_adam_optimizer(model, lr=1e-3, weight_decay=0.0):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_lbfgs_optimizer(model, lr=1.0, max_iter=500, tolerance_grad=1e-7, tolerance_change=1e-9):
    return optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=50,
        line_search_fn="strong_wolfe"
    )


def combined_loss(model, H, t, n_qubits, data_samples, U,
                  lambda_phy=1.0, lambda_data=1.0, lambda_init=1.0):
    """
    Combined PINN loss: physics + data + init
    """
    loss_phy = physics_loss(model, H, t, n_qubits)
    loss_data = data_loss(model, samples, t, n_qubits)
    loss_init = init_loss(model, U, n_qubits)
    return lambda_phy * loss_phy + lambda_data * loss_data + lambda_init * loss_init


def train_with_adam_then_lbfgs(model, H, t, n_qubits, samples, U,
                               adam_epochs=500, lr=1e-3, device="cpu"):
    """
    First trains with Adam, then refines with L-BFGS on the combined PINN loss.
    """
    model.to(device)

    # ---- Stage 1: Adam ----
    adam = get_adam_optimizer(model, lr=lr)
    for epoch in range(adam_epochs):
        adam.zero_grad()
        loss = combined_loss(model, H, t, n_qubits, samples, U)
        loss.backward()
        adam.step()
        if epoch % 100 == 0:
            print(f"[Adam Epoch {epoch}] Loss = {loss.item()}")

    # ---- Stage 2: L-BFGS ----
    lbfgs = get_lbfgs_optimizer(model)

    def closure():
        lbfgs.zero_grad()
        loss = combined_loss(model, H, t, n_qubits, samples, U)
        loss.backward()
        return loss

    lbfgs.step(closure)
    return model


# ============================================================
# Unit test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # Example setup: 2 qubits, PINN model
    n_qubits = 2
    pinn = PINN(n_qubits=n_qubits, n_hidden=16, n_layer=2)

    # Hamiltonian (Pauli-Z ⊗ I)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    I = torch.eye(2, dtype=torch.complex64)
    H = torch.kron(Z, I)   # 4x4 Hamiltonian for 2 qubits

    # Time
    t = 0.5

    # Fake data samples (bitstrings + time)
    samples = ["00", "01", "11", "10"]


    # Initial unitary (identity)
    U = torch.eye(2 ** n_qubits, dtype=torch.complex64)

    # Compute individual losses
    loss_phy = physics_loss(pinn, H, t, n_qubits)
    loss_data = data_loss(pinn, samples, t, n_qubits)
    loss_init = init_loss(pinn, U, n_qubits)
    print("Initial physics loss:", loss_phy.item())
    print("Initial data loss:", loss_data.item())
    print("Initial init loss:", loss_init.item())

    # Train with Adam + L-BFGS
    trained_model = train_with_adam_then_lbfgs(
        pinn, H, t, n_qubits, samples, U, adam_epochs=50, lr=1e-2
    )

    # Final combined loss
    final_loss = combined_loss(trained_model, H, t, n_qubits, samples, U).item()
    print("Final combined loss:", final_loss)

    assert final_loss < (loss_phy + loss_data + loss_init).item(), "Loss did not decrease!"
    print("Unit test passed: Combined loss minimized with Adam+L-BFGS")
