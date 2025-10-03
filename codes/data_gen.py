import numpy as np
from scipy.linalg import expm,qr

# Define Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = [I, X, Y, Z]

def ket0(n):
    """Return |0>^⊗n as a state vector."""
    v = np.array([1, 0], dtype=complex)
    state = v
    for _ in range(n - 1):
        state = np.kron(state, v)
    return state

def query(H, U, t, M, num_queries, num_qubits):
    """
    Simulate quantum process and sample outcomes.
    
    Args:
        H: Hamiltonian (2^n x 2^n matrix)
        U: unitary operator (2^n x 2^n matrix)
        t: evolution time (float)
        M: measurement operator (2^n x 2^n matrix)
        num_queries: number of measurement samples
        num_qubits: number of qubits
    
    Returns:
        samples (list): sampled bitstrings
        probs (dict): probability distribution over computational basis
    """
    # Initialize |0>^⊗n
    state0 = ket0(num_qubits)

    # Apply U
    state1 = U @ state0

    # Time evolution
    U_evolve = expm(-1j * H * t)
    state2 = U_evolve @ state1

    # Apply measurement operator M
    state3 = M @ state2

    # Compute probabilities in computational basis
    probs = np.abs(state3) ** 2
    probs = probs / np.sum(probs)  # normalize

    # Sampling
    basis_states = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]
    samples = np.random.choice(basis_states, size=num_queries, p=probs)
    samples.append(t)
    # Collect distribution as dict
    prob_dict = {basis_states[i]: probs[i] for i in range(2**num_qubits)}

    return samples.tolist(), prob_dict

if __name__ == "__main__":
    num_qubits = 2
    dim = 2 ** num_qubits

    # Generate random Hermitian Hamiltonian H
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (A + A.conj().T) / 2

    # Generate random unitary U (QR decomposition trick)
    B = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Q, _ = qr(B)
    U = Q

    # Random measurement operator (here: also a random unitary)
    C = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Q2, _ = qr(C)
    M = Q2

    # Run query
    samples, prob_dist = query(H, U, t=0.5, M=M, num_queries=4, num_qubits=num_qubits)

    print("Samples:", samples)
    print("Probability distribution:")
    for k, v in prob_dist.items():
        print(f"  {k}: {v:.4f}")