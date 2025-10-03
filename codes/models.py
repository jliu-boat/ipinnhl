import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, n_qubits: int, n_hidden: int, n_layer: int):
        super(PINN, self).__init__()
        
        input_dim = n_qubits + 1  # +1 for time or extra parameter
        output_dim = 2

        layers = []
        layers.append(nn.Linear(input_dim, n_hidden))
        layers.append(nn.SiLU())

        for _ in range(n_layer - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(n_hidden, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DNN(nn.Module):
    def __init__(self, n_qubits: int, n_hidden: int, n_layer: int):
        super(DNN, self).__init__()
        
        input_dim = n_qubits + 1  # +1 for time
        output_dim = 2

        layers = []
        layers.append(nn.Linear(input_dim, n_hidden))
        layers.append(nn.Tanh())

        for _ in range(n_layer - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(n_hidden, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
# ----------------- Unit test -----------------
if __name__ == "__main__":
    n_qubits = 3
    n_hidden = 64
    n_layer = 4

    model = PINN(n_qubits, n_hidden, n_layer)

    # Example input: batch of 5 samples, each with n_qubits+1 features
    x = torch.randn(5, n_qubits + 1)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output:", y)
    model = DNN(n_qubits, n_hidden, n_layer)

    # Example input: batch of 5 samples, each with n_qubits+1 features
    x = torch.randn(5, n_qubits + 1)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output:", y)
