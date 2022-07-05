from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, Dropout
from utils import *


def build_mlp_model(layers: list = [256, 256],
                    activation: str = "Relu",
                    dropout:float = 0.0,
                    batch_norm: bool = True):

    torch_layers = []
    for i, (inp, out) in enumerate(zip(layers[:-1], layers[1:])):
        torch_layers.append(Linear(inp, out))
        torch_layers.append(ReLU()) if activation.lower() == "relu" else None
        if i < len(layers)-2:
            torch_layers.append(Dropout(p=dropout))
    return Sequential(*torch_layers)


if __name__ == '__main__':
    # Construct the model
    layers = [8, 256, 256, 3]
    build_mlp_model(layers)
