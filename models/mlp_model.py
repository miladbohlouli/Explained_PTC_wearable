from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
from utils import *

mlp_config = config("mlp")


def build_mlp_model(layers: list = [256, 256],
                    activation: str = "Relu",
                    batch_norm: bool = True):

    torch_layers = []
    for inp, out in zip(layers[:-1], layers[1:]):
        torch_layers.append(Linear(inp, out))
        torch_layers.append(ReLU()) if activation == "Relu" else None
        # torch_layers.append(BatchNorm1d(out)) if batch_norm else None

    return Sequential(*torch_layers)


if __name__ == '__main__':
    # Construct the model
    layers = [8, 256, 256, 3]
    build_mlp_model(layers)
