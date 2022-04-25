from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, MaxPool1d
from utils import *


def build_mlp_model(layers: list = [256, 256],
                    activation: str = "Relu",
                    batch_norm: bool = True):

    torch_layers = []
    for inp, out in zip(layers[:-1], layers[1:]):
        torch_layers.append(Linear(inp, out))
        if activation.lower() == "relu": torch_layers.append(ReLU())
        if batch_norm: torch_layers.append(BatchNorm1d(out))

    return Sequential(*torch_layers)


if __name__ == '__main__':
    # Construct the model
    model = build_mlp_model(convert_str_to_list(mlp_config["layers"]))


