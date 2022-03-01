from torch.nn import Linear, ReLU, Sequential

# Todo: add the script file including all the input settings

def build_mlp_model(layers: list = [256, 256],
                    activation: list = "Relu"):
    torch_layers = []
    for inp, out in zip(layers[:-1], layers[1:]):
        torch_layers.append(Linear(inp, out))
        torch_layers.append(ReLU()) if activation == "Relu" else None
    return Sequential(*torch_layers)


if __name__ == '__main__':
    # Construct the model
    layers = [8, 256, 256, 3]
    build_mlp_model(layers)
