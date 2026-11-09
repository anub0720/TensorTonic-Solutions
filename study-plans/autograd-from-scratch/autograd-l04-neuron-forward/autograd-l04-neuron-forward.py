import torch

def neuron_forward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> tuple:
    """
    Returns a tuple of scalar tensors: preactivation and tanh output.
    """
    a=torch.dot(weights,inputs)+bias
    y=torch.tanh(a)
    return (a,y)
