def local_vjp(
    operation: str, inputs: list, output: float, upstream_gradient: float
) -> list:
    """Returns a list of float gradient contributions in input order."""
    g = float(upstream_gradient)

    if operation == "add":
        return [g, g]
    elif operation == "mul":
        a, b = inputs[0], inputs[1]
        return [float(g * b), float(g * a)]
    elif operation == "tanh":
        t = float(output)
        return [float(g * (1.0 - t * t))]
    else:
        raise ValueError(f"Unsupported operation: {operation}")