def check_gradient_statistics(named_parameters):
    for name, param in named_parameters:
        if param.grad is not None:
            grad = param.grad.detach().cpu().numpy()
            print(
                f"Layer: {name}, Min Gradient: {grad.min()}, Max Gradient: {grad.max()}, Mean Gradient: {grad.mean()}"
            )
