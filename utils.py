import torch


def calculate_gradient_penalty(discriminator, real_output, fake_output):
    epsilon = torch.rand((real_output.shape[0], 1, 1))
    interpolates = (epsilon * real_output + (1 - epsilon) * fake_output).requires_grad_(
        True
    )

    interpolate_output = discriminator(interpolates)
    # gradients = torch.autograd.grad(outputs=interpolate_output,inputs=interpolates)

    grad_outputs = torch.ones(interpolate_output.size(), requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=interpolate_output,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(p=2, dim=1) - 1) ** 2)
    return gradient_penalty


def metrics(true, pred_mean, pred_median):
    std = torch.std(true)
    mean = torch.mean(true)
    abs_mean = torch.mean(abs(true))

    mse = (
        torch.mean(
            (true[abs(true) <= 3 * std + mean] - pred_mean[abs(true) <= 3 * std + mean])
            ** 2
        )
        / abs_mean
    )
    mae = (
        torch.mean(
            abs(
                true[abs(true) <= 3 * std + mean]
                - pred_median[abs(true) <= 3 * std + mean]
            )
        )
        / abs_mean
    )
    median_se = (
        torch.median(
            (true[abs(true) <= 3 * std + mean] - pred_mean[abs(true) <= 3 * std + mean])
            ** 2
        )
        / abs_mean
    )
    median_ae = (
        torch.median(
            abs(
                true[abs(true) <= 3 * std + mean]
                - pred_median[abs(true) <= 3 * std + mean]
            )
        )
        / abs_mean
    )
    return mse.item(), mae.item(), median_se.item(), median_ae.item()
