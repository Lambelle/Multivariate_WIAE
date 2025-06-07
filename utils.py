import torch
import numpy as np


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


def crps(y_true_all, y_pred_all, sample_weight=None):
    num_samples = y_pred_all.shape[1]
    total_crps = []
    y_pred = np.transpose(y_pred_all, (1, 0))
    absolute_error = np.mean(np.abs(y_pred - y_true_all), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * np.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    crps = np.average(per_obs_crps, weights=sample_weight)
    total_crps.append(crps)
    return sum(total_crps) / len(total_crps)


def uncond_coverage(alpha: float, y_true:np.ndarray, y_predict: np.ndarray):
    num_sample = y_predict.shape[1]
    interval_coverage = int(num_sample * (1 - alpha))
    total_uncond_conv = []
    total_width = []
    if interval_coverage % 2 == 0:
        start_index = interval_coverage // 2
        end_index = start_index
    else:
        start_index = interval_coverage // 2
        end_index = start_index + 1

    single_predict = y_predict
    single_true = y_true
    single_predict = np.sort(single_predict, axis=1)
    lower_bound = single_predict[:, start_index]
    upper_bound = single_predict[:, -end_index]
    uncond_conv = np.sum(
            np.logical_and(lower_bound < single_true, single_true < upper_bound)
    ) / (single_true.size)
    width = np.mean(upper_bound-lower_bound)
    total_width.append(width)
    total_uncond_conv.append(uncond_conv.item())

    return sum(total_uncond_conv) / len(total_uncond_conv), sum(total_width)/len(total_width)

def correct_directions(y_true:torch.tensor,y_pred:torch.tensor):
    y_pred_pos = np.mean(y_pred,axis=1)
    y_pred_pos = (y_pred_pos>0)
    y_true_pos = (y_true >0)
    correct_directions = np.logical_and(y_pred_pos,y_true_pos)

    return np.sum(correct_directions)/(correct_directions.size)


def metrics(true, pred_mean, pred_median, pred_all, pred_step):
    std = np.std(true)
    mean = np.mean(true)
    abs_mean = np.mean(abs(true))
    square_mean = np.mean(np.square(true))

    mse = (
        np.mean(
            (true[abs(true) <= 3 * std + mean] - pred_mean[abs(true) <= 3 * std + mean])
            ** 2
        )
        / square_mean
    )
    mae = (
        np.mean(
            abs(
                true[abs(true) <= 3 * std + mean]
                - pred_median[abs(true) <= 3 * std + mean]
            )
        )
        / abs_mean
    )
    median_se = (
        np.median(
            (true[abs(true) <= 3 * std + mean] - pred_mean[abs(true) <= 3 * std + mean])
            ** 2
        )
        / square_mean
    )
    median_ae = (
        np.median(
            abs(
                true[abs(true) <= 3 * std + mean]
                - pred_median[abs(true) <= 3 * std + mean]
            )
        )
        / abs_mean
    )

    mape = np.mean(
        abs(
            true[abs(true) <= 3 * std + mean] - pred_median[abs(true) <= 3 * std + mean]
        )
        / (
            0.5
            * (
                abs(true[abs(true) <= 3 * std + mean])
                + abs(pred_median[abs(true) <= 3 * std + mean])
            )
        )
    )

    if pred_step > true.shape[0]:
        mase = 0
    else:
        mase = np.mean(abs(true - pred_median)) / np.mean(
            abs(true[pred_step:] - true[:-pred_step])
        )
        mase = mase.item()

    crps_score = crps(true, pred_all)
    uncond_cov_90, pinaw_90 = uncond_coverage(0.9, true, pred_all)
    uncond_cov_50, pinaw_50 = uncond_coverage(0.5, true, pred_all)
    uncond_cov_10, pinaw_10 = uncond_coverage(0.1, true, pred_all)

    correct_dir = correct_directions(true,pred_all)

    return (
        mse.item(),
        mae.item(),
        median_se.item(),
        median_ae.item(),
        mape.item(),
        mase,
        crps_score.item(),
        uncond_cov_90,
        pinaw_90,
        uncond_cov_50,
        pinaw_50,
        uncond_cov_10,
        pinaw_10,
        correct_dir.item()
    )
