import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from data_loader import Custom_Dataset
from utils import calculate_gradient_penalty, metrics


def arguement():
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_path", type=str, required=True)
    parser.add_argument("-dataset", type=str, required=True)
    parser.add_argument("-data_bad", type=str, required=False)
    parser.add_argument("-pred_step", type=int, default=1)

    parser.add_argument("-degree", type=int, default=4)
    parser.add_argument("-block", type=int, default=100)
    parser.add_argument("-stride", type=int, default=100)

    parser.add_argument("-output_dim", type=int, default=1)
    parser.add_argument("-hidden_dim", type=int, default=100)
    parser.add_argument("-seq_len", type=int, default=50)
    parser.add_argument("-num_feature", type=int, default=2)
    parser.add_argument("-filter_size", type=int, default=20)

    parser.add_argument("-batch_size", type=int, default=60)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-lrD", type=float, default=1e-5)
    parser.add_argument("-lrG", type=float, default=1e-5)
    parser.add_argument("-num_critic", type=int, default=10)

    parser.add_argument("-gp_coef_inn", type=float, default=0.1)
    parser.add_argument("-coef_recons", type=float, default=0.1)
    parser.add_argument("-gp_coef_recons", type=float, default=0.1)
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-sample_size", type=int, default=1000)

    parser.add_argument("--univariate", action="store_true", default=False)

    opt = parser.parse_args()

    return opt


def train_epoch(
    train_dataloader,
    encoder,
    decoder,
    inn_discriminator,
    recons_discriminator,
    optimizer_generator,
    optimizer_discriminator,
    opt,
):
    encoder.train()
    decoder.train()
    inn_discriminator.train()
    recons_discriminator.train()
    loss_D = []
    loss_G = []

    for x_input in tqdm(train_dataloader):
        start_index = 2* (opt.filter_size -1)
        end_index = -1*opt.pred_step
        for i in range(opt.num_critic):
            optimizer_discriminator.zero_grad()

            inn = encoder(x_input)
            x_recons = decoder(inn)
            x_recons[:,:,:end_index] = x_input[:,:,start_index:end_index]
            inn_real = 2 * torch.rand(inn.shape) - 1
            inn_fake_output = inn_discriminator(inn)
            inn_real_output = inn_discriminator(inn_real)
            inn_score_real = inn_real_output.mean()
            inn_score_fake = inn_fake_output.mean()

            recons_fake_output = recons_discriminator(x_recons)
            remaining_length = x_recons.shape[2]
            if opt.univariate:
                recons_real_output = recons_discriminator(
                    x_input[:, 0, -remaining_length:].unsqueeze(1)
                )
            else:
                recons_real_output = recons_discriminator(
                    x_input[:, :, -remaining_length:]
                )
            recons_score_real = recons_real_output.mean()
            recons_score_fake = recons_fake_output.mean()

            inn_gradient_penalty = calculate_gradient_penalty(
                inn_discriminator, inn_real, inn
            )
            if opt.univariate:
                recons_gradient_penalty = calculate_gradient_penalty(
                    recons_discriminator,
                    x_input[:, 0, -remaining_length:].unsqueeze(1),
                    x_recons,
                )
            else:
                recons_gradient_penalty = calculate_gradient_penalty(
                    recons_discriminator, x_input[:, :, -remaining_length:], x_recons
                )

            loss_discriminator = (
                inn_score_fake
                - inn_score_real
                + opt.gp_coef_inn * inn_gradient_penalty
                + opt.coef_recons
                * (
                    recons_score_fake
                    - recons_score_real
                    + opt.gp_coef_recons * recons_gradient_penalty
                )
            )

            loss_D.append(loss_discriminator.item())

            loss_discriminator.backward()
            optimizer_discriminator.step()

        # Train Generators
        optimizer_generator.zero_grad()

        inn = encoder(x_input)
        x_recons = decoder(inn)

        inn_fake_output = inn_discriminator(inn)
        recons_fake_output = recons_discriminator(x_recons)

        loss_generator = (
            -inn_fake_output.mean() - opt.coef_recons * recons_fake_output.mean()
        )

        # loss_generator = -torch.std(inn,dim=2).mean() - torch.std(x_recons,dim=2).mean()
        loss_generator.backward()
        optimizer_generator.step()

        loss_G.append(loss_generator.item())

    return sum(loss_G) / len(loss_G), sum(loss_D) / len(loss_D)


def eval_epoch(
    test_dataloader,
    encoder,
    decoder,
    opt,
    save_predict=False,
):
    encoder.eval()
    decoder.eval()
    MSE = []
    MAE = []
    Median_se = []
    Median_ae = []
    MAPE = []
    MASE = []
    CRPS = []
    UC_90 = []
    UC_50 = []
    UC_10 = []
    PINAW_90 = []
    PINAW_50 = []
    PINAW_10 = []
    CD = []
    MAX_V = -1 * float('inf') * torch.ones(opt.num_feature)
    MIN_V = float('inf') * torch.ones(opt.num_feature)
    if opt.univariate:
        num_feature = 1
    else:
        num_feature = opt.num_feature
    if save_predict:
        all_pred_mean = np.empty((1, num_feature))
        all_pred_median = np.empty((1, num_feature))
        all_true = np.empty((1, num_feature))
    for x_input, x_true, x_mean, x_std in test_dataloader:
        if opt.univariate:
            x_true = x_true[:, 0, :].unsqueeze(1)
            x_mean = x_mean[:, 0].unsqueeze(1)
            x_std = x_std[:, 0].unsqueeze(1)
        inn = encoder(x_input)
        inn = inn.detach().numpy()
        step = opt.pred_step
        decoder_in_len = opt.seq_len - 2 * opt.filter_size + 2  # 12


        # x_pred_mean = decoder(torch.tensor(inn)).detach().numpy()
        # x_pred_median = decoder(torch.tensor(inn)).detach().numpy()

        x_pred_median = np.empty((inn.shape[0], num_feature, decoder_in_len))
        x_pred_mean = np.empty((inn.shape[0], num_feature, decoder_in_len))
        x_pred_all = np.empty(
            (inn.shape[0], opt.sample_size, num_feature, decoder_in_len)
        )
        for row in range(inn.shape[0]):
            for j in range(decoder_in_len):
                inn_test_temp = np.tile(inn[row, :, :].copy(), (opt.sample_size, 1, 1))
                inn_test_temp[
                    :, :, j + opt.filter_size - step : j + opt.filter_size
                ] = np.random.uniform(
                    low=-1.0, high=1.0, size=(opt.sample_size, num_feature, step)
                )
                decoder_out = decoder(torch.tensor(inn_test_temp))
                decoder_out = decoder_out.detach().numpy()
                x_pred_median[row, :, j] = np.median(decoder_out[:, :, j], axis=0)
                x_pred_mean[row, :, j] = np.mean(decoder_out[:, :, j], axis=0)
                x_pred_all[row, :, :, j] = decoder_out[:, :, j]
        # Take only some rows for evaluation

        x_true = x_true * x_std.unsqueeze(2) + x_mean.unsqueeze(2)
        max_v, _ = torch.max(x_true, dim=2)
        max_v, _ = torch.max(max_v, dim=0)
        min_v, _ = torch.min(x_true, dim=2)
        min_v, _ = torch.min(min_v, dim=0)
        MAX_V = torch.max(MAX_V, max_v)
        MIN_V = torch.min(MIN_V, min_v)
        x_pred_mean = x_pred_mean * np.expand_dims(
            x_std.detach(), axis=2
        ) + np.expand_dims(x_mean.detach(), axis=2)
        x_pred_median = x_pred_median * np.expand_dims(
            x_std.detach(), axis=2
        ) + np.expand_dims(x_mean.detach(), axis=2)
        x_pred_all = x_pred_all * np.expand_dims(
            x_std.detach(), axis=(1, 3)
        ) + np.expand_dims(x_mean.detach(), axis=(1, 3))

        if "NYISO" not in opt.dataset and "CTS" not in opt.dataset:
            (
                mse,
                mae,
                median_se,
                median_ae,
                mape,
                mase,
                crps_score,
                uncond_cov_90,
                pinaw_90,
                uncond_cov_50,
                pinaw_50,
                uncond_cov_10,
                pinaw_10,
                cd,
            ) = metrics(x_true, x_pred_mean, x_pred_median, x_pred_all, step)
        else:
            (
                mse,
                mae,
                median_se,
                median_ae,
                mape,
                mase,
                crps_score,
                uncond_cov_90,
                pinaw_90,
                uncond_cov_50,
                pinaw_50,
                uncond_cov_10,
                pinaw_10,
                cd,
            ) = metrics(
                x_true[:, 0, :].unsqueeze(1),
                np.expand_dims(x_pred_mean[:, 0, :], axis=1),
                np.expand_dims(x_pred_median[:, 0, :], axis=1),
                np.expand_dims(x_pred_all[:, :, 0, :], axis=2),
                step,
            )
            MAX_V = MAX_V[0]
            MIN_V = MIN_V[0]
        MSE.append(mse)
        MAE.append(mae)
        Median_se.append(median_se)
        Median_ae.append(median_ae)
        MAPE.append(mape)
        MASE.append(mase)
        CRPS.append(crps_score)
        UC_90.append(uncond_cov_90)
        UC_50.append(uncond_cov_50)
        UC_10.append(uncond_cov_10)
        PINAW_90.append(pinaw_90)
        PINAW_50.append(pinaw_50)
        PINAW_10.append(pinaw_10)
        CD.append(cd)

        if save_predict:
            all_mae = round(sum(MAE) / len(MAE), 4)
            all_mse = round(sum(MSE) / len(MSE), 4)
            all_median_se = round(sum(Median_se) / len(Median_se), 4)
            all_median_ae = round(sum(Median_ae) / len(Median_ae), 4)
            all_mape = round(sum(MAPE) / len(MAPE), 4)
            all_mase = round(sum(MASE) / len(MASE), 4)
            all_crps = round(sum(CRPS) / len(CRPS), 4)
            all_uc_90 = round(sum(UC_90) / len(UC_90), 4)
            all_uc_50 = round(sum(UC_50) / len(UC_50), 4)
            all_uc_10 = round(sum(UC_10)/len(UC_10),4)
            pinaw_90 = sum(PINAW_90)/len(PINAW_90)/(MAX_V-MIN_V)
            pinaw_50 = sum(PINAW_50) / len(PINAW_50) / (MAX_V - MIN_V)
            pinaw_10 = sum(PINAW_10) / len(PINAW_10) / (MAX_V - MIN_V)
            all_pinaw_90 = round(pinaw_90.item(),4)
            all_pinaw_50 = round(pinaw_50.item(), 4)
            all_pinaw_10 = round(pinaw_10.item(), 4)
            all_cd = round(sum(CD)/len(CD),4)

            x_pred_mean = np.transpose(x_pred_mean, axes=(0, 2, 1))
            x_pred_median = np.transpose(x_pred_median, axes=(0, 2, 1))
            x_true = np.transpose(x_true, axes=(0, 2, 1))

            all_pred_mean = np.append(
                all_pred_mean,
                x_pred_mean.reshape((-1, num_feature)),
                axis=0,
            )
            all_pred_median = np.append(
                all_pred_median,
                x_pred_median.reshape((-1, num_feature)),
                axis=0,
            )
            all_true = np.append(all_true, x_true.reshape((-1, num_feature)), axis=0)

    if save_predict:
        if opt.univariate:
            median_fig_name = "Revised_{}_{}_univariate/Median_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}UC90_{}UC50_{}UC10_{}W90_{}W50_{}W10_{}.jpg".format(
                opt.dataset,
                opt.pred_step,
                opt.lrG,
                opt.gp_coef_inn,
                opt.gp_coef_recons,
                opt.coef_recons,
                opt.seed,
                all_uc_90,
                all_uc_50,
                all_uc_10,
                all_pinaw_90,
                all_pinaw_50,
                all_pinaw_10,
            )

            mean_fig_name = "Revised_{}_{}_univariate/Mean_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}MSE_{}MAE_{}MAPE_{}MASE_{}CRPS_{}CD_{}.jpg".format(
                opt.dataset,
                opt.pred_step,
                opt.lrG,
                opt.gp_coef_inn,
                opt.gp_coef_recons,
                opt.coef_recons,
                opt.seed,
                all_mse,
                all_mae,
                all_mape,
                all_mase,
                all_crps,
                all_cd,
            )
            path = "{}_{}_univariate".format(opt.dataset, opt.pred_step)
        else:
            median_fig_name = "Revised_{}_{}/Median_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}UC90_{}UC50_{}UC10_{}W90_{}W50_{}W10_{}.jpg".format(
                opt.dataset,
                opt.pred_step,
                opt.lrG,
                opt.gp_coef_inn,
                opt.gp_coef_recons,
                opt.coef_recons,
                opt.seed,
                all_uc_90,
                all_uc_50,
                all_uc_10,
                all_pinaw_90,
                all_pinaw_50,
                all_pinaw_10,
            )

            mean_fig_name = "Revised_{}_{}/Mean_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}MSE_{}MAE_{}MAPE_{}MASE_{}CRPS_{}CD_{}.jpg".format(
                opt.dataset,
                opt.pred_step,
                opt.lrG,
                opt.gp_coef_inn,
                opt.gp_coef_recons,
                opt.coef_recons,
                opt.seed,
                all_mse,
                all_mae,
                all_mape,
                all_mase,
                all_crps,
                all_cd,
            )
            path = "Revised_{}_{}".format(opt.dataset, opt.pred_step)
        if not os.path.exists(path):
            os.mkdir(path)

        all_true = all_true[1:, :]
        all_pred_mean = all_pred_mean[1:, :]
        all_pred_median = all_pred_median[1:, :]

        plt.figure()
        plt.plot(all_true[:, 0], label="Ground Truth")
        plt.plot(all_pred_mean[:, 0], label="Mean Estimation")
        plt.legend()
        plt.savefig(mean_fig_name)
        plt.close()

        plt.figure()
        plt.plot(all_true[:, 0], label="Ground Truth")
        plt.plot(all_pred_median[:, 0], label="Median Estimation")
        plt.legend()
        plt.savefig(median_fig_name)
        plt.close()

    return (
        sum(MSE) / len(MSE),
        sum(MAE) / len(MAE),
        sum(Median_se) / len(Median_se),
        sum(Median_ae) / len(Median_ae),
        sum(MAPE) / len(MAPE),
        sum(MASE) / len(MASE),
        sum(CRPS) / len(CRPS),
    )


def main(opt):
    if opt.univariate:
        encoder = Generator(opt.num_feature, 1, opt.filter_size, opt.seq_len, "encoder")
        decoder = Generator(1, 1, opt.filter_size, opt.seq_len, "decoder")
        inn_discriminator = Discriminator(
            (opt.seq_len - opt.filter_size + 1), opt.hidden_dim
        )
        recons_discriminator = Discriminator(
            (opt.seq_len - 2 * (opt.filter_size - 1)), opt.hidden_dim
        )
    else:
        encoder = Generator(
            opt.num_feature, opt.num_feature, opt.filter_size, opt.seq_len, "encoder"
        )
        decoder = Generator(
            opt.num_feature, opt.num_feature, opt.filter_size, opt.seq_len, "decoder"
        )
        inn_discriminator = Discriminator(
            (opt.seq_len - opt.filter_size + 1) * opt.num_feature, opt.hidden_dim
        )
        recons_discriminator = Discriminator(
            (opt.seq_len - 2 * (opt.filter_size - 1)) * opt.num_feature, opt.hidden_dim
        )
    optimizer_generator = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=opt.lrG,
    )
    optimizer_discriminator = torch.optim.Adam(
        list(recons_discriminator.parameters()) + list(inn_discriminator.parameters()),
        lr=opt.lrD,
    )
    train_data = Custom_Dataset(
        opt.seq_len, opt.data_path, opt.dataset, "train", opt.seq_len
    )
    test_data = Custom_Dataset(
        opt.seq_len, opt.data_path, opt.dataset, "test", opt.seq_len, opt.filter_size
    )
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    iter_best_mse = float("inf")
    iter_best_mae = float("inf")
    iter_best_median_se = float("inf")
    iter_best_median_ae = float("inf")
    iter_best_mape = float("inf")
    iter_best_mase = float("inf")
    iter_best_crps = float("inf")
    for i in range(opt.epochs):
        loss_G, loss_D = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            inn_discriminator,
            recons_discriminator,
            optimizer_generator,
            optimizer_discriminator,
            opt,
        )
        print(
            "Epoch {}: Generator Loss: {}, Discriminator Loss:{}".format(
                i + 1, loss_G, loss_D
            )
        )
        if (i + 1) % 20 == 0:
            mse, mae, median_se, median_ae, mape, mase, crps_score = eval_epoch(
                test_dataloader,
                encoder,
                decoder,
                opt,
                False,
            )
            print(
                "Test result-MSE:{}, MAE:{}, Median SE:{}, Median AE:{}, MAPE:{}, MASE:{},CRPS:{}".format(
                    mse,
                    mae,
                    median_se,
                    median_ae,
                    mape,
                    mase,
                    crps_score,
                )
            )
            if mse < iter_best_mse:
                epoch = i + 1
                eval_epoch(
                    test_dataloader,
                    encoder,
                    decoder,
                    opt,
                    save_predict=True,
                )
            iter_best_mse = min(iter_best_mse, mse)
            iter_best_mae = min(iter_best_mae, mae)
            iter_best_median_se = min(iter_best_median_se, median_se)
            iter_best_median_ae = min(iter_best_median_ae, median_ae)
            iter_best_mape = min(iter_best_mape, mape)
            iter_best_mase = min(iter_best_mase, mase)
            iter_best_crps = min(iter_best_crps, crps_score)

            print(
                "Best Testing Results for this iteration at epoch {}, with MSE:{}, MAE:{}, Median SE:{}, Median AE:{}, MAPE:{}, MASE:{},CRPS:{}".format(
                    epoch,
                    iter_best_mse,
                    iter_best_mae,
                    iter_best_median_se,
                    iter_best_median_ae,
                    iter_best_mape,
                    iter_best_mase,
                    iter_best_crps,
                )
            )

    return iter_best_mse, iter_best_mae, iter_best_median_se, iter_best_median_ae


if __name__ == "__main__":
    opt = arguement()
    torch.manual_seed(opt.seed)
    print(
        "---------------------------------------------New Parameter Run----------------------------------------------"
    )
    print("[Info]-Dataset:{}, Prediction Step:{}".format(opt.dataset, opt.pred_step))
    main(opt)
