import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model import Generator, Discriminator
from data_loader import Custom_Dataset
from utils import calculate_gradient_penalty, metrics


def arguement():
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_path", type=str, required=True)
    parser.add_argument("-dataset", type=str, required=True)
    parser.add_argument("-data_bad", type=str, required=False)
    parser.add_argument("-pred_step", type=int, default=1)
    parser.add_argument("-test_perc", type=float, default=0.2)

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
    parser.add_argument("-lrD", type=float, default=1e-3)
    parser.add_argument("-lrG", type=float, default=1e-3)
    parser.add_argument("-num_critic", type=int, default=1e-3)

    parser.add_argument("-gp_coef_inn", type=float, default=1)
    parser.add_argument("-coef_recons", type=float, default=1)
    parser.add_argument("-gp_coef_recons", type=float, default=1)
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-sample_size",type=int,default=1000)

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
        optimizer_discriminator.zero_grad()

        inn = encoder(x_input)
        x_recons = decoder(inn)
        inn_real = 2 * torch.rand(inn.shape) - 1

        inn_fake_output = inn_discriminator(inn)
        inn_real_output = inn_discriminator(inn_real)
        inn_score_real = inn_real_output.mean().item()
        inn_score_fake = inn_fake_output.mean().item()

        recons_fake_output = recons_discriminator(x_recons)
        remaining_length = x_recons.shape[2]
        recons_real_output = recons_discriminator(x_input[:, :, -remaining_length:])
        recons_score_real = recons_real_output.mean().item()
        recons_score_fake = recons_fake_output.mean().item()

        inn_gradient_penalty = calculate_gradient_penalty(
            inn_discriminator, inn_real, inn
        )
        recons_gradient_penalty = calculate_gradient_penalty(
            recons_discriminator, x_input[:, :, -remaining_length:], x_recons
        )

        loss_discriminator = (
            inn_score_fake
            - inn_score_real
            + opt.gp_coef_inn * inn_gradient_penalty
            + opt.coef_recons
            * (
                +recons_score_fake
                - recons_score_real
                + opt.gp_coef_recons * recons_gradient_penalty
            )
        )

        loss_D.append(loss_discriminator.item())

        loss_discriminator.backward()
        optimizer_discriminator.step()

        for i in range(opt.num_critic):
            optimizer_generator.zero_grad()

            inn = encoder(x_input)
            x_recons = decoder(inn)

            inn_fake_output = inn_discriminator(inn)
            recons_fake_output = recons_discriminator(x_recons)

            loss_generator = (
                -inn_fake_output.mean() - opt.coef_recons * recons_fake_output.mean()
            )
            loss_generator.backward()
            optimizer_generator.step()

            loss_G.append(loss_generator.item())

    return sum(loss_G) / len(loss_G), sum(loss_D) / len(loss_D)


def eval_epoch(
    test_dataloader,
    encoder,
    decoder,
    inn_discriminator,
    recons_discriminator,
    opt,
):
    MSE=[]
    MAE=[]
    Median_se=[]
    Median_ae=[]
    for x_input, x_true in test_dataloader:
        inn = encoder(x_input)
        step = opt.pred_step
        decoder_in_len = opt.seq_len - 2 * opt.filter_size + 2  # 12
        print("step=", step)
        inn_row_num = opt.seq_len - opt.filter_size + 1

        recons_future_test_all = np.zeros((opt.sample_size, opt.batch_size, decoder_in_len))
        print("recons_future_test_all", recons_future_test_all.shape)

        x_pred_median = np.empty((opt.batch_size, decoder_in_len))
        x_pred_mean = np.empty((opt.batch_size, decoder_in_len))
        for row in range(opt.batch_size):
            for j in range(decoder_in_len):
                inn_test_temp = np.matlib.repmat(inn[row].copy(), opt.sample_size, 1)
                inn_test_temp[:, j + opt.filter_size - step:j + opt.filter_size] = 2 * torch.rand((opt.sample_size, step)) - 1
                decoder_out = decoder(inn_test_temp)
                x_pred_median[row, j] = np.median(decoder_out[:, j], axis=0)
                x_pred_mean[row, j] = np.mean(decoder_out[:, j], axis=0)

        mse, mae, median_se, median_ae = metrics(x_true, x_pred_mean,x_pred_median)
        MSE.append(mse)
        MAE.append(mae)
        Median_se.append(median_se)
        Median_ae.append(median_ae)

    return (
        sum(MSE) / len(MSE),
        sum(MAE) / len(MAE),
        sum(Median_se) / len(Median_se),
        sum(Median_ae) / len(Median_ae),
    )



def main(opt):
    encoder = Generator(
        opt.num_feature, opt.num_feature, opt.filter_size, opt.seq_len, "encoder"
    )
    decoder = Generator(
        opt.num_feature, opt.num_feature, opt.filter_size, opt.seq_len, "decoder"
    )
    inn_discriminator = Discriminator(opt.seq_len - opt.filter_size + 1, opt.hidden_dim)
    recons_discriminator = Discriminator(
        opt.seq_len - 2 * (opt.filter_size - 1), opt.hidden_dim
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
        opt.seq_len, opt.data_path, opt.dataset, "train", opt.pred_step
    )
    test_data = Custom_Dataset(
        opt.seq_len, opt.data_path, opt.dataset, "test", opt.pred_step
    )
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    iter_best_mse = float("inf")
    iter_best_mae = float("inf")
    iter_best_median_se = float("inf")
    iter_best_median_ae = float("inf")
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
                i, loss_G, loss_D
            )
        )
        mse, mae, median_se, median_ae = eval_epoch(
            test_dataloader,
            encoder,
            decoder,
            inn_discriminator,
            recons_discriminator,
            opt,
        )
        print(
            "Test result-MSE:{}, MAE:{}, Median SE:{}, Median AE:{}".format(
                mse, mae, median_se, median_ae
            )
        )
    #     iter_best_mse = min(iter_best_mse, mse)
    #     iter_best_mae = min(iter_best_mae, mae)
    #     iter_best_median_se = min(iter_best_median_se, median_se)
    #     iter_best_median_ae = min(iter_best_median_ae, median_ae)
    #
    # return iter_best_mse, iter_best_mae, iter_best_median_se, iter_best_median_ae

    return None


if __name__ == "__main__":
    opt = arguement()
    main(opt)