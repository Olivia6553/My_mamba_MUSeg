import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.network import Mamba_encoder, Mamba_decoder
from models.channel import Channel
from data.center_block_branch_dataset import CenterBlockBranchDataset
from utils.utils import *
from utils.distortion import *
from utils.utils import seed_torch


@torch.no_grad()
def eval_branch(config, branch_type="roi"):
    image_dir = "/home/wengyijia/datasets/MUSeg/test_official/Image_1024x896"
    grid_dir  = "/home/wengyijia/datasets/MUSeg/test_official/BlockROI_binary_grid_7x8"

    dataset = CenterBlockBranchDataset(
        image_dir=image_dir,
        grid_dir=grid_dir,
        branch_type=branch_type,
        positions=[(3, 3), (3, 4)],
        block_h=128,
        block_w=128,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=config.DATA.TEST_BATCH,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    encoder = Mamba_encoder(config).cuda()
    decoder = Mamba_decoder(config).cuda()
    channel = Channel(config)

    encoder_name = (
        config.TRAIN.ENCODER_PATH +
        f"{branch_type}_OUTCHANS{config.MODEL.VSSM.OUT_CHANS}_depth{config.MODEL.VSSM.DEPTHS}_embed{config.MODEL.VSSM.EMBED_DIM}_rsl{config.DATA.IMG_SIZE}.pt"
    )
    decoder_name = (
        config.TRAIN.DECODER_PATH +
        f"{branch_type}_OUTCHANS{config.MODEL.VSSM.OUT_CHANS}_depth{config.MODEL.VSSM.DEPTHS}_embed{config.MODEL.VSSM.EMBED_DIM}_rsl{config.DATA.IMG_SIZE}.pt"
    )

    # load_weights(encoder, encoder_name)
    # load_weights(decoder, decoder_name)
    # encoder.load_state_dict(torch.load(encoder_name, map_location="cpu"))
    # decoder.load_state_dict(torch.load(decoder_name, map_location="cpu"))
    encoder = torch.load(encoder_name, map_location="cpu").cuda()
    decoder = torch.load(decoder_name, map_location="cpu").cuda()

    encoder.eval()
    decoder.eval()

    matrix = eval_matrix(config)

    print(f"\n========== Eval branch: {branch_type} ==========")

    scores = []

    for input_image, meta in tqdm(test_loader):
        SNR = config.CHANNEL.SNR[0]
        input_image = input_image.cuda()

        feature = encoder(input_image, SNR)
        received, pwr, h = channel.forward(feature, SNR)

        if config.CHANNEL.TYPE == 'rayleigh':
            sigma_square = 1.0 / (10 ** (SNR / 10))
            received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
        elif config.CHANNEL.TYPE == 'awgn':
            pass
        else:
            raise ValueError("channel type error")

        received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
        recon_image = decoder(received, SNR)

        performance = matrix(recon_image, input_image)
        scores.append(performance)

    if len(scores) > 0:
        print(f"[{branch_type}] avg metric = {sum(scores) / len(scores)}")