import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.network import Mamba_encoder, Mamba_decoder
from models.channel import Channel
from data.center_block_branch_dataset import CenterBlockBranchDataset
from utils.utils import *
from utils.distortion import *
from utils.utils import seed_torch


def train_branch(config, branch_type="roi"):
    # ===== 固定随机种子，必须放在 DataLoader 和模型初始化之前 =====
    seed_torch()

    print(f"[Train] branch type = {branch_type}")
    print(f"[Train] channel type = {config.CHANNEL.TYPE}")
    print(f"[Train] encoder save path = {config.TRAIN.ENCODER_PATH}")
    print(f"[Train] decoder save path = {config.TRAIN.DECODER_PATH}")

    # ===== 数据路径 =====
    image_dir = "/root/autodl-tmp/datasets/MUSeg/train_official/Image_1024x896"
    grid_dir  = "/root/autodl-tmp/datasets/MUSeg/train_official/BlockROI_binary_grid_7x8"

    dataset = CenterBlockBranchDataset(
        image_dir=image_dir,
        grid_dir=grid_dir,
        branch_type=branch_type,
        positions=[(3, 3), (3, 4)],
        block_h=128,
        block_w=128,
    )
    def seed_worker(worker_id):
        worker_seed = 1024 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(1024)

    train_loader = DataLoader(
        dataset,
        batch_size=config.DATA.TRAIN_BATCH,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    encoder = Mamba_encoder(config).cuda()
    decoder = Mamba_decoder(config).cuda()
    channel = Channel(config)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    encoder_params = count_params(encoder)
    decoder_params = count_params(decoder)
    total_params = encoder_params + decoder_params
    trainable_params = sum(
        p.numel()
        for p in list(encoder.parameters()) + list(decoder.parameters())
        if p.requires_grad
    )
    
    print(f"Encoder params: {encoder_params / 1e6:.4f} M")
    print(f"Decoder params: {decoder_params / 1e6:.4f} M")
    print(f"Total params: {total_params / 1e6:.4f} M")
    print(f"Trainable params: {trainable_params / 1e6:.4f} M")

    
    optimizer_encoder = optim.AdamW(encoder.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=1e-4)
    optimizer_decoder = optim.AdamW(decoder.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=1e-4)

    cosine_encoder = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_encoder, T_max=config.TRAIN.EPOCHS, eta_min=0, last_epoch=-1
    )
    warm_encoder = GradualWarmupScheduler(
        optimizer=optimizer_encoder, multiplier=2.0, warm_epoch=0.1, after_scheduler=cosine_encoder
    )

    cosine_decoder = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_decoder, T_max=config.TRAIN.EPOCHS, eta_min=0, last_epoch=-1
    )
    warm_decoder = GradualWarmupScheduler(
        optimizer=optimizer_decoder, multiplier=2.0, warm_epoch=0.1, after_scheduler=cosine_decoder
    )

    criterion = loss_matrix(config)
    matrix = eval_matrix(config)

    encoder.train()
    decoder.train()

    print(f"\n========== Train branch: {branch_type} ==========")
    print(config.MODEL.VSSM.EMBED_DIM, config.MODEL.VSSM.DEPTHS)

    # seed_torch()

    for e in range(config.TRAIN.EPOCHS):
        loss_ave = 0.0

        with tqdm(train_loader, dynamic_ncols=False) as pbar:
            for i, (input_image, meta) in enumerate(pbar):
                SNR_list = config.CHANNEL.SNR
                SNR_index = torch.randint(0, len(SNR_list), (1,)).item()
                SNR = SNR_list[SNR_index]

                input_image = input_image.cuda()

                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()

                # encoder
                feature = encoder(input_image, SNR)
                CBR = feature.numel() / input_image.numel() / 2

                # channel
                received, pwr, h = channel.forward(feature, SNR)

                if config.CHANNEL.TYPE == 'rayleigh':
                    sigma_square = 1.0 / (10 ** (SNR / 10))
                    received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
                elif config.CHANNEL.TYPE == 'awgn':
                    pass
                else:
                    raise ValueError("channel type error")

                # decoder
                received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
                recon_image = decoder(received, SNR)

                loss = criterion(recon_image, input_image, feature, opt_idx=0, global_step=e)
                loss.backward()

                performance = matrix(recon_image, input_image)
                loss_ave += loss.item()

                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

                optimizer_encoder.step()
                optimizer_decoder.step()

                pbar.set_postfix({
                    'branch': branch_type,
                    'e': e,
                    'loss': (loss.item(), loss_ave / (i + 1)),
                    'matrix': performance,
                    'CBR': CBR,
                    'SNR': SNR,
                    'LR': optimizer_encoder.state_dict()['param_groups'][0]["lr"],
                })

        warm_encoder.step()
        warm_decoder.step()

        if (e + 1) % config.TRAIN.SAVE_FRE == 0:
            save_model(
                encoder,
                save_path=config.TRAIN.ENCODER_PATH +
                f"{branch_type}_OUTCHANS{config.MODEL.VSSM.OUT_CHANS}_depth{config.MODEL.VSSM.DEPTHS}_embed{config.MODEL.VSSM.EMBED_DIM}_rsl{config.DATA.IMG_SIZE}.pt"
            )
            save_model(
                decoder,
                save_path=config.TRAIN.DECODER_PATH +
                f"{branch_type}_OUTCHANS{config.MODEL.VSSM.OUT_CHANS}_depth{config.MODEL.VSSM.DEPTHS}_embed{config.MODEL.VSSM.EMBED_DIM}_rsl{config.DATA.IMG_SIZE}.pt"
            )