#### 2026/5/5 双支路结果可视化+压缩率输出
import os
import csv
import re
from pathlib import Path



import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.network import Mamba_encoder, Mamba_decoder
from models.channel import Channel
from data.center_block_branch_dataset import CenterBlockBranchDataset
from utils.utils import *
from utils.distortion import *
from utils.utils import seed_torch


def _to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _safe_name(name):
    """
    把文件名处理成适合保存图片的名字
    """
    name = str(name)
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
    return name


def _get_meta_value(meta, key, idx):
    """
    DataLoader 默认 collate 后，meta 可能是 list / tuple / tensor。
    这个函数用于稳定取出第 idx 个样本的 meta 信息。
    """
    value = meta[key]

    if torch.is_tensor(value):
        return value[idx].item()

    if isinstance(value, (list, tuple)):
        return value[idx]

    return value


def save_visual_result(input_image, recon_image, meta, save_dir, global_start_idx, max_save=20):
    """
    保存可视化结果：
      input_xxx.png   原图块
      recon_xxx.png   重建块
      diff_xxx.png    误差图
      compare_xxx.png [原图 | 重建 | 误差]
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    input_image = input_image.detach().cpu().clamp(0, 1)
    recon_image = recon_image.detach().cpu().clamp(0, 1)

    batch_size = input_image.shape[0]

    saved = 0
    for b in range(batch_size):
        global_idx = global_start_idx + b
        if global_idx >= max_save:
            break

        source_name = _safe_name(_get_meta_value(meta, "source_name", b))
        row = _get_meta_value(meta, "row", b)
        col = _get_meta_value(meta, "col", b)

        prefix = f"{global_idx:04d}_{source_name}_r{row}_c{col}"

        x = input_image[b:b + 1]
        y = recon_image[b:b + 1]

        diff = torch.abs(x - y)
        diff = diff / (diff.max() + 1e-8)

        # 单独保存
        vutils.save_image(x, save_dir / f"input_{prefix}.png")
        vutils.save_image(y, save_dir / f"recon_{prefix}.png")
        vutils.save_image(diff, save_dir / f"diff_{prefix}.png")

        # 横向拼接：[原图 | 重建 | 误差]
        compare = torch.cat([x, y, diff], dim=3)
        vutils.save_image(compare, save_dir / f"compare_{prefix}.png")

        saved += 1

    return saved


@torch.no_grad()
def eval_branch(config, branch_type="roi"):
    image_dir = "/home/wengyijia/datasets/MUSeg/test_official/Image_1024x896"
    grid_dir = "/home/wengyijia/datasets/MUSeg/test_official/BlockROI_binary_grid_7x8"

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

    channel = Channel(config)

    encoder_name = (
        config.TRAIN.ENCODER_PATH
        + f"{branch_type}_OUTCHANS{config.MODEL.VSSM.OUT_CHANS}"
        + f"_depth{config.MODEL.VSSM.DEPTHS}"
        + f"_embed{config.MODEL.VSSM.EMBED_DIM}"
        + f"_rsl{config.DATA.IMG_SIZE}.pt"
    )

    decoder_name = (
        config.TRAIN.DECODER_PATH
        + f"{branch_type}_OUTCHANS{config.MODEL.VSSM.OUT_CHANS}"
        + f"_depth{config.MODEL.VSSM.DEPTHS}"
        + f"_embed{config.MODEL.VSSM.EMBED_DIM}"
        + f"_rsl{config.DATA.IMG_SIZE}.pt"
    )

    print(f"[Eval] encoder checkpoint = {encoder_name}")
    print(f"[Eval] decoder checkpoint = {decoder_name}")

    # 你的 save_model 保存的是整个模型对象，所以这里直接 torch.load
    encoder = torch.load(encoder_name, map_location="cpu").cuda()
    decoder = torch.load(decoder_name, map_location="cpu").cuda()

    encoder.eval()
    decoder.eval()

    matrix = eval_matrix(config)

    # ===== 可视化输出目录 =====
    output_root = Path("/home/wengyijia/mambajscc/MambaJSCC/outputs/branch_eval")
    vis_dir = output_root / branch_type / "visual"
    output_root.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / branch_type / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n========== Eval branch: {branch_type} ==========")
    print(f"[Eval] visual results will be saved to: {vis_dir}")
    print(f"[Eval] metrics csv will be saved to: {csv_path}")

    scores = []
    total_cbr = 0.0
    total_compression_ratio = 0.0
    total_samples = 0
    saved_vis_count = 0
    max_save_vis = 20

    rows_for_csv = []

    for input_image, meta in tqdm(test_loader):
        SNR = config.CHANNEL.SNR[0]
        input_image = input_image.cuda()
        batch_size = input_image.shape[0]

        # ===== encoder =====
        feature = encoder(input_image, SNR)

        # 和 train_branch.py 保持一致的 CBR 定义
        cbr = feature.numel() / input_image.numel() / 2

        # 等效压缩率：越大表示压缩越强
        # 这里是基于项目已有 CBR 定义得到的 1/CBR
        compression_ratio = 1.0 / (cbr + 1e-12)

        # ===== channel =====
        received, pwr, h = channel.forward(feature, SNR)

        if config.CHANNEL.TYPE == "rayleigh":
            sigma_square = 1.0 / (10 ** (SNR / 10))
            received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
        elif config.CHANNEL.TYPE == "awgn":
            pass
        else:
            raise ValueError("channel type error")

        # ===== decoder =====
        received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
        recon_image = decoder(received, SNR)

        performance = matrix(recon_image, input_image)
        performance_value = _to_float(performance)
        scores.append(performance_value)

        total_cbr += cbr * batch_size
        total_compression_ratio += compression_ratio * batch_size
        total_samples += batch_size

        # ===== 保存可视化 =====
        if saved_vis_count < max_save_vis:
            saved_now = save_visual_result(
                input_image=input_image,
                recon_image=recon_image,
                meta=meta,
                save_dir=vis_dir,
                global_start_idx=saved_vis_count,
                max_save=max_save_vis,
            )
            saved_vis_count += saved_now

        # ===== 保存每个样本的指标 =====
        for b in range(batch_size):
            rows_for_csv.append({
                "branch": branch_type,
                "source_name": _get_meta_value(meta, "source_name", b),
                "row": _get_meta_value(meta, "row", b),
                "col": _get_meta_value(meta, "col", b),
                "SNR": SNR,
                "metric_PSNR_like": performance_value,
                "CBR": cbr,
                "compression_ratio_1_over_CBR": compression_ratio,
                "input_numel": int(input_image[b].numel()),
                "feature_numel_per_sample": int(feature.numel() / batch_size),
            })

    avg_metric = sum(scores) / len(scores)
    avg_cbr = total_cbr / total_samples
    avg_compression_ratio = total_compression_ratio / total_samples

    print(f"[{branch_type}] avg metric = {avg_metric:.6f}")
    print(f"[{branch_type}] avg CBR = {avg_cbr:.6f}")
    print(f"[{branch_type}] avg compression ratio = {avg_compression_ratio:.6f}")
    print(f"[{branch_type}] saved visual samples = {saved_vis_count}")

    # ===== 写 CSV =====
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "branch",
                "source_name",
                "row",
                "col",
                "SNR",
                "metric_PSNR_like",
                "CBR",
                "compression_ratio_1_over_CBR",
                "input_numel",
                "feature_numel_per_sample",
            ]
        )
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print(f"[{branch_type}] metrics saved to {csv_path}")






# import os
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from models.network import Mamba_encoder, Mamba_decoder
# from models.channel import Channel
# from data.center_block_branch_dataset import CenterBlockBranchDataset
# from utils.utils import *
# from utils.distortion import *
# from utils.utils import seed_torch

# ### 2026/5/3 wyj: 可视化
# import torchvision.utils as vutils


# @torch.no_grad()
# def eval_branch(config, branch_type="roi"):
#     image_dir = "/home/wengyijia/datasets/MUSeg/test_official/Image_1024x896"
#     grid_dir  = "/home/wengyijia/datasets/MUSeg/test_official/BlockROI_binary_grid_7x8"

#     ### 2026/5/3 wyj: 可视化
#     # ===== 可视化保存路径 =====
#     save_dir = f"./vis_{branch_type}"
#     os.makedirs(save_dir, exist_ok=True)

#     dataset = CenterBlockBranchDataset(
#         image_dir=image_dir,
#         grid_dir=grid_dir,
#         branch_type=branch_type,
#         positions=[(3, 3), (3, 4)],
#         block_h=128,
#         block_w=128,
#     )
#     test_loader = DataLoader(
#         dataset,
#         batch_size=config.DATA.TEST_BATCH,
#         shuffle=False,
#         num_workers=config.DATA.NUM_WORKERS,
#         pin_memory=True,
#         drop_last=False,
#     )

#     encoder = Mamba_encoder(config).cuda()
#     decoder = Mamba_decoder(config).cuda()
#     channel = Channel(config)

#     encoder_name = (
#         config.TRAIN.ENCODER_PATH +
#         f"{branch_type}_OUTCHANS{config.MODEL.VSSM.OUT_CHANS}_depth{config.MODEL.VSSM.DEPTHS}_embed{config.MODEL.VSSM.EMBED_DIM}_rsl{config.DATA.IMG_SIZE}.pt"
#     )
#     decoder_name = (
#         config.TRAIN.DECODER_PATH +
#         f"{branch_type}_OUTCHANS{config.MODEL.VSSM.OUT_CHANS}_depth{config.MODEL.VSSM.DEPTHS}_embed{config.MODEL.VSSM.EMBED_DIM}_rsl{config.DATA.IMG_SIZE}.pt"
#     )

#     # load_weights(encoder, encoder_name)
#     # load_weights(decoder, decoder_name)
#     # encoder.load_state_dict(torch.load(encoder_name, map_location="cpu"))
#     # decoder.load_state_dict(torch.load(decoder_name, map_location="cpu"))
#     encoder = torch.load(encoder_name, map_location="cpu").cuda()
#     decoder = torch.load(decoder_name, map_location="cpu").cuda()

#     encoder.eval()
#     decoder.eval()

#     matrix = eval_matrix(config)

#     print(f"\n========== Eval branch: {branch_type} ==========")

#     scores = []

#     for input_image, meta in tqdm(test_loader):
#         SNR = config.CHANNEL.SNR[0]
#         input_image = input_image.cuda()

#         feature = encoder(input_image, SNR)
#         received, pwr, h = channel.forward(feature, SNR)

#         if config.CHANNEL.TYPE == 'rayleigh':
#             sigma_square = 1.0 / (10 ** (SNR / 10))
#             received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
#         elif config.CHANNEL.TYPE == 'awgn':
#             pass
#         else:
#             raise ValueError("channel type error")

#         received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
#         recon_image = decoder(received, SNR)

#         performance = matrix(recon_image, input_image)
#         scores.append(performance)

#     if len(scores) > 0:
#         print(f"[{branch_type}] avg metric = {sum(scores) / len(scores)}")