import os
from configs.config import get_config
from utils.utils import seed_torch
from run.infer_roi_dual_full import infer_roi_dual_full


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


PROJECT_PATH = "/root/autodl-tmp/mambajscc/MambaJSCC/"


class RoiArgs:
    model_config_path = PROJECT_PATH + "configs/vssm/vssm_tiny_MUSeg.yaml"
    train_config_path = PROJECT_PATH + "configs/train/vssm_tiny_MUSeg.yaml"


class BgArgs:
    model_config_path = PROJECT_PATH + "configs/vssm/vssm_tiny_MUSeg_bg.yaml"
    train_config_path = PROJECT_PATH + "configs/train/vssm_tiny_MUSeg.yaml"


def main():
    seed_torch()

    roi_config = get_config(RoiArgs)
    bg_config = get_config(BgArgs)

    infer_roi_dual_full(
        roi_config=roi_config,
        bg_config=bg_config,
        image_dir="/root/autodl-tmp/datasets/MUSeg/test_official/Image_1024x896",
        coarse_roi_ckpt="/root/autodl-tmp/mambajscc/MambaJSCC/checkpoints/coarse_roi/best_coarse_roi_finetune.pth",
        output_dir="/root/autodl-tmp/mambajscc/MambaJSCC/outputs/roi_dual_full",
        threshold=0.5,
        image_start=400,
        max_images=20,
    )


if __name__ == "__main__":
    main()