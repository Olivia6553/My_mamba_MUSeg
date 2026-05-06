import os
from configs.config import get_config
from utils.utils import seed_torch
from run.train_branch import train_branch
from run.eval_branch import eval_branch


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class args:
    branch = "roi"   # "roi" or "bg"
    mode = "eval"  # "train" or "eval"

    project_path = "/home/wengyijia/mambajscc/MambaJSCC/"

    # ROI 分支用原配置，BG 分支用瘦身配置
    if branch == "roi":
        model_config_path = project_path + "configs/vssm/vssm_tiny_MUSeg.yaml"
    else:
        model_config_path = project_path + "configs/vssm/vssm_tiny_MUSeg_bg.yaml"

    train_config_path = project_path + "configs/train/vssm_tiny_MUSeg.yaml"


def main(args):
    config = get_config(args)

    if args.mode == "train":
        seed_torch()
        train_branch(config, branch_type=args.branch)
    elif args.mode == "eval":
        seed_torch()
        eval_branch(config, branch_type=args.branch)
    else:
        raise ValueError("mode must be train or eval")


if __name__ == "__main__":
    main(args)