'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
'''

#### 2026/3/31 wyj：
# 直接修 main.py

# 把这几行：

# from utils.utils import GPUManager
# gm=GPUManager()
# device_idx=gm.auto_choice(mode=3)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)

# 改成：

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#### 2026/4/2 wyj：git
#check  19:34


import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import accuracy, AverageMeter

from configs.config import get_config


from utils.utils import seed_torch

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma
from run.train import train_MambaJSCC
from run.eval import eval_MambaJSCC


#### wyj
# from utils.utils import GPUManager

# gm=GPUManager()
# device_idx=gm.auto_choice(mode=3)

# os.environ["CUDA_VISIBLE_DEVICES"] =  str(device_idx)

##4.2 学习git
os.environ["CUDA_VISIBLE_DEVICES"] = "0"






class args:
    '''
    config_name : different experiment in our paper 
    the major suppoet config in shown in the config of DIV2K
    '''
    ##config_name='DIV2K'
    config_name='MUSeg'
    project_path='/home/wengyijia/mambajscc/MambaJSCC/'  ### change here to your own path
    model_config_path = project_path + 'configs/vssm/vssm_tiny_{}.yaml'.format(config_name)
    train_config_path= project_path + 'configs/train/vssm_tiny_{}.yaml'.format(config_name)
    mode='eval'
    



def main(args):

    config = get_config(args)

    if args.mode=='train': 

        seed_torch()
        train_MambaJSCC(config) 
        seed_torch()
        eval_MambaJSCC(config)
        
    elif args.mode=='eval':

        seed_torch()
        eval_MambaJSCC(config)

main(args)
