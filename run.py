import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'], # logs/
                               name=config['model_params']['name'],) #VanillaVAE

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params']) #定义模型架构VanillaVAE(in_channel=3, latent_dim=128)
experiment = VAEXperiment(model,
                          config['exp_params']) # 模型和各种运行参数（比如学习率，权重衰减，scheduler_gamma，kld_weight

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0) #定义数据集，继承自LightningDataModule，数据集路径，batch_size，patch_size,

data.setup()
runner = Trainer(logger=tb_logger, #设置日志记录器，这里是TensorBoard日志记录器，将训练过程中的指标和可视化信息记录到TensorBoard中
                 callbacks=[
                     LearningRateMonitor(), #内置的回调函数，用于监视学习率并将其记录到TensorBoard中，以便可以可视化的跟踪学习率的变化
                     ModelCheckpoint(save_top_k=2, #内置的回调函数，用于在训练过程中保存模型检查点，save_top_k=2保留两个性能最好的模型检查点
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), #检查点文件的保存目录
                                     monitor= "val_loss", #指定了用于监视性能的指标，这里使用验证集上的损失
                                     save_last= True), # 表示保存最后一个模型检查点
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),#strategy参数用于配置分布式训练策略，这里使用DDP分布式数据并行策略
                 **config['trainer_params']) # 用于传递其他与Trainer相关的配置参数，包括训练的epoch等


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True) # 在logs/VanillaVAE/version_9路径下创建一个名为Samples的目录
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)