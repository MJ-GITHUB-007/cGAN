import argparse
import time
import platform
import os

import torch
from torch import multiprocessing
from torch.distributed import init_process_group, destroy_process_group

from iobm.container.configs import cGAN_train_configs
from iobm.container.core import cGAN

def parse_arguments():
    parser = argparse.ArgumentParser(description='cGAN Configuration and Training')

    parser.add_argument('--data', type=str, required=True, help='Directory name containing the data')
    parser.add_argument('--model', type=str, required=False, help='Pretrained to load. Leave blank to initialize model')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training data')

    return parser.parse_args()

# Necessary code
args = parse_arguments()
configs = cGAN_train_configs(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def display_device():
    print(f"\nFound {configs.n_classes} possible classes of data: {configs.data_name}\n")
    if not torch.cuda.is_available():
        print("No GPU available\nNeed atleast one GPU for training\n")
        exit()
    else:
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]

        print(f"Found {num_gpus} GPU(s)")
        for i, gpu_name in enumerate(gpu_names):
            print(f"GPU {i}: {gpu_name}")
        print()

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl" if platform.system() == "Linux" else "gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank:int, world_size: int) -> None:

    display_device()
    ddp_setup(rank=rank, world_size=world_size)

    trainer = cGAN(
        gpu_id=rank,
        data_name=configs.data_name,
        n_classes=configs.n_classes,
        input_model=configs.input_model,
        project_path=configs.project_path,
        latent_size=configs.latent_size,
        embedding_size=configs.embedding_size,
        batch_size=configs.batch_size,
        generator_lr=configs.generator_lr,
        discriminator_lr=configs.discriminator_lr,
        lambda_gp=configs.lambda_gp
    )
    trainer.train(num_epochs=configs.epochs)
    destroy_process_group()

def run_cGAN_training():
    start_time = time.time()
    
    world_size = torch.cuda.device_count()
    multiprocessing.spawn(fn=main, args=(world_size,), nprocs=world_size)

    end_time = time.time()
    total_seconds = end_time-start_time
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = round(total_seconds % 60, 0)
    print(f"Total training time : {int(hours)} hour(s) {int(minutes)} minute(s) {int(seconds)} second(s).\n")

if __name__ == "__main__":

    run_cGAN_training()
