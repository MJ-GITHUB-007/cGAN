import os
import torch
import argparse
import time

from iobm.container.configs import cGAN_generate_configs
from iobm.container.core import cGAN_Generator

def parse_arguments():
    parser = argparse.ArgumentParser(description='cGAN Configuration and Training')

    parser.add_argument('--model', type=str, required=True, help='Pretrained to load')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training data')
    parser.add_argument('--class_id', type=int, required=True, help='Type of image to generate')
    parser.add_argument('--quantity', type=int, required=True, help='Number of images to generate')

    return parser.parse_args()
  
# Necessary code
args = parse_arguments()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configs = cGAN_generate_configs(args)
print(f"\nGenerating {configs.quantity} images of class: {configs.class_name}, (id: {configs.class_id})")

if torch.cuda.is_available():
    print(f"Using GPU : {torch.cuda.get_device_name(0)}\n")
else:
    print("No GPU available, using CPU.\n")

def run_cGAN_generating() -> None:

    generator = cGAN_Generator(
        device=device,
        class_id=configs.class_id,
        quantity=configs.quantity,
        n_classes=configs.n_classes,
        project_path=configs.project_path,
        output_model=configs.output_model,
        latent_size=configs.latent_size,
        embedding_size=configs.embedding_size,
        batch_size=configs.batch_size,
        dir_num = configs.dir_num
    )
    start_time = time.time()
    generator.generate()
    end_time = time.time()
    total_seconds = end_time-start_time
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = round(total_seconds % 60, 0)
    print(f"Total generating time : {int(hours)} hour(s) {int(minutes)} minute(s) {int(seconds)} second(s).\n")

if __name__ == "__main__":
    run_cGAN_generating()