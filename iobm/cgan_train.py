import torch
import argparse
import time

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
print(f"\nFound {configs.n_classes} possible classes of data: {configs.data_name}")

if torch.cuda.is_available():
    print(f"Using GPU : {torch.cuda.get_device_name(0)}\n")
else:
    print("No GPU available, using CPU.\n")

def run_cGAN_training() -> None:

    trainer = cGAN(
        device=device,
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
    start_time = time.time()
    trainer.train(num_epochs=configs.epochs)
    end_time = time.time()
    total_seconds = end_time-start_time
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = round(total_seconds % 60, 0)
    print(f"Total training time : {int(hours)} hour(s) {int(minutes)} minute(s) {int(seconds)} second(s).\n")

if __name__ == "__main__":
    run_cGAN_training()
