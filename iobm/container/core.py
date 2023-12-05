import os
import csv

import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.nn import BCELoss

from tqdm import tqdm
import matplotlib.pyplot as plt

from iobm.container.data import DatasetCollector
from iobm.container.models import Generator, Discriminator

class cGAN():
    def __init__(
        self,
        device,
        data_name,
        n_classes,
        project_path,
        input_model,
        latent_size,
        embedding_size,
        batch_size,
        generator_lr=0.0002,
        discriminator_lr=0.0002,
        lambda_gp=10,
        display_plots=True,
    ) -> None:

        self.device = device
        self.data_name = data_name
        self.n_classes = n_classes
        self.project_path = project_path
        self.input_model = input_model
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.lambda_gp = lambda_gp
        self.display = display_plots
        self.last_epoch = 0

        dataset = DatasetCollector(data_name=self.data_name, project_path=self.project_path, rescale=True)
        self.data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.model_path = os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_model.pth')
        self.class_dict = dataset.get_dict()

        self.generator = Generator(device=self.device, latent_size=self.latent_size, embedding_size=self.embedding_size, n_classes=self.n_classes)
        self.discriminator = Discriminator(device=self.device, embedding_size=self.embedding_size, n_classes=self.n_classes)

        self.optimizer_generator = Adam(self.generator.parameters(), lr=self.generator_lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=False, amsgrad=False)
        self.optimizer_discriminator = Adam(self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=False, amsgrad=False)
        
        self.generator_loss = BCELoss()
        self.discriminator_loss = BCELoss()

        if self.input_model:
            self.train_message = f"existing cGAN model \'{os.path.basename(self.input_model)}\'"
            master_dict = torch.load(self.input_model)
            self.generator.load_state_dict(master_dict['generator_state_dict'])
            self.discriminator.load_state_dict(master_dict['discriminator_state_dict'])
            self.last_epoch = master_dict['info_dict']['last_epoch']
        else:
            self.train_message = f"cGAN model"

    def train(self, num_epochs):

        print(f"Training {self.train_message} for {num_epochs} epoch(s)...\n")
        log_file_path = f"log_{self.__get_dir_num('logs')}.csv"
        log_file_path = os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_logs', log_file_path)

        if self.display:

            plot_file_path = f"log_{self.__get_dir_num('plots')}.png"
            plot_file_path = os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_plots', plot_file_path)

            fig, axes = plt.subplots(1, 2, figsize=(9, 5))
            ax1, ax2 = axes
            D_losses, G_losses = [], []
        
        # Open a CSV file for writing the logs
        with open(log_file_path, 'a', newline='') as csvfile:

            fieldnames = ['Epoch', 'D_loss', 'G_loss']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # If the file is empty, write the headers
            if csvfile.tell() == 0:
                csv_writer.writeheader()

            for epoch in range(self.last_epoch, num_epochs+self.last_epoch):

                progress_bar = tqdm(
                    self.data_loader,
                    unit='batch',
                    total=len(self.data_loader),
                    bar_format=f'Epoch {epoch + 1}/{num_epochs+self.last_epoch} '+'|{bar:20}{r_bar}'
                )

                epoch_D_loss, epoch_G_loss = 0, 0

                for index, batch in enumerate(progress_bar):
                    real_images, labels = batch['image'], batch['label']
                    real_images = real_images
                    labels = labels
                    labels = labels.unsqueeze(1).long()

                    real_target = Variable(torch.ones(real_images.size(0), 1))
                    fake_target = Variable(torch.zeros(real_images.size(0), 1))

                    # Train Discriminator
                    self.optimizer_discriminator.zero_grad()

                    D_real_output = self.discriminator((real_images, labels))
                    D_real_loss = self.discriminator_loss(D_real_output.to(self.device), real_target.to(self.device))

                    noise_vector = Variable(torch.randn(real_images.size(0), self.latent_size))
                    noise_vector = noise_vector.to(self.device)
                    generated_image = self.generator((noise_vector, labels))

                    D_fake_output = self.discriminator((generated_image.detach(), labels))
                    D_fake_loss = self.discriminator_loss(D_fake_output.to(self.device), fake_target.to(self.device))

                    D_total_loss = (D_real_loss + D_fake_loss) / 2

                    D_total_loss.backward(retain_graph=True)
                    self.optimizer_discriminator.step()

                    # Train Generator
                    self.optimizer_generator.zero_grad()

                    G_output = self.discriminator((generated_image, labels))
                    G_loss = self.generator_loss(G_output.to(self.device), real_target.to(self.device))

                    G_loss.backward()
                    self.optimizer_generator.step()

                    progress_bar.set_postfix({
                        "D_loss": D_total_loss.item(),
                        "G_loss": G_loss.item(),
                    })

                    epoch_D_loss = D_total_loss.item()
                    epoch_G_loss = G_loss.item()
                
                if self.display:

                    D_losses.append(epoch_D_loss)
                    G_losses.append(epoch_G_loss)

                    # Update live plots
                    ax1.clear()
                    ax2.clear()

                    # Adjust x-axis values
                    x_values = list(range(self.last_epoch + 1, epoch + 2))

                    ax1.plot(x_values, D_losses, label='Discriminator Loss')
                    ax1.set_title('Discriminator loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()

                    ax2.plot(x_values, G_losses, label='Generator Loss')
                    ax2.set_title('Generator loss')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.legend()

                    plt.suptitle('Live Training Stats', fontsize=14)
                    plt.pause(0.1)

                # Append a new row to the CSV file
                csv_writer.writerow({'Epoch': epoch+1, 'D_loss': epoch_D_loss, 'G_loss': epoch_G_loss})

                print()
                self.__save_model(epoch+1)

        print(f"Training complete")
        if self.display:
            # plt.show()
            # plt.pause(0.001)
            plt.savefig(plot_file_path)
            plt.close()

    def __save_model(self, last_epoch: int=0):
        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs'))
        
        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'train')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'train'))

        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs'))
        
        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_logs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_logs'))
        
        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_plots')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_plots'))

        info_dict = {
            'last_epoch': last_epoch
        }
        
        dict_to_save = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'class_dict': self.class_dict,
            'info_dict': info_dict
        }

        torch.save(dict_to_save, self.model_path)
    
    def __get_dir_num(self, dir_type: str) -> int:
        
        if dir_type == 'logs':
            files = os.listdir(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_logs'))
        else:
            files = os.listdir(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_plots'))
        file_num = len(files)

        if file_num == 0:
            return 1
        
        try:
            nums = []
            for file in files:
                num = file.split('.')[0]
                num = num.split('_')[-1]
                num = num.strip()
                num = int(num)
                nums.append(num)
            return max(nums)+1
        except:
            raise Exception(f"Directory corrupted. Delete '{self.data_name}_logs' directory and avoid renaming of files")

class cGAN_Generator():
    def __init__(
            self,
            class_id,
            quantity,
            device,
            n_classes,
            project_path,
            output_model,
            latent_size,
            embedding_size,
            batch_size,
            dir_num
        ) -> None:

        self.class_id = class_id
        self.quantity = quantity
        self.device = device
        self.n_classes = n_classes
        self.project_path = project_path
        self.output_model = output_model
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.dir_num = dir_num

        self.generator = Generator(device=self.device, latent_size=self.latent_size, embedding_size=self.embedding_size, n_classes=self.n_classes)
        self.generator.to(self.device)
        master_dict = torch.load(self.output_model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.generator.load_state_dict(master_dict['generator_state_dict'])
    
    def generate(self):
        self.generator.eval()  # Set the generator to evaluation mode

        with torch.no_grad():
            # Generate random noise vectors
            noise_vector = torch.randn(self.quantity, self.latent_size).to(self.device)

            # Generate labels (each element is [cls_idx])
            labels = torch.full((self.quantity, 1), self.class_id).to(self.device)

            # Setting up the progress bar
            progress_bar = tqdm(
                unit=' images',
                total=self.quantity,
                bar_format=f'Generating Images '+'|{bar:20}{r_bar}'
            )

            # Generate images in batches
            generated_images_list = []
            for i in range(0, self.quantity, self.batch_size):

                batch_noise = noise_vector[i:i+self.batch_size]
                batch_labels = labels[i:i+self.batch_size]

                batch_generated_images = self.generator((batch_noise, batch_labels))
                generated_images_list.append(batch_generated_images)

                progress_bar.update(len(batch_generated_images))

            # Close the progress bar
            progress_bar.close()

            # Concatenate the image batches
            generated_images = torch.cat(generated_images_list, dim=0)

        print()

        # Save the generated images
        for idx, image in tqdm(
            enumerate(generated_images),
            unit=' images',
            total=len(generated_images),
            bar_format=f'Saving Images '+'|{bar:20}{r_bar}'
        ):
            save_path = os.path.join(self.project_path, 'cGAN_outputs', 'generate', f'g{self.dir_num} _ class_id-{self.class_id} _ quantity-{self.quantity}', f'class_{self.class_id}__{idx+1}.png')
            save_image(image, save_path)

        print("\nImages generation complete")