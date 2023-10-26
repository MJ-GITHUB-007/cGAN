import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import os
from tqdm import tqdm

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
            generator_lr = 0.0002,
            discriminator_lr = 0.0002,
            lambda_gp = 10
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

        dataset = DatasetCollector(data_name=self.data_name, project_path=self.project_path, rescale=True)
        self.data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.model_path = os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_generator.pth')
        self.class_dict = dataset.get_dict()

        self.generator = Generator(device=self.device, latent_size=self.latent_size, embedding_size=self.embedding_size, n_classes=self.n_classes)
        self.discriminator = Discriminator(device=self.device, embedding_size=self.embedding_size, n_classes=self.n_classes)
        if self.input_model:
            self.train_message = f"existing cGAN model \'{os.path.basename(self.input_model)}\'"
            master_dict = torch.load(self.input_model)
            self.generator.load_state_dict(master_dict['model_state_dict'])
        else:
            self.train_message = f"cGAN model"

        self.optimizer_generator = Adam(self.generator.parameters(), lr=self.generator_lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=False, amsgrad=False)
        self.optimizer_discriminator = Adam(self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=False, amsgrad=False)
        
        self.criterion_generator = nn.BCELoss()
        self.criterion_discriminator = nn.BCELoss()

    def train(self, num_epochs):

        print(f"Training {self.train_message} for {num_epochs} epoch(s)...\n")
        
        for epoch in range(num_epochs):

            progress_bar = tqdm(
                self.data_loader,
                unit='batch',
                total=len(self.data_loader),
                bar_format=f'Epoch {epoch + 1}/{num_epochs} '+'|{bar:20}{r_bar}'
            )
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
                D_real_loss = self.criterion_discriminator(D_real_output.to(self.device), real_target.to(self.device))

                noise_vector = torch.randn(real_images.size(0), self.latent_size)
                noise_vector = noise_vector.to(self.device)
                generated_image = self.generator((noise_vector, labels))

                D_fake_output = self.discriminator((generated_image.detach(), labels))
                D_fake_loss = self.criterion_discriminator(D_fake_output.to(self.device), fake_target.to(self.device))

                D_total_loss = (D_real_loss + D_fake_loss) / 2

                D_total_loss.backward()
                self.optimizer_discriminator.step()

                # Train Generator
                self.optimizer_generator.zero_grad()

                G_output = self.discriminator((generated_image, labels))
                G_loss = self.criterion_generator(G_output.to(self.device), real_target.to(self.device))

                G_loss.backward()
                self.optimizer_generator.step()

                progress_bar.set_postfix({
                    "D_loss": D_total_loss.item(),
                    "G_loss": G_loss.item(),
                })
            print()
            self.save_generator()

        print(f"Training complete")

    def save_generator(self):
        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs'))
        
        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'train')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'train'))

        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs'))
        
        dict_to_save = {
            'model_state_dict': self.generator.state_dict(),
            'info_dict': self.class_dict
        }

        torch.save(dict_to_save, self.model_path)

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
        self.generator.load_state_dict(master_dict['model_state_dict'])
    
    def generate(self):
        self.generator.eval()  # Set the generator to evaluation mode

        with torch.no_grad():
            # Generate random noise vectors
            noise_vector = Variable(torch.randn(self.quantity, self.latent_size)).to(self.device)

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
            save_path = os.path.join(self.project_path, 'cGAN_outputs', 'generate', f'generation_{self.dir_num}', f'class_{self.class_id}__{idx+1}.png')
            save_image(image, save_path)

        print("\nImages generation complete")