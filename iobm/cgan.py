import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import argparse
import pickle
from PIL import Image
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='cGAN Configuration and Training')

    parser.add_argument('--data', type=str, required=True, help='Directory name containing the data')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training data')
    parser.add_argument('--latent_size', type=int, default=100, help='Size of the latent space')
    parser.add_argument('--embedding_size', type=int, default=100, help='Size of class embedding')
    parser.add_argument('--generator_lr', type=float, default=0.001, help='Learning rate for the generator')
    parser.add_argument('--discriminator_lr', type=float, default=0.001, help='Learning rate for the discriminator')

    return parser.parse_args()

class cGAN_configs():
    def __init__(self, args) -> None:
        self.data_name = args.data
        self.project_path = os.getcwd()
        self.root_path = os.path.join(self.project_path, args.data)
        
        if not self.__is_image_directory(self.root_path):
            raise Exception(f"Data directory not structured properly")

        if not os.path.exists(os.path.join(self.project_path, 'cGAN outputs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN outputs'))
        if not os.path.exists(os.path.join(self.project_path, 'cGAN outputs', f'{self.data_name}_outs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN outputs', f'{self.data_name}_outs'))

        self.model_path = os.path.join(self.project_path, 'cGAN outputs', f'{self.data_name}_outs', f'{self.data_name}_generator.pth')
        self.dict_path = os.path.join(self.project_path, 'cGAN outputs', f'{self.data_name}_outs', f'{self.data_name}_class2index.pkl')

        try:
            self.n_classes = len(os.listdir(self.root_path))
        except Exception as e:
            raise ValueError(f'Error in getting number of classes: {e}')

        sample_image_path = self.__get_sample_image_path()
        sample_image = Image.open(sample_image_path)
        sample_image_shape = sample_image.size
        min_size = min(sample_image_shape[0], sample_image_shape[1])

        if min_size > 256:
            raise Exception(f"Image size {min_size} is too small\nExpecting images_size of atleast 256")

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.image_size = 256
        self.latent_size = args.latent_size
        self.embedding_size = args.embedding_size
        self.__is_positive()

        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr
    
    def __is_positive(self) -> None:
        if (
            self.epochs <= 0 or
            self.batch_size <= 0 or
            self.latent_size <= 0 or
            self.embedding_size <= 0
        ):
            raise ValueError(f"Expecting positive values of input arguments")
    
    def __is_image_directory(self, path) -> bool:
        if not os.path.isdir(path):
            return False

        image_extensions = set()
        for entry in os.listdir(path):

            entry_path = os.path.join(path, entry)
            if not os.path.isdir(entry_path):
                return False
            
            files_in_subdir = []
            for file in os.listdir(entry_path):
                if not os.path.isfile(os.path.join(entry_path, file)):
                    return False
                
                files_in_subdir.append(file.lower())
            
            if not files_in_subdir:
                return False  # Subdirectory is empty
            
            subdir_extensions = {file.split('.')[-1] for file in files_in_subdir}
            if len(subdir_extensions) == 1:
                image_extensions.update(subdir_extensions)
            else:
                return False  # Subdirectories have different image extensions

        return len(image_extensions) == 1
    
    def __get_sample_image_path(self):
        # Modify this method based on your actual directory structure and logic to get a sample image path
        sample_class_dir = os.path.join(self.root_path, os.listdir(self.root_path)[0])
        sample_image = os.path.join(sample_class_dir, os.listdir(sample_class_dir)[0])
        return sample_image
    
# Necessary code
args = parse_arguments()
configs = cGAN_configs(args)
print(f"\nFound {configs.n_classes} possible classes of data: {configs.data_name}")

class DatasetCollector(Dataset):
    def __init__(self, rescale=True):
        self.class_to_idx = {cls: idx for idx, cls in enumerate(os.listdir(configs.root_path))}

        if not os.path.exists(os.path.join(configs.project_path, 'cGAN outputs')):
            os.mkdir(os.path.join(configs.project_path, 'cGAN outputs'))
        if not os.path.exists(os.path.join(configs.project_path, 'cGAN outputs', f'{configs.data_name}_outs')):
            os.mkdir(os.path.join(configs.project_path, 'cGAN outputs', f'{configs.data_name}_outs'))
        
        with open(configs.dict_path, 'wb') as file_obj:
            pickle.dump(self.class_to_idx, file_obj)

        self.images = self.load_images()
        if rescale:
            self.transform = transforms.Compose([
                transforms.Resize((configs.image_size, configs.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((configs.image_size, configs.image_size)),
                transforms.ToTensor()
            ])

    def load_images(self):
        images = []
        for class_name in sorted(os.listdir(configs.root_path)):
            class_folder = os.path.join(configs.root_path, class_name)
            class_idx = self.class_to_idx[class_name]
            for filename in os.listdir(class_folder):
                image_path = os.path.join(class_folder, filename)
                images.append((image_path, class_idx))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, class_idx = self.images[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': class_idx}

# Define the custom rescale layer class
class TensorScaler(nn.Module):
    def __init__(self, scale_factor: int, offset: int):
        super(TensorScaler, self).__init__()
        self.scale_factor = scale_factor
        self.offset = offset

    def forward(self, x):
        # Scale the tensor and apply an offset
        return x * self.scale_factor + self.offset

# Define Generator class
class Generator(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
        ) -> None:
        super(Generator, self).__init__(*args, **kwargs)

        self.label_conditioned_generator = nn.Sequential(
            nn.Embedding(num_embeddings=configs.n_classes, embedding_dim=configs.embedding_size),
            nn.Linear(in_features=configs.embedding_size, out_features=16)
        )

        self.latent = nn.Sequential(
            nn.Linear(in_features=configs.latent_size, out_features=4*4*512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.model = nn.Sequential(
            # 4x4 to 8x8
            nn.ConvTranspose2d(
                in_channels=513,
                out_channels=64*8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=64*8,
                momentum=0.1,
                eps=0.8
            ),
            nn.ReLU(inplace=True),

            # 8x8 to 16x16
            nn.ConvTranspose2d(
                in_channels=64*8,
                out_channels=64*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=64*4,
                momentum=0.1,
                eps=0.8
            ),
            nn.ReLU(inplace=True),

            # 16x16 to 32x32
            nn.ConvTranspose2d(
                in_channels=64*4,
                out_channels=64*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=64*2,
                momentum=0.1,
                eps=0.8
            ),
            nn.ReLU(inplace=True),

            # 32x32 to 64x64
            nn.ConvTranspose2d(
                in_channels=64*2,
                out_channels=64*1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=64*1,
                momentum=0.1,
                eps=0.8
            ),
            nn.ReLU(inplace=True),

            # 64x64 to 128x128
            nn.ConvTranspose2d(
                in_channels=64*1,
                out_channels=10,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=10,
                momentum=0.1,
                eps=0.8
            ),
            nn.ReLU(inplace=True),

            # 128x128 to 256x256
            nn.ConvTranspose2d(
                in_channels=10,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh(),
            TensorScaler(scale_factor=255/2.0, offset=255/2.0)
        )
    
    def forward(self, inputs):
        # get noise and label
        noise_vector, label = inputs

        # converting label 1x1x1 to 1x4x4
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)

        # converting latent 512x1x1 to 512x4x4
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512,4,4)

        # converting matrix 512x1x1 to image 3, 256, 256
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        #print(image.size())
        return image

# Define Generator class
class Discriminator(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
        ) -> None:
        super(Discriminator, self).__init__(*args, **kwargs)

        self.image_scaler = TensorScaler(scale_factor=2/255.0, offset=-1.0)

        self.label_condition_disc = nn.Sequential(
                nn.Embedding(num_embeddings=configs.n_classes, embedding_dim=configs.embedding_size),
                nn.Linear(in_features=configs.embedding_size, out_features=3*256*256)
        )

        self.model = nn.Sequential(
            # 256x256 to 128x128
            nn.Conv2d(
                in_channels=6,
                out_channels=64*1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 128x128 to 43x43
            nn.Conv2d(
                in_channels=64*1,
                out_channels=64*2,
                kernel_size=4,
                stride=3,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=64*2,
                momentum=0.1,
                eps=0.8
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 43x43 to 15x15
            nn.Conv2d(
                in_channels=64*2,
                out_channels=64*4,
                kernel_size=4,
                stride=3,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=64*4,
                momentum=0.1,
                eps=0.8
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 15x15 to 6x6
            nn.Conv2d(
                in_channels=64*4,
                out_channels=64*6,
                kernel_size=4,
                stride=3,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=64*6,
                momentum=0.1,
                eps=0.8
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 6x6 to 3x3
            nn.Conv2d(
                in_channels=64*6,
                out_channels=64*8,
                kernel_size=4,
                stride=3,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=64*8,
                momentum=0.1,
                eps=0.8
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_features=4608, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        # getting image and label
        img, label = inputs

        # scaling down image
        img = self.image_scaler(img)

        # getting label encoded
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 3, 256, 256)

        # concatenating image and encoded label
        concat = torch.cat((img, label_output), dim=1)
        #print(concat.size())

        # getting output
        output = self.model(concat)
        return output

class cGAN():
    def __init__(self) -> None:
        dataset = DatasetCollector(rescale=True)
        self.train_state = False

        self.data_loader = DataLoader(dataset=dataset, batch_size=configs.batch_size, shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion_generator = nn.BCELoss()
        self.criterion_discriminator = nn.BCELoss()

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.optimizer_generator = Adam(self.generator.parameters(), lr=configs.generator_lr)
        self.optimizer_discriminator = Adam(self.discriminator.parameters(), lr=configs.discriminator_lr)

    def train(self, num_epochs):
        
        for epoch in range(num_epochs):

            progress_bar = tqdm(self.data_loader, total=len(self.data_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', position=0, leave=True, ncols=150)

            for index, batch in enumerate(progress_bar):
                real_images, labels = batch['image'], batch['label']
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)
                labels = labels.unsqueeze(1).long()

                real_target = Variable(torch.ones(real_images.size(0), 1).to(self.device))
                fake_target = Variable(torch.zeros(real_images.size(0), 1).to(self.device))

                # Train Discriminator
                self.optimizer_discriminator.zero_grad()

                D_real_output = self.discriminator((real_images, labels))
                D_real_loss = self.criterion_discriminator(D_real_output, real_target)

                noise_vector = torch.randn(real_images.size(0), configs.latent_size, device=self.device)
                noise_vector = noise_vector.to(self.device)
                generated_image = self.generator((noise_vector, labels))

                D_fake_output = self.discriminator((generated_image.detach(), labels))
                D_fake_loss = self.criterion_discriminator(D_fake_output, fake_target)

                D_total_loss = (D_real_loss + D_fake_loss) / 2

                D_total_loss.backward()
                self.optimizer_discriminator.step()

                # Train Generator
                self.optimizer_generator.zero_grad()

                G_output = self.discriminator((generated_image, labels))
                G_loss = self.criterion_generator(G_output, real_target)

                G_loss.backward()
                self.optimizer_generator.step()

                progress_bar.set_postfix({
                    "D_loss": D_total_loss.item(),
                    "G_loss": G_loss.item(),
                })
            print()
            self.train_state = True

    def save_generator(self):
        if not os.path.exists(os.path.join(configs.project_path, 'cGAN outputs')):
            os.mkdir(os.path.join(configs.project_path, 'cGAN outputs'))
        if not os.path.exists(os.path.join(configs.project_path, 'cGAN outputs', f'{configs.data_name}_outs')):
            os.mkdir(os.path.join(configs.project_path, 'cGAN outputs', f'{configs.data_name}_outs'))

        if not self.train_state:
            raise Exception('Train generator first')
        torch.save(self.generator.state_dict(), configs.model_path)

def run_cGAN() -> None:
    try:
        args = parse_arguments()
        cgan_config = cGAN_configs(args)
    except Exception:
        print(Exception)

    trainer = cGAN()
    print(f"Training cGAN model for {cgan_config.epochs} epoch(s)...\n")
    trainer.train(num_epochs=cgan_config.epochs)
    print(f"Training complete")
    trainer.save_generator()
    print(f"Generator saved as \"{f'{configs.data_name}_generator.pth'}\"\n")

if __name__ == "__main__":
    run_cGAN()