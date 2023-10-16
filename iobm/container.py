import torch
from torch import nn, autograd
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import os
import pickle
import argparse
from tqdm import tqdm
from PIL import Image

class Configs():
    def __init__(self) -> None:
        self.image_size = 256
        self.latent_size = 100
        self.embedding_size = 100
        self.generator_lr = 0.001
        self.discriminator_lr = 0.001
        self.lambda_gp = 10

class cGAN_train_configs(Configs):
    def __init__(self, args) -> None:
        super(cGAN_train_configs, self).__init__()

        self.model = args.model
        self.project_path = os.getcwd()
        self.root_path = os.path.join(self.project_path, args.data)
        self.input_model = os.path.join(self.project_path, self.model) if self.model else None
        self.n_classes = len(os.listdir(self.root_path))
        self.data_name = args.data
        self.epochs = args.epochs
        self.batch_size = args.batch_size

        self.__is_positive()
        self.__check_args()
        if not self.__is_image_directory(self.root_path):
            raise Exception(f"Data directory not structured properly")

        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs'))
        
        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'train')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'train'))

        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs'))

        self.model_path = os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_generator.pth')
    
    def __check_args(self) -> None:
        
        if self.model:
            if self.model.split('.')[-1] !='pth':
                raise TypeError(f"Expected a .pth file for model")
            
            if not os.path.exists(self.input_model):
                raise Exception(f"Model specified doesn't exists")
            
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                loaded_generator = Generator(device=device, latent_size=self.latent_size, embedding_size=self.embedding_size, n_classes=self.n_classes)
                master_dict = torch.load(self.input_model)
                loaded_generator.load_state_dict(master_dict['model_state_dict'])
                loaded_n = len(master_dict['info_dict'])
                if self.n_classes != loaded_n:
                    raise Exception("Number of classes mismatch in model")
            except:
                print(self.input_model)
                raise Exception(f"Couldn't load the model")
        
    def __is_positive(self) -> None:
        if (
            self.epochs <= 0 or
            self.batch_size <= 0 or
            self.latent_size <= 0 or
            self.embedding_size <= 0 or
            self.generator_lr <=0 or
            self.discriminator_lr <= 0
        ):
            raise argparse.ArgumentTypeError(f"Expecting positive values of input arguments")
    
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

class cGAN_generate_configs(Configs):
    def __init__(self, args) -> None:
        super(cGAN_generate_configs, self).__init__()

        self.model = args.model
        self.project_path = os.getcwd()
        self.output_model = os.path.join(self.project_path, self.model) if self.model else None
        self.class_id = args.class_id
        self.quantity = args.quantity
        self.batch_size = args.batch_size

        master_dict = torch.load(self.output_model)
        self.n_classes = len(master_dict['info_dict'])
        self.__is_bounded()
        self.class_name = master_dict['info_dict'][self.class_id]
        self.__check_model()
        self.__check_args()

        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs'))
        
        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'generate')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'generate'))

        self.dir_num = self.__get_dir_num()

        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'generate', f'generation_{self.dir_num}')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'generate', f'generation_{self.dir_num}'))

    def __check_model(self) -> None:

        if self.model.split('.')[-1] !='pth':
            raise TypeError(f"Expected a .pth file for model")
        
        if not os.path.exists(self.output_model):
            raise Exception(f"Model specified doesn't exists")

    def __check_args(self) -> None:
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            loaded_generator = Generator(device=device, latent_size=self.latent_size, embedding_size=self.embedding_size, n_classes=self.n_classes)
            master_dict = torch.load(self.output_model)
            loaded_generator.load_state_dict(master_dict['model_state_dict'])
            loaded_n = len(master_dict['info_dict'])
            if self.n_classes != loaded_n:
                raise Exception("Number of classes mismatch in model")
        except:
            print(self.output_model)
            raise Exception(f"Couldn't load the model")
        
    def __is_bounded(self) -> None:
        if (
            self.class_id < 0 or
            self.quantity <= 0 or
            self.batch_size <= 0 or
            self.latent_size <= 0 or
            self.embedding_size <= 0 or
            self.generator_lr <=0 or
            self.discriminator_lr <= 0
        ):
            raise argparse.ArgumentTypeError(f"Expecting positive values of input arguments")
        
        if self.class_id >= self.n_classes:
            raise argparse.ArgumentTypeError(f"Expecting value of class_id between 0 and {self.n_classes - 1}")
    
    def __get_dir_num(self) -> int:
        
        dirs = os.listdir(os.path.join(self.project_path, 'cGAN_outputs', 'generate'))
        dir_num = len(dirs)

        if dir_num == 0:
            return 1
        
        nums = []
        for dir in dirs:
            num = dir.split('_')[-1]
            num = int(num)
            nums.append(num)
        return max(nums)+1

class DatasetCollector(Dataset):
    def __init__(
            self,
            data_name,
            project_path,
            rescale=True
        ):
        self.data_name = data_name
        self.project_path = project_path
        self.root_path = os.path.join(self.project_path, self.data_name)
        self.image_size = 256

        self.original_dict = {cls: idx for idx, cls in enumerate(sorted(os.listdir(self.root_path)))}
        self.class_dict = {value: key for key, value in self.original_dict.items()}

        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs'))
        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'train')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'train'))
        if not os.path.exists(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs')):
            os.mkdir(os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs'))
        
        self.dict_path = os.path.join(self.project_path, 'cGAN_outputs', 'train', f'{self.data_name}_outs', f'{self.data_name}_index2class.pkl')
        
        with open(self.dict_path, 'wb') as file_obj:
            pickle.dump(self.class_dict, file_obj)

        self.images = self.load_images()
        if rescale:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()
            ])

    def load_images(self):
        images = []
        for class_name in sorted(os.listdir(self.root_path)):
            class_folder = os.path.join(self.root_path, class_name)
            class_idx = self.original_dict[class_name]
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
    
    def get_dict(self):
        return self.class_dict

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
            device,
            latent_size,
            embedding_size,
            n_classes,
            *args,
            **kwargs
        ) -> None:
        super(Generator, self).__init__(*args, **kwargs)
        self.device = device
        self.latent_size = latent_size
        self.embedding_size = embedding_size
        self.n_classes = n_classes

        self.label_conditioned_generator = nn.Sequential(
            nn.Embedding(num_embeddings=self.n_classes, embedding_dim=self.embedding_size),
            nn.Linear(in_features=self.embedding_size, out_features=16)
        ).to(self.device)

        self.latent = nn.Sequential(
            nn.Linear(in_features=self.latent_size, out_features=4*4*512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ).to(self.device)

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
        ).to(device)
    
    def forward(self, inputs):
        # get noise and label
        noise_vector, label = inputs
        noise_vector, label = noise_vector.to(self.device), label.to(self.device)

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
            device,
            embedding_size,
            n_classes,
            *args,
            **kwargs
        ) -> None:
        super(Discriminator, self).__init__(*args, **kwargs)
        self.device = device
        self.embedding_size = embedding_size
        self.n_classes = n_classes

        self.image_scaler = TensorScaler(scale_factor=2/255.0, offset=-1.0)

        self.label_condition_disc = nn.Sequential(
                nn.Embedding(num_embeddings=self.n_classes, embedding_dim=self.embedding_size),
                nn.Linear(in_features=self.embedding_size, out_features=3*256*256)
        ).to(self.device)

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
        ).to(self.device)
    
    def forward(self, inputs):
        # getting image and label
        img, label = inputs
        img, label = img.to(self.device), label.to(self.device)

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
        master_dict = torch.load(self.output_model)
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
                unit='images',
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

        print(f"\nImage Generation complete")
        print("Saving images")

        # Save the generated images
        for idx, image in enumerate(generated_images):
            save_path = os.path.join(self.project_path, 'cGAN_outputs', 'generate', f'generation_{self.dir_num}', f'class_{self.class_id}__{idx+1}.png')
            save_image(image, save_path)

        print("Images saved")

if __name__ == '__main__':
    pass