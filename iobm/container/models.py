import torch
from torch import nn

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