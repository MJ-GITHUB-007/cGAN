from torch.utils.data import Dataset
from torchvision import transforms

import os
import pickle
from PIL import Image

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