import os
import argparse

import torch

from iobm.container.models import Generator

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

        master_dict = torch.load(self.output_model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
            master_dict = torch.load(self.output_model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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