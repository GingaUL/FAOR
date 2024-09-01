import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='in_memory'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            self.filenames = sorted(os.listdir(os.path.join(root_path,'HR')))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        self.maps = []
        for filename in self.filenames:
            file = os.path.join(root_path, 'HR', filename)
            map_file = os.path.join(root_path, 'map', filename)
            
            self.files.append(transforms.ToTensor()(
                Image.open(file).convert('RGB')))
            self.maps.append(transforms.ToTensor()(
                Image.open(map_file)))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        y = self.maps[idx % len(self.files)]
    
        return x, y

@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
