import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image

# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
# AddGaussianNoise(0., 0.1)
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ImgDataset(Dataset):
    """Image dataset."""
    # mean=[0.4944, 0.2425, 0.0836], std=[0.3459, 0.1717, 0.0803] for fundus 224 299 img
    # mean=[0.6053, 0.2904, 0.1010], std=[0.3762, 0.1948, 0.0892] for fundus seed_78 for 384_384 img
    # 
    # mean=[0.0228, 0.0228, 0.0228], std: [0.0534, 0.0534, 0.0534] for sc2img

    def __init__(self, csv_file, root_dir, resample=False, 
                 transform=T.Compose(
        [T.ToPILImage(), T.ToTensor(),
        T.Normalize(mean=[0.6053, 0.2904, 0.1010], std=[0.3762, 0.1948, 0.0892]) ])):
                 # 
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        if resample: self.labels = self.labels.sample(frac=0.8)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])
        image = read_image(img_name)
        labels = self.labels.iloc[idx, 1:]
        # image -> transform -> 3channel
        if image.shape[0]==1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0]>3:
            image = image[:3,:,:]
        if self.transform: image = self.transform(image)
                
        labels = labels.to_numpy()
        labels = torch.from_numpy(labels.astype(int))
        labels = torch.squeeze(labels)
        return image, labels
    

class ImgDataset_withimagename(Dataset):
    """Image dataset."""
    # mean=[0.4944, 0.2425, 0.0836], std=[0.3459, 0.1717, 0.0803] for fundus 224 299 img
    # mean=[0.6053, 0.2904, 0.1010], std=[0.3762, 0.1948, 0.0892] for fundus seed_78 for 384_384 img
    # 
    # mean=[0.0228, 0.0228, 0.0228], std: [0.0534, 0.0534, 0.0534] for sc2img

    def __init__(self, csv_file, root_dir,
                 transform=T.Compose(
        [T.ToPILImage(), T.ToTensor(), 
                 T.Normalize(mean=[0.6053, 0.2904, 0.1010], std=[0.3762, 0.1948, 0.0892]) ])):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        if resample: self.labels = self.labels.sample(frac=0.8)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])
        image = read_image(img_name)
        labels = self.labels.iloc[idx, 1:]
        # image -> transform -> 3channel
        if image.shape[0]==1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0]>3:
            image = image[:3,:,:]
        if self.transform: image = self.transform(image)
                
        labels = labels.to_numpy()
        labels = torch.from_numpy(labels.astype(int))
        labels = torch.squeeze(labels)
        return image, labels, self.labels.iloc[idx, 0]
    
class ImgDataset_withaugment(Dataset):
    """Image dataset."""
    # mean=[0.4944, 0.2425, 0.0836], std=[0.3459, 0.1717, 0.0803] for fundus 224 299 img
    # mean=[0.6053, 0.2904, 0.1010], std=[0.3762, 0.1948, 0.0892] for fundus seed_78 for 384_384 img
    # 
    # mean=[0.0228, 0.0228, 0.0228], std: [0.0534, 0.0534, 0.0534] for sc2img

    def __init__(self, csv_file, root_dir, resample=False, rodegree=0, randcolor=False, crop=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        if resample: self.labels = self.labels.sample(frac=0.8)
        self.root_dir = root_dir
        tfmslist = [T.ToPILImage(), T.ToTensor(),
        T.Normalize(mean=[0.6053, 0.2904, 0.1010], std=[0.3762, 0.1948, 0.0892])]
        
        # the order of those T makes difference
        if rodegree: tfmslist.insert(1, T.RandomRotation(degrees=rodegree))
        if crop:
            if crop=='rand':
                tfmslist.insert(1, T.RandomCrop(size=384))
            elif crop=='center':
                tfmslist.insert(1, T.CenterCrop(size=384))
        
        if randcolor: tfmslist.insert(0, T.ColorJitter(brightness=.2, hue=.1))
        self.transform = T.Compose(tfmslist)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])
        image = read_image(img_name)
        labels = self.labels.iloc[idx, 1:]
        # image -> transform -> 3channel
        if image.shape[0]==1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0]>3:
            image = image[:3,:,:]
        if self.transform: image = self.transform(image)
                
        labels = labels.to_numpy()
        labels = torch.from_numpy(labels.astype(int))
        labels = torch.squeeze(labels)
        return image, labels
    