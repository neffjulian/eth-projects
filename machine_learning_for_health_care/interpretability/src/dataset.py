import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split
import os
import cv2

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split


class ImageDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        # Define data transform
        train_transform = []
        if self.transform is not None:
            train_transform+=self.transform
        train_transform += [
                transforms.Resize(128),             # resize shortest side to 128 pixels
                transforms.CenterCrop(128),         # crop longest side to 128 pixels at center
                transforms.ToTensor()               # convert PIL image to tensor
        ]
        train_transform = transforms.Compose(train_transform)
        test_transform=transforms.Compose([
                transforms.Resize(128),             # resize shortest side to 128 pixels
                transforms.CenterCrop(128),         # crop longest side to 128 pixels at center
                transforms.ToTensor()               # convert PIL image to tensor
        ])
        
        # Initialize train/test sets
        data_path = Path("../data/images")
        train_dataset = ImageFolder(data_path, transform=train_transform)
        test_dataset = ImageFolder(data_path, transform=test_transform)
        classes = train_dataset.find_classes(data_path)[1]
        print(f"Loaded samples into dataset with label 'no'={classes['no']} and 'yes'={classes['yes']}")
        
        # Split dataset into train/test sets and stratify over labels to balance datasets with set seed 
        # DO NOT CHANGE THE SEED
        generator = torch.Generator().manual_seed(390397)
        train_len = int(0.8*len(train_dataset))
        test_len = int((len(train_dataset)-train_len)/2)
        train_dataset = random_split(
            dataset=train_dataset, 
            lengths=[train_len, test_len, test_len],
            generator=generator)[0]
        val_dataset, test_dataset = random_split(
            dataset=test_dataset, 
            lengths=[train_len, test_len, test_len],
            generator=generator)[1:]
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class PyradiomicsDataset(Dataset):
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.data = data.astype(np.float64).values
        self.labels = labels

    def __len__(self):
        return self.labels.__len__()

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


class PyradiomicsDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Define relevant features 
        rel_feat = ['diagnostics_Versions_PyRadiomics', 
                    'diagnostics_Versions_Numpy', 
                    'diagnostics_Versions_SimpleITK', 
                    'diagnostics_Versions_PyWavelet', 
                    'diagnostics_Versions_Python', 
                    'diagnostics_Configuration_Settings', 
                    'diagnostics_Configuration_EnabledImageTypes', 
                    'diagnostics_Image-original_Hash', 
                    'diagnostics_Image-original_Dimensionality', 
                    'diagnostics_Image-original_Spacing', 
                    'diagnostics_Image-original_Size', 
                    'diagnostics_Image-original_Mean', 
                    'diagnostics_Image-original_Minimum', 
                    'diagnostics_Image-original_Maximum', 
                    'diagnostics_Mask-original_Hash', 
                    'diagnostics_Mask-original_Spacing', 
                    'diagnostics_Mask-original_Size', 
                    'diagnostics_Mask-original_BoundingBox', 
                    'diagnostics_Mask-original_VoxelNum', 
                    'diagnostics_Mask-original_VolumeNum', 
                    'diagnostics_Mask-original_CenterOfMassIndex', 
                    'diagnostics_Mask-original_CenterOfMass']
        
        # Load train/test sets from csvs
        data_path = "../data/radiomics"
        train_data = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
        train_data.drop(inplace=True,axis=1,labels=rel_feat)
        train_labels = np.load(os.path.join(data_path, 'train_labels.npy'))
        val_data = pd.read_csv(os.path.join(data_path, 'validation_data.csv'))
        val_data.drop(inplace=True,axis=1,labels=rel_feat)
        val_labels = np.load(os.path.join(data_path, 'validation_labels.npy'))
        test_data = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
        test_data.drop(inplace=True,axis=1,labels=rel_feat)
        test_labels = np.load(os.path.join(data_path, 'test_labels.npy'))

        self.train_set = PyradiomicsDataset(train_data, train_labels)
        self.test_set = PyradiomicsDataset(test_data, test_labels)
        self.val_set = PyradiomicsDataset(val_data, val_labels)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

def get_radiomics_dataset():
    # Define relevant features 
    rel_feat = ['diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet', 'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings', 'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash', 'diagnostics_Image-original_Dimensionality', 'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 'diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass']
    
    # Load train/test sets from csvs
    data_path = "../data/radiomics"
    train_data = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
    train_data.drop(inplace=True,axis=1,labels=rel_feat)
    train_labels = np.load(os.path.join(data_path, 'train_labels.npy'))
    val_data = pd.read_csv(os.path.join(data_path, 'validation_data.csv'))
    val_data.drop(inplace=True,axis=1,labels=rel_feat)
    val_labels = np.load(os.path.join(data_path, 'validation_labels.npy'))
    test_data = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
    test_data.drop(inplace=True,axis=1,labels=rel_feat)
    test_labels = np.load(os.path.join(data_path, 'test_labels.npy'))
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

class SIFTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        label, feature = self.data[idx]
        feature = np.expand_dims(feature, axis=0)
        return torch.tensor(feature, dtype=torch.float), torch.tensor(label, dtype=torch.long)

class SIFTDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        print("SIFT DataModule: setup...")
        data_path = "../data/images"

        data = []
        for file in os.listdir(data_path + "/yes"):
            img = cv2.imread(data_path + "/yes/" + file)
            img2 = cv2.flip(img, 0)
            sift = cv2.SIFT_create()
            kp, desc = sift.detectAndCompute(img, None)
            features = np.resize(desc, (256, 256))
            label = np.array([1])
            data.append([label, features])
            img = cv2.flip(img, 0)
            kp, desc = sift.detectAndCompute(img, None)
            features = np.resize(desc, (256, 256))
            label = np.array([1])
            data.append([label, features])


        for file in os.listdir(data_path + "/no"):
            img = cv2.imread(data_path + "/no/" + file)
            kp, desc = sift.detectAndCompute(img, None)
            features = np.resize(desc, (256, 256))
            label = np.array([0])
            data.append([label, features])
            img = cv2.flip(img, 0) 
            kp, desc = sift.detectAndCompute(img, None)
            features = np.resize(desc, (256, 256))
            label = np.array([0])
            data.append([label, features])

        train, test = train_test_split(data, test_size= 0.1)
        train, val = train_test_split(train, test_size=0.15)

        self.train_set = SIFTDataset(train)
        self.test_set = SIFTDataset(test)
        self.val_set = SIFTDataset(val)

        print("SIFT Datamodule: ...complete!")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)