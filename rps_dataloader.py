"""
RPS_Dataloader: A class to manage data loading for Rock-Paper-Scissors CNN
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
from chromakey.torch import chroma_key
import random
import numpy as np
from PIL import Image
from torchvision.datasets import Places365
from torchvision.transforms.v2.functional import resize

class AugmentedLoader:
    def __init__(self, base_loader, augmentation_func):
        self.base_loader = base_loader
        self.augmentation_func = augmentation_func

    def __iter__(self):
        # Ogni volta che viene chiamato iter(), ripartiamo dall'inizio del loader originale
        for images, labels in self.base_loader:
            aug_images = self.augmentation_func(images)
            yield aug_images, labels

    def __len__(self):
        # Importante per le barre di caricamento (tqdm) e calcoli statistici
        return len(self.base_loader)
class GreenAugRandom(torch.nn.Module):
    def __init__(self, return_mask=False):
        super().__init__()
        self.return_mask = return_mask

    def forward(
        self,
        image,
        keycolor,
        background_image=None,
        tola=10,
        tolb=30,
        mask_threshold=None,
    ):
        image_out, mask = chroma_key(
            image,
            keycolor=keycolor,
            background_image=background_image,
            tola=tola,
            tolb=tolb,
        )
        if mask_threshold is not None:
            mask = (mask > mask_threshold).float()
            image_out = (image * mask[:, None, :, :]).to(torch.uint8)

        if self.return_mask:
            return image_out, mask
        return image_out

class RPS_Dataloader:
    """
    A dataloader class for Rock-Paper-Scissors image classification.
    
    Attributes:
        batch_size (int): Number of samples per batch
        img_size (tuple): Target image size (height, width)
        normalize_mean (tuple): Mean values for normalization
        normalize_std (tuple): Standard deviation values for normalization
        data_dir (str): Root directory containing train/val/test folders
    """
    
    def __init__(self, 
                 data_dir='../data',
                 batch_size=32, 
                 img_size=(200, 300),
                #  normalize_mean=(0.5, 0.5, 0.5),
                #  normalize_std=(0.5, 0.5, 0.5),
                 num_workers=0):
        """
        Initialize the RPS_Dataloader.
        
        Args:
            data_dir (str): Root directory containing train/val/test folders
            batch_size (int): Number of samples per batch. Default: 32
            img_size (tuple): Target image size (H, W). Default: (128, 128)
            normalize_mean (tuple): Mean values for normalization. Default: (0.5, 0.5, 0.5)
            normalize_std (tuple): Std dev values for normalization. Default: (0.5, 0.5, 0.5)
            num_workers (int): Number of workers for data loading. Default: 0
        """
        self.batch_size = batch_size
        self.img_size = img_size
        # self.normalize_mean = normalize_mean
        # self.normalize_std = normalize_std
        self.data_dir = data_dir
        self.num_workers = num_workers
        
        # Create transforms
        # self.transform = self._create_transform()
        
        # Load datasets
        self.train_dataset = None
        self.test_dataset = None
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self._load_datasets()
    
    # def _create_transform(self):
    #     """
    #     Create image transformation pipeline.
        
    #     Returns:
    #         transforms.Compose: Composition of transformations
    #     """
    #     transform = transforms.Compose([
    #         transforms.Resize(self.img_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize(self.normalize_mean, self.normalize_std)
    #     ])
    #     return transform
    
    def _load_datasets(self):
        """
        Load train, validation, and test datasets from disk.
        """
        try:
            # 1. Trasformazione base per calcolare le statistiche sul Train
            # Usiamo solo Resize e ToTensor per avere dati puliti
            base_transform = transforms.Compose([
                transforms.Resize(self.img_size), # Adatta alla dimensione desiderata
                transforms.ToTensor()
            ])
            
            temp_train = datasets.ImageFolder(root=f'{self.data_dir}/train', transform=base_transform)

            imgs = torch.stack([img_t for img_t ,_ in temp_train], dim = 3)

            print(f"✓ Base dataset loaded for statistics calculation: {imgs.shape} samples")
            
            mean = imgs.view(3, -1).mean(dim = 1)  # Calcola la media su tutti i canali e le immagini
            std = imgs.view(3, -1).std(dim = 1)     # Calcola la deviazione standard su tutti i canali e le immagini

            print(f"✓ Calculated mean: {mean}, std: {std}")
            self.normalize_mean = mean.tolist()
            self.normalize_std = std.tolist()
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)     # Normalizzazione
            ])
            self.train_dataset = datasets.ImageFolder(
                root=f'{self.data_dir}/train', 
                transform=self.transform
            )
            self.test_dataset = datasets.ImageFolder(
                root=f'{self.data_dir}/test', 
                transform=self.transform
            )
            print(f"✓ Datasets loaded successfully")
            print(f"  Train: {len(self.train_dataset)} samples")
            print(f"  Test:  {len(self.test_dataset)} samples")
        except Exception as e:
            print(f"✗ Error loading datasets: {e}")
            raise
    
    def _create_loaders(self, val_split=0.2):
        """
        Create DataLoader objects for train, validation, and test sets.
        """
        # Split training dataset into train and validation
        val_size = int(len(self.train_dataset) * val_split)
        train_size = len(self.train_dataset) - val_size
        self.train_subset, self.val_subset = random_split(
            self.train_dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.val_loader = DataLoader(
            self.val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def get_loaders(self):
        """
        Get all three data loaders.
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if self.train_loader is None:
            self._create_loaders()
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_train_loader(self):
        """
        Get the training data loader.
        
        Returns:
            DataLoader: Training data loader
        """
        if self.train_loader is None:
            self._create_loaders()
        return self.train_loader
    
    def get_val_loader(self):
        """
        Get the validation data loader.
        
        Returns:
            DataLoader: Validation data loader
        """
        if self.val_loader is None:
            self._create_loaders()
        return self.val_loader
    
    def get_test_loader(self):
        """
        Get the test data loader.
        
        Returns:
            DataLoader: Test data loader
        """
        if self.test_loader is None:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
        return self.test_loader
    
    def get_class_names(self):
        """
        Get the class names from the dataset.
        
        Returns:
            list: List of class names
        """
        if self.train_dataset is not None:
            return self.train_dataset.classes
        return None
    
    def get_num_classes(self):
        """
        Get the number of classes.
        
        Returns:
            int: Number of classes
        """
        if self.train_dataset is not None:
            return len(self.train_dataset.classes)
        return None
    
    def get_dataset_info(self):
        """
        Get information about the datasets.
        
        Returns:
            dict: Dictionary with dataset information
        """
        info = {
            'batch_size': self.batch_size,
            'img_size': self.img_size,
            'normalize_mean': self.normalize_mean,
            'normalize_std': self.normalize_std,
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'test_samples': len(self.test_dataset) if self.test_dataset else 0,
            'num_classes': self.get_num_classes(),
            'class_names': self.get_class_names()
        }
        return info
    
    def __repr__(self):
        """
        String representation of the dataloader.
        """
        info = self.get_dataset_info()
        return (f"RPS_Dataloader(\n"
                f"  batch_size={info['batch_size']},\n"
                f"  img_size={info['img_size']},\n"
                f"  num_classes={info['num_classes']},\n"
                f"  class_names={info['class_names']},\n"
                f"  train_samples={info['train_samples']},\n"
                f"  val_samples={info['val_samples']},\n"
                f"  test_samples={info['test_samples']}\n"
                f")")

class RPSDataLoaderAugmented:
    def __init__(self, 
                 data_dir='../data',
                 batch_size=32, 
                 img_size=(300, 200),
                #  normalize_mean=(0.5, 0.5, 0.5),
                #  normalize_std=(0.5, 0.5, 0.5),
                 num_workers=0):
        """
        Initialize the RPS_Dataloader.
        
        Args:
            data_dir (str): Root directory containing train/val/test folders
            batch_size (int): Number of samples per batch. Default: 32
            img_size (tuple): Target image size (H, W). Default: (128, 128)
            normalize_mean (tuple): Mean values for normalization. Default: (0.5, 0.5, 0.5)
            normalize_std (tuple): Std dev values for normalization. Default: (0.5, 0.5, 0.5)
            num_workers (int): Number of workers for data loading. Default: 0
        """
        self.batch_size = batch_size
        self.img_size = img_size
        # self.normalize_mean = normalize_mean
        # self.normalize_std = normalize_std
        self.data_dir = data_dir
        self.num_workers = num_workers
        
        # Load datasets
        self.train_dataset = None
        self.test_dataset = None
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self._load_datasets()
    
    
    def _load_datasets(self):
        try:
            # 1. Trasformazione base per calcolare le statistiche sul Train
            # Usiamo solo Resize e ToTensor per avere dati puliti
            base_transform = transforms.Compose([
                transforms.Resize(self.img_size), # Adatta alla dimensione desiderata
                transforms.ToTensor()
            ])
            
            temp_train = datasets.ImageFolder(root=f'{self.data_dir}/train', transform=base_transform)

            imgs = torch.stack([img_t for img_t ,_ in temp_train], dim = 3)

            print(f"✓ Base dataset loaded for statistics calculation: {imgs.shape} samples")
            
            mean = imgs.view(3, -1).mean(dim = 1)  # Calcola la media su tutti i canali e le immagini
            std = imgs.view(3, -1).std(dim = 1)     # Calcola la deviazione standard su tutti i canali e le immagini
            self.normalize_mean = mean.tolist()
            self.normalize_std = std.tolist()
            print(f"✓ Calculated mean: {mean}, std: {std}")
            # 2. TRASFORMAZIONI PER IL TRAINING (Con Augmentation)
            self.train_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),      # Augmentation
                transforms.RandomRotation(degrees=15),       # Augmentation
                transforms.ColorJitter(brightness=0.2),      # Augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)     # Normalizzazione
            ])

            # 3. TRASFORMAZIONI PER IL TEST (Senza Augmentation)
            self.test_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)     # STESSA media/std del train
            ])

            # 4. Caricamento finale
            self.train_dataset = datasets.ImageFolder(
                root=f'{self.data_dir}/train', transform=self.train_transform)
                
            self.test_dataset = datasets.ImageFolder(
                root=f'{self.data_dir}/test', transform=self.test_transform)

            print(f"✓ Datasets loaded with Data Augmentation")

        except Exception as e:
            print(f"✗ Error: {e}")
            raise 
    
    def _create_loaders(self, val_split=0.2):
        """
        Create DataLoader objects for train, validation, and test sets.
        """
        # Split training dataset into train and validation
        val_size = int(len(self.train_dataset) * val_split)
        train_size = len(self.train_dataset) - val_size
        self.train_subset, self.val_subset = random_split(
            self.train_dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.val_loader = DataLoader(
            self.val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def get_loaders(self):
        """
        Get all three data loaders.
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if self.train_loader is None:
            self._create_loaders()
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_train_loader(self):
        """
        Get the training data loader.
        
        Returns:
            DataLoader: Training data loader
        """
        if self.train_loader is None:
            self._create_loaders()
        return self.train_loader
    
    def get_val_loader(self):
        """
        Get the validation data loader.
        
        Returns:
            DataLoader: Validation data loader
        """
        if self.val_loader is None:
            self._create_loaders()
        return self.val_loader
    
    def get_test_loader(self):
            """
            Get the test data loader.
            
            Returns:
                DataLoader: Test data loader
            """
            if self.test_loader is None:
                self.test_loader = DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers
                )
            return self.test_loader
        
    def get_class_names(self):
        """
        Get the class names from the dataset.
        
        Returns:
            list: List of class names
        """
        if self.train_dataset is not None:
            return self.train_dataset.classes
        return None
    
    def get_num_classes(self):
        """
        Get the number of classes.
        
        Returns:
            int: Number of classes
        """
        if self.train_dataset is not None:
            return len(self.train_dataset.classes)
        return None
    
    def get_dataset_info(self):
        """
        Get information about the datasets.
        
        Returns:
            dict: Dictionary with dataset information
        """
        info = {
            'batch_size': self.batch_size,
            'img_size': self.img_size,
            'normalize_mean': self.normalize_mean,
            'normalize_std': self.normalize_std,
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'test_samples': len(self.test_dataset) if self.test_dataset else 0,
            'num_classes': self.get_num_classes(),
            'class_names': self.get_class_names()
        }
        return info
    
    def __repr__(self):
        """
        String representation of the dataloader.
        """
        info = self.get_dataset_info()
        return (f"RPS_Dataloader(\n"
                f"  batch_size={info['batch_size']},\n"
                f"  img_size={info['img_size']},\n"
                f"  num_classes={info['num_classes']},\n"
                f"  class_names={info['class_names']},\n"
                f"  train_samples={info['train_samples']},\n"
                f"  val_samples={info['val_samples']},\n"
                f"  test_samples={info['test_samples']}\n"
                f")")
    


class RPSDataLoaderGreenAugmented:
    def __init__(self, 
                 data_dir='../data',
                 batch_size=32, 
                 img_size=(128, 128),
                 normalize_mean=(0.5010796785354614, 0.49731189012527466, 0.40465670824050903), #mean=[0.5010796785354614, 0.49731189012527466, 0.40465670824050903], std=[0.26255369186401367, 0.2370109260082245, 0.24746748805046082]
                 normalize_std=(0.26255369186401367, 0.2370109260082245, 0.24746748805046082),
                 num_workers=0,
                 calc_normalization_stats=False):
        """
        Initialize the RPS_Dataloader.
        
        Args:
            data_dir (str): Root directory containing train/val/test folders
            batch_size (int): Number of samples per batch. Default: 32
            img_size (tuple): Target image size (H, W). Default: (128, 128)
            normalize_mean (tuple): Mean values for normalization. Default: (0.5, 0.5, 0.5)
            normalize_std (tuple): Std dev values for normalization. Default: (0.5, 0.5, 0.5)
            num_workers (int): Number of workers for data loading. Default: 0
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.data_dir = data_dir
        self.num_workers = num_workers
        
        # Load datasets
        self.train_dataset = None
        self.test_dataset = None
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.augmenter = GreenAugRandom()
        self.keycolor = '#338533'   #"#439f82"
        print("... Preparazione sfondi Places365 (split='val') ...")
        self.bg_dataset = Places365(
            root=f"{data_dir}/places_small",
            split='val',
            small=True,
            download=True
        )
        self._load_datasets(calc_normalization_stats)

    
    
    def _load_datasets(self, calc_normalization_stats=False):
        try:
            if calc_normalization_stats:
                self.normalize_mean, self.normalize_std = self._calculate_dynamic_stats()
                print(f"✓ Dynamic normalization stats: mean={self.normalize_mean}, std={self.normalize_std}")
            # 2. TRASFORMAZIONI PER IL TRAINING (Con Augmentation)
            self.train_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ])

            # 3. TRASFORMAZIONI PER IL TEST (Senza Augmentation)
            self.test_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ])

            # 4. Caricamento finale
            self.train_dataset = datasets.ImageFolder(
                root=f'{self.data_dir}/train', transform=self.train_transform)
                
            self.test_dataset = datasets.ImageFolder(
                root=f'{self.data_dir}/test', transform=self.test_transform)

            print(f"✓ Datasets loaded with Green Augmentation")

        except Exception as e:
            print(f"✗ Error: {e}")
            raise 
    


    def apply_green_aug(self, images):
        """
        Adattamento fedele del codice originale fornito
        images: Batch di mani su green screen [B, 3, H, W] in range [0, 1]
        """
        b, c, h, w = images.shape
        device = images.device
        
        # 1. Preparazione Backgrounds dal dataset Places365
        bg_list = []
        for _ in range(b):
            idx = random.randint(0, len(self.bg_dataset) - 1)
            bg_img, _ = self.bg_dataset[idx]
            
            # Converti in tensore [0, 1] e ridimensiona
            bg_t = transforms.functional.to_tensor(bg_img.convert("RGB")).to(device)
            bg_t = resize(bg_t, (h, w), antialias=True)
            bg_list.append(bg_t)
        
        background = torch.stack(bg_list) # [B, C, H, W]

        # 2. Esecuzione Augmentation (Logica originale)
        # GreenAugRandom si aspetta [B, C, H, W] in float [0, 1]
        images_aug = self.augmenter(
            images, 
            keycolor=[self.keycolor] * b, 
            tola=30, 
            tolb=35, 
            background_image=background
        )
        
        images_aug = transforms.functional.normalize(
            images_aug, 
            mean=self.normalize_mean, 
            std=self.normalize_std
        )

        return images_aug
    
    def augmented_train_loader(self):
        """Restituisce un oggetto iterabile che si comporta come un loader classico"""
        if self.train_loader is None:
            self._create_loaders()
        
        # Restituiamo il wrapper, non un generatore già avviato
        return AugmentedLoader(self.train_loader, self.apply_green_aug)

    def _create_loaders(self, val_split=0.2):
        """
        Create DataLoader objects for train, validation, and test sets.
        """
        # Split training dataset into train and validation
        val_size = int(len(self.train_dataset) * val_split)
        train_size = len(self.train_dataset) - val_size
        self.train_subset, self.val_subset = random_split(
            self.train_dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.val_loader = DataLoader(
            self.val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def get_loaders(self):
        """
        Get all three data loaders.
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if self.train_loader is None:
            self._create_loaders()
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_train_loader(self):
        """
        Get the training data loader.
        
        Returns:
            DataLoader: Training data loader
        """
        if self.train_loader is None:
            self._create_loaders()
        return self.train_loader
    
    def get_val_loader(self):
        """
        Get the validation data loader.
        
        Returns:
            DataLoader: Validation data loader
        """
        if self.val_loader is None:
            self._create_loaders()
        return self.val_loader
    
    def get_test_loader(self):
        """
        Get the test data loader.
        
        Returns:
            DataLoader: Test data loader
        """
        if self.test_loader is None:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
        return self.test_loader
    
    def get_class_names(self):
        """
        Get the class names from the dataset.
        
        Returns:
            list: List of class names
        """
        if self.train_dataset is not None:
            return self.train_dataset.classes
        return None
    
    def get_num_classes(self):
        """
        Get the number of classes.
        
        Returns:
            int: Number of classes
        """
        if self.train_dataset is not None:
            return len(self.train_dataset.classes)
        return None
    
    def get_dataset_info(self):
        """
        Get information about the datasets.
        
        Returns:
            dict: Dictionary with dataset information
        """
        info = {
            'batch_size': self.batch_size,
            'img_size': self.img_size,
            'normalize_mean': self.normalize_mean,
            'normalize_std': self.normalize_std,
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'test_samples': len(self.test_dataset) if self.test_dataset else 0,
            'num_classes': self.get_num_classes(),
            'class_names': self.get_class_names()
        }
        return info
    
    def __repr__(self):
        """
        String representation of the dataloader.
        """
        info = self.get_dataset_info()
        return (f"RPS_Dataloader(\n"
                f"  batch_size={info['batch_size']},\n"
                f"  img_size={info['img_size']},\n"
                f"  num_classes={info['num_classes']},\n"
                f"  class_names={info['class_names']},\n"
                f"  train_samples={info['train_samples']},\n"
                f"  val_samples={info['val_samples']},\n"
                f"  test_samples={info['test_samples']}\n"
                f")")
    def _calculate_dynamic_stats(self, num_batches=20):
        """
        Calcola media e std su un campione di immagini già passate attraverso 
        la sostituzione dello sfondo.
        """
        # Creiamo un loader temporaneo per il campionamento
        temp_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        all_means = []
        all_stds = []
        
        # Disabilitiamo temporaneamente la normalizzazione in apply_green_aug 
        # salvando i valori attuali
        old_mean, old_std = self.normalize_mean, self.normalize_std
        self.normalize_mean, self.normalize_std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

        for i, (imgs, _) in enumerate(temp_loader):
            if i >= num_batches: break
            
            # Applica l'augmentation (con normalizzazione neutra)
            # images_aug sarà in range [0, 1] perché mean=0 e std=1
            aug_imgs = self.apply_green_aug(imgs) 
            
            # Calcolo su questo batch: [B, C, H, W] -> [C]
            all_means.append(aug_imgs.mean(dim=[0, 2, 3]))
            all_stds.append(aug_imgs.std(dim=[0, 2, 3]))

        # Ripristiniamo (o aggiorneremo dopo il return)
        final_mean = torch.stack(all_means).mean(dim=0)
        final_std = torch.stack(all_stds).mean(dim=0)
        
        return final_mean.tolist(), final_std.tolist()