import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, IterableDataset
from typing import Callable
from data import transform
__all__ = ["CIFARGenerator", "CIFARDataset"]

N_CIFAR10 = 10_000
N_CIFAR101 = 2_000


class CIFARGenerator(IterableDataset):
    def __init__(self,
                 src_dir, 
                 transform: Callable = transform.Identity(),
                 seed: int = None,):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.src_dir = Path(src_dir)
        self.transform = transform
        self.data = {
            "nul": np.load(self.src_dir/"cifar10_data.npy"),
            "alt": np.load(self.src_dir/"cifar10.1_v6_data.npy"),}

    def __iter__(self):
        return self

    def __next__(self):
        i = self.rng.integers(N_CIFAR10).item()
        j = self.rng.integers(N_CIFAR101).item()
        nul = self.data["nul"][i]
        alt = self.data["alt"][j]
        return self.transform((nul, alt))



class CIFARDataset(Dataset):
    def __init__(self,
                 src_dir, 
                 transform: Callable = transform.Identity(),
                 seed: int = 0,):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.src_dir = Path(src_dir)
        self.transform = transform
        self.order_nul = self.rng.permutation(N_CIFAR10).tolist()
        self.order_alt = self.rng.permutation(N_CIFAR101).tolist()
        self.data = {
            "nul": np.load(self.src_dir/"cifar10_data.npy"),
            "alt": np.load(self.src_dir/"cifar10.1_v6_data.npy"),}

    def __len__(self):
        return min(N_CIFAR10, N_CIFAR101)

    def __getitem__(self, idx):
        i = self.order_nul[idx]
        j = self.order_alt[idx]
        nul = self.data["nul"][i]
        alt = self.data["alt"][j]
        return self.transform((nul, alt))

    def collate_hook(self, any):
        self.order_nul = self.rng.permutation(N_CIFAR10).tolist()
        self.order_alt = self.rng.permutation(N_CIFAR101).tolist()
        return any



class CIFARFeatureDataset(Dataset):
    def __init__(self,
                 src_dir, 
                 transform: Callable = transform.Identity(),
                 seed: int = 0,):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.src_dir = Path(src_dir)
        self.transform = transform
        self.order_nul = self.rng.permutation(N_CIFAR10).tolist()
        self.order_alt = self.rng.permutation(N_CIFAR101).tolist()
        self.data = {
            "nul": np.load(self.src_dir/"cifar10_features.npy"),
            "alt": np.load(self.src_dir/"cifar10.1_features.npy"),}

    def __len__(self):
        return min(N_CIFAR10, N_CIFAR101)

    def __getitem__(self, idx):
        i = self.order_nul[idx]
        j = self.order_alt[idx]
        nul = self.data["nul"][i]
        alt = self.data["alt"][j]
        return self.transform((nul, alt))

    def collate_hook(self, any):
        self.order_nul = self.rng.permutation(N_CIFAR10).tolist()
        self.order_alt = self.rng.permutation(N_CIFAR101).tolist()
        return any




class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, num_features=100):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights)
        feature_dim = self.model.fc.in_features    # 2048
        # modifications to increase feature noise / problem difficulty
        # self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(feature_dim, num_features)
        # get preprocessing transforms
        self.preprocess = weights.transforms()
        self.model.eval()

    def forward(self, x):
        return self.model(x)


def parse_cifar_features(src_dir, dst_dir, batch_size=10):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    featurizer = ResNet50FeatureExtractor().to(device)
    transforms = featurizer.preprocess

    img_np = dict()
    img_tensor = dict()
    # CIFAR-10
    with open(src_dir/"cifar-10-batches-py"/"test_batch", "rb") as f:
        pkl = pickle.load(f, encoding="bytes")
        img_np["cifar10"] = pkl[b"data"].reshape(-1,3,32,32)
        img_tensor["cifar10"] = torch.from_numpy(img_np["cifar10"])
    # CIFAR-10.1
    img = np.load(src_dir/"cifar10.1_v6_data.npy")
    img_np["cifar10.1"] = np.transpose(img, (0,3,1,2))  # (N, C, H, W)
    img_tensor["cifar10.1"] = torch.from_numpy(img_np["cifar10.1"])

    # extract features for test set
    features = defaultdict(list)
    with torch.no_grad():
        for i in tqdm(range(0, N_CIFAR10, batch_size)):
            img_batch = img_tensor["cifar10"][i:min(i+batch_size, N_CIFAR10)]  # (b, 3, 32, 32)
            input = transforms(img_batch)   # (b, 3, 224, 224)
            input = input.to(device)
            feature = featurizer(input)     # (b, num_features)
            features["cifar10"].append(feature.cpu().numpy())
        for i in tqdm(range(0, N_CIFAR101, batch_size)):
            img_batch = img_tensor["cifar10.1"][i:min(i+batch_size, N_CIFAR101)]  # (b, 3, 32, 32)
            input = transforms(img_batch)   # (b, 3, 224, 224)
            input = input.to(device)
            feature = featurizer(input)     # (b, num_features)
            features["cifar10.1"].append(feature.cpu().numpy())

    # concatenate features and save
    features["cifar10"] = np.concatenate(features["cifar10"], axis=0)       # (N_CIFAR10, num_features)
    features["cifar10.1"] = np.concatenate(features["cifar10.1"], axis=0)   # (N_CIFAR101, num_features)
    print(features["cifar10"].shape)
    print(features["cifar10.1"].shape)
    np.save(dst_dir/"cifar10_features.npy", features["cifar10"])
    np.save(dst_dir/"cifar10.1_features.npy", features["cifar10.1"])


def parse_cifar10_images(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    images = list()
    for i in range(1,6):
        with open(src_dir/f"data_batch_{i}", "rb") as f:
            dict = pickle.load(f, encoding="bytes")
            img_np = dict[b"data"].reshape(-1,3,32,32).transpose((0,2,3,1))
            images.append(img_np)   # (..., H, W, C)
    images = np.concatenate(images, axis=0)
    print(f"{images.shape}")
    np.save(dst_dir/"cifar10_data.npy", images)



def main():
    src_dir = Path("../cache/cifar10/raw")
    img_dir = Path("../cache/cifar10/processed")
    parse_cifar_features(src_dir, img_dir)

if __name__=='__main__':
    main()

