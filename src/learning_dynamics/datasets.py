import pickle
import torch

class PendulumDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.thetas = data["thetas"]
            self.winds = data["winds"]
            self.frames = data["frames"]
            self.time = data["time"]
            self.dt = data["dt"]
            
        _, _, self.height, self.width = self.frames.shape
        self.transform = transform

    def __getitem__(self, index):
        frames = torch.tensor(self.frames[index]).double()
        thetas = torch.tensor(self.thetas[index]).double()
        if self.transform is not None: 
            return self.transform((frames, thetas))
        return frames, thetas

    def __len__(self):
        return self.thetas.shape[0]

class ShallowWaterDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.frames = data["frames"]
            self.time = data["time"]
            self.dt = data["dt"]
            
        _, _, self.height, self.width = self.frames.shape
        self.transform = transform

    def __getitem__(self, index):
        frames = torch.tensor(self.frames[index]).double()
        if self.transform is not None: 
            return self.transform(frames)
        return frames

    def __len__(self):
        return self.frames.shape[0]