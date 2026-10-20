import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        root,
        train=True,
        seq_len=20,
        image_size=28,
        velocity_range_x=(-3, 3),
        velocity_range_y=(-3, 3),
        num_digits=2,
        transform=None,
        download=True,
        random=True,
        seed=42
    ):
        super().__init__()
        self.mnist = MNIST(root=root, train=train, download=download)
        self.seq_len = seq_len
        self.image_size = image_size
        self.vx_min, self.vx_max = velocity_range_x
        self.vy_min, self.vy_max = velocity_range_y
        self.num_digits = num_digits
        self.transform = transform
        self.to_tensor = ToTensor()
        self.random = random
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        imgs = []
        labels = []
        for n in range(self.num_digits):
            if self.random:
                idx = np.random.randint(0, len(self.mnist))
            else:
                idx = self.rng.randint(0, len(self.mnist))
            img_pil, lbl = self.mnist[idx]
            img = self.to_tensor(img_pil)
            imgs.append(img)
            labels.append(lbl)

        velocities_x = []
        velocities_y = []
        positions_x = []
        positions_y = []

        seq = torch.zeros(
            self.seq_len, 1, self.image_size, self.image_size, dtype=imgs[0].dtype
        )

        for _ in range(self.num_digits):
            if self.random:
                vx = np.random.randint(self.vx_min, self.vx_max + 1)
                vy = np.random.randint(self.vy_min, self.vy_max + 1)
            else:
                vx = self.rng.randint(self.vx_min, self.vx_max + 1)
                vy = self.rng.randint(self.vy_min, self.vy_max + 1)
            velocities_x.append(vx)
            velocities_y.append(vy)

            if self.random:
                x0 = np.random.randint(0, self.image_size)
                y0 = np.random.randint(0, self.image_size)
            else:
                x0 = self.rng.randint(0, self.image_size)
                y0 = self.rng.randint(0, self.image_size)
            positions_x.append(x0)
            positions_y.append(y0)

        for t in range(self.seq_len):
            frame = torch.zeros(1, self.image_size, self.image_size)

            for i, img in enumerate(imgs):
                pad_size = (self.image_size - img.shape[1]) // 2
                padded_img = F.pad(img, (pad_size, pad_size, pad_size, pad_size), "constant", 0)

                if padded_img.shape[1] < self.image_size:
                    padded_img = F.pad(padded_img, (0, self.image_size - padded_img.shape[2], 0, self.image_size - padded_img.shape[1]), "constant", 0)

                img = padded_img

                shift_x = positions_x[i] + velocities_x[i] * t
                shift_y = positions_y[i] + velocities_y[i] * t

                moved = torch.roll(
                    img,
                    shifts=(shift_y, shift_x),
                    dims=(1, 2),
                )
                frame += moved

            seq[t] = frame.clamp(0.0, 1.0)

        if self.transform:
            seq = self.transform(seq)

        if self.num_digits == 1:
            return seq, labels[0]
        else:
            return seq, torch.tensor(labels, dtype=torch.long)


class FixedVelocityMovingMNIST(MovingMNISTDataset):
    def __init__(self, vx, vy, **kwargs):
        super().__init__(velocity_range_x=(vx, vx),
                         velocity_range_y=(vy, vy),
                         **kwargs)
