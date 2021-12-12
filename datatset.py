import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resize


def load_images(root_path: str):
    """return paths for all images found in folder and subfolders"""
    images = []
    assert os.path.isdir(root_path), '%s is not a valid directory' % root_path

    for root, _, fnames in sorted(os.walk(root_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class SuperResolutionDataset(Dataset):
    def __init__(self, root, transforms=None, resize: int = 4):
        self.resize = resize
        self.files = load_images(root)
        self.transforms = transforms or ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_hr = self.transforms(img)
        size = [img_hr.size(1) // self.resize, img_hr.size(2) // self.resize]
        img_lr = resize(img_hr, size, interpolation=Image.BICUBIC)

        return img_lr, img_hr

    def __len__(self):
        return len(self.files)
