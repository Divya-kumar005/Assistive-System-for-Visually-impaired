import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FlickrDataset(Dataset):
    def __init__(self, image_dir, captions_dict, tokenizer, max_length):
        self.image_dir = image_dir
        self.captions = captions_dict
        self.images = list(captions_dict.keys())
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption = self.captions[img_name][0]
        seq = self.tokenizer.texts_to_sequences([caption])[0]

        seq = seq[:self.max_length]
        seq += [0] * (self.max_length - len(seq))

        return image, torch.tensor(seq)