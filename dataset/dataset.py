from distutils.command.config import config
import json
import os
import random

from scipy.ndimage import label
from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import os
from torchvision.transforms.functional import hflip, resize

import math
import random
from random import random as rand

class DGM4_Dataset(Dataset):

    def __init__(self, config, ann_file, transform, max_words=30, is_train=True):

        self.root_dir = '/home/async/data-disk/zxy/deepfake/datasets'
        self.ann = []
        self.noise_root = None

        if 'noise_image_root' in config:
            self.noise_root = config['noise_image_root']

        for f in ann_file:
            data = json.load(open(f, 'r'))
            for ann in data:
                if 'noiseimage' not in ann:
                    base_name = os.path.basename(ann['image'])
                    noise_path = os.path.join(self.noise_root, base_name) if self.noise_root else ann['image'].replace(
                        'DGM4', 'DGM4+noise', 1)
                    ann['noiseimage'] = noise_path
                self.ann.append(ann)

        if 'dataset_division' in config:
            division_factor = config['dataset_division']
            self.ann = self.ann[:int(len(self.ann) / division_factor)]

        self.transform = transform
        self.max_words = max_words
        self.image_res = config.get('image_res', 224)
        self.is_train = is_train

    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)

    def __getitem__(self, index):
        ann = self.ann[index]

        orig_img_dir = ann['image']
        orig_image_path = os.path.join(self.root_dir, orig_img_dir)

        noise_img_dir = ann['noiseimage']
        noise_image_path = os.path.join(self.root_dir, noise_img_dir)

        try:
            orig_image = Image.open(orig_image_path).convert('RGB')
            noise_image = Image.open(noise_image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            raise

        orig_W, orig_H = orig_image.size
        noise_W, noise_H = noise_image.size
        fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        has_bbox = False
        try:
            if 'fake_image_box' in ann and len(ann['fake_image_box']) == 4:
                xmin, ymin, xmax, ymax = map(int, ann['fake_image_box'])
                if xmax > xmin and ymax > ymin:
                    x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin
                    has_bbox = True
        except (KeyError, IndexError, TypeError):
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)

        do_hflip = False
        if self.is_train and random.random() < 0.5:
            orig_image = orig_image.transpose(Image.FLIP_LEFT_RIGHT)
            noise_image = noise_image.transpose(Image.FLIP_LEFT_RIGHT)
            do_hflip = True

        orig_image = orig_image.resize((self.image_res, self.image_res), Image.BILINEAR)
        noise_image = noise_image.resize((self.image_res, self.image_res), Image.BILINEAR)

        orig_image = self.transform(orig_image)
        noise_image = self.transform(noise_image)

        if has_bbox:
            if do_hflip:
                x = (orig_W - x) - w

            x = self.image_res / orig_W * x
            w = self.image_res / orig_W * w
            y = self.image_res / orig_H * y
            h = self.image_res / orig_H * h

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([
                center_x / self.image_res,
                center_y / self.image_res,
                w / self.image_res,
                h / self.image_res
            ], dtype=torch.float)

        label = ann['fake_cls']

        caption = pre_caption(ann['text'], self.max_words)

        fake_text_pos_list = torch.zeros(self.max_words)
        fake_text_pos = ann.get('fake_text_pos', [])
        for pos in fake_text_pos:
            if pos < self.max_words:
                fake_text_pos_list[pos] = 1


        return {
            'orig_image': orig_image,
            'noise_image': noise_image,
            'label': label,
            'caption': caption,
            'fake_image_box': fake_image_box,
            'fake_text_pos': fake_text_pos_list,
        }