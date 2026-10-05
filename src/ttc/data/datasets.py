import json
import os
import pickle
from abc import ABC
from collections import defaultdict
from glob import glob
from pathlib import Path

import torch
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .class_names import (
    CALTECH101_CLASS_NAME,
    DESCRIBABLE_TEXTURES_CLASS_NAME,
    EUROSAT_CLASS_NAME,
    FGVC_CLASS_NAME,
    FLOWERS102_CLASS_NAME,
    FOOD101_CLASS_NAME,
    IMAGENET_CLASS_NAME,
    OXFORD_IIIT_PETS_CLASS_NAME,
    STANFORDCARS_CLASS_NAME,
    SUN397_CLASS_NAME,
    UCF101_CLASS_NAME,
)
from .prompts import (
    CALTECH101_PROMPT,
    DESCRIBABLE_TEXTURES_PROMPT,
    EUROSAT_PROMPT,
    FGVC_PROMPT,
    FLOWERS102_PROMPT,
    FOOD101_PROMPT,
    IMAGENET_PROMPT,
    OXFORD_IIIT_PETS_PROMPT,
    STANFORDCARS_PROMPT,
    SUN397_PROMPT,
    UCF101_PROMPT,
)

DATA_DIR = Path(__file__).resolve().parent


class VLMDataset(ABC):
    root: str
    dataset_path: str
    n_class: int
    class_name: list
    _num2name: dict
    _name2num: dict
    prompt: list
    imgs: list
    targets: list

    def __init__(self, root, imgs, targets, class_name_list, transform, target_transform, n_shot, seed=1):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.n_shot = n_shot
        self.seed = seed

        self.set_class_name(class_name_list)
        self.origin_imgs, self.origin_targets = imgs, targets
        self.sampling(n_shot)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def prompt(self):
        return [
            lambda c: f'a photo {c}.'
        ]

    def set_class_name(self, class_name_list):
        self.class_name = class_name_list
        self._num2name = {i: name for i, name in enumerate(class_name_list)}
        self._name2num = {name: i for i, name in enumerate(class_name_list)}

    def num2str(self, num):
        return self._num2name.get(num, f'{num} does not exist in {self.dataset_path}')

    def str2num(self, class_name):
        return self._name2num.get(class_name, f'{class_name} does not exist in {self.dataset_path}')

    def _data_dict(self):
        data_dict = defaultdict(list)
        for i in range(len(self.origin_imgs)):
            data_dict[self.origin_targets[i]].append(self.origin_imgs[i])
        return data_dict

    def sampling(self, n_shot):
        self.n_shot = n_shot

        if n_shot == 0:
            self.imgs, self.targets = self.origin_imgs, self.origin_targets
            return

        sample_path = os.path.join(self.root, f'sampling_seed{self.seed}', f'soonge_{self.dataset_name}_{n_shot}s.pkl')
        with open(sample_path, 'rb') as f:
            data = pickle.load(f)
            self.imgs, self.targets = data['s_imgs'], data['s_targets']
            self.imgs = [os.path.join(self.root, self.dataset_path, x) for x in self.imgs]

    @staticmethod
    def loader(path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __str__(self):
        return f'{self.__class__.__name__} | # class: {self.n_class} | # root: {self.dataset_path} | prompt: {self.prompt}'

    @staticmethod
    def _split_warning(dataset_name, split, state):
        if split is not state:
            print(f'{colored("[DATASET_WARNING]", "red")} {dataset_name} is not supported "split" argument.')


imagenet_a_class_number = [
    6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108,
    110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307,
    308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386,
    397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483,
    486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606,
    607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758,
    763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845,
    847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954,
    956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988
]

imagenet_r_class_number = [
    1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113,
    122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207,
    208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289,
    291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344,
    347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437,
    441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594,
    596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820,
    824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951,
    953, 954, 957, 963, 965, 967, 980, 981, 983, 988
]


class ImageNetX(VLMDataset, Dataset, ABC):
    dataset_path = 'imageNet-X'
    n_class = 1000
    class_number = range(0, 1000)

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = ImageFolder(os.path.join(root, self.dataset_path))
        class_name_list = [IMAGENET_CLASS_NAME[i] for i in self.class_number]
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return IMAGENET_PROMPT

    def project_logits(self, logits):
        if logits.shape[-1] == self.n_class:
            return logits
        return logits[:, self.class_number]


class ImageNetR(ImageNetX):
    dataset_path = 'imageNet-R'
    n_class = 200
    class_number = imagenet_r_class_number


class ImageNetA(ImageNetX):
    dataset_path = 'imageNet-A'
    n_class = 200
    class_number = imagenet_a_class_number


class ImageNetSketch(ImageNetX):
    dataset_path = 'imageNet-Sketch'

    def project_logits(self, logits):
        return logits


class ImageNetV2(VLMDataset, Dataset):
    dataset_path = 'imageNet-V2'
    n_class = 1000

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        imgs, targets = list(), list()
        for sample in sorted(glob(os.path.join(root, self.dataset_path, '*/*'))):
            imgs.append(sample)
            targets.append(int(os.path.basename(os.path.dirname(sample))))
        class_name_list = IMAGENET_CLASS_NAME
        super().__init__(root, imgs, targets, class_name_list, transform, target_transform, n_shot)

    @property
    def prompt(self):
        return IMAGENET_PROMPT


class ObjectNet(VLMDataset, Dataset):
    dataset_path = 'objectnet-1.0'
    n_class = 113

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        _, _, self.folders_to_ids, self.classname_map = self.get_metadata(root)

        self.class_name = sorted(list(self.folders_to_ids.keys()))
        self.rev_class_idx_map = dict()
        self.class_idx_map = dict()
        for idx, name in enumerate(self.class_name):
            self.rev_class_idx_map[idx] = self.folders_to_ids[name]
            for imagenet_idx in self.rev_class_idx_map[idx]:
                self.class_idx_map[imagenet_idx] = idx

        self.class_name = [self.classname_map[c].lower() for c in self.class_name]

        imgs, targets = list(), list()
        for idx, class_name in enumerate(sorted(list(self.folders_to_ids.keys()))):
            class_img = glob(os.path.join(root, self.dataset_path, 'images', class_name, '*'))
            imgs.extend(class_img)
            targets.extend([idx for _ in range(len(class_img))])

        super().__init__(root, imgs, targets, IMAGENET_CLASS_NAME, transform, target_transform, n_shot)

    def get_metadata(self, root):
        metadata = Path(os.path.join(root, self.dataset_path, 'mappings'))

        with open(metadata / 'folder_to_objectnet_label.json', 'r') as f:
            folder_map = json.load(f)
            folder_map = {v: k for k, v in folder_map.items()}
        with open(metadata / 'objectnet_to_imagenet_1k.json', 'r') as f:
            objectnet_map = json.load(f)

        with open(metadata / 'pytorch_to_imagenet_2012_id.json', 'r') as f:
            pytorch_map = json.load(f)
            pytorch_map = {v: k for k, v in pytorch_map.items()}

        with open(metadata / 'imagenet_to_label_2012_v2', 'r') as f:
            imagenet_map = {v.strip(): str(pytorch_map[i]) for i, v in enumerate(f)}

        folder_to_ids, class_sublist = dict(), list()

        for objectnet_name, imagenet_names in objectnet_map.items():
            imagenet_names = imagenet_names.split('; ')
            imagenet_ids = [int(imagenet_map[imagenet_name]) for imagenet_name in imagenet_names]
            class_sublist.extend(imagenet_ids)
            folder_to_ids[folder_map[objectnet_name]] = imagenet_ids

        class_sublist = sorted(class_sublist)
        class_sublist_mask = [(i in class_sublist) for i in range(1000)]
        classname_map = {v: k for k, v in folder_map.items()}
        return class_sublist, class_sublist_mask, folder_to_ids, classname_map

    @property
    def prompt(self):
        return IMAGENET_PROMPT

    def project_logits(self, logits):
        device = logits.device
        if logits.shape[1] == self.n_class:
            return logits
        if torch.is_tensor(logits):
            logits = logits.detach()
        logits_projected = torch.zeros((logits.shape[0], self.n_class), device=device)
        for k, v in self.rev_class_idx_map.items():
            logits_projected[:, k] = torch.amax(logits[:, v], dim=1)
        return logits_projected

    def project_labels(self, labels):
        device = labels.device
        projected_labels = [self.class_idx_map[int(label)] for label in labels]
        return torch.tensor(projected_labels, dtype=torch.long, device=device)

    @staticmethod
    def loader(path):
        with open(path, "rb") as f:
            img = Image.open(f)
            width, height = img.size
            img = img.crop((2, 2, width - 2, height - 2))
            return img.convert('RGB')


json_configs = {
    # dataset path, split json, CuPL prompts, class names, prompt, number of classes
    'imagenet':
        ('imageNet', 'splits/split_soonge_imagenet.json', 'gpt3_prompts/CuPL_prompts_imagenet.json',
         IMAGENET_CLASS_NAME, IMAGENET_PROMPT, 1000),
    'caltech101':
        ('caltech101/101_ObjectCategories', 'splits/split_zhou_Caltech101.json',
         'gpt3_prompts/CuPL_prompts_caltech101.json', CALTECH101_CLASS_NAME, CALTECH101_PROMPT,
         100),
    'flowers102':
        ('flowers-102/jpg', 'splits/split_zhou_OxfordFlowers.json', 'gpt3_prompts/CuPL_prompts_flowers102.json',
         FLOWERS102_CLASS_NAME, FLOWERS102_PROMPT, 102),
    'food101':
        ('food-101/images', 'splits/split_zhou_Food101.json', 'gpt3_prompts/CuPL_prompts_food101.json',
         FOOD101_CLASS_NAME, FOOD101_PROMPT, 101),
    'dtd':
        ('dtd/images', 'splits/split_zhou_DescribableTextures.json', 'gpt3_prompts/CuPL_prompts_dtd.json',
         DESCRIBABLE_TEXTURES_CLASS_NAME, DESCRIBABLE_TEXTURES_PROMPT, 47),
    'oxfordiiitpet':
        ('oxford-iiit-pet/images', 'splits/split_zhou_OxfordPets.json', 'gpt3_prompts/CuPL_prompts_oxfordpets.json',
         OXFORD_IIIT_PETS_CLASS_NAME, OXFORD_IIIT_PETS_PROMPT, 37),
    'sun397':
        ('SUN397', 'splits/split_zhou_SUN397.json', 'gpt3_prompts/CuPL_prompts_sun397.json',
         SUN397_CLASS_NAME, SUN397_PROMPT, 397),
    'ucf101':
        ('UCF-101-midframes', 'splits/split_zhou_UCF101.json', 'gpt3_prompts/CuPL_prompts_ucf101.json',
         UCF101_CLASS_NAME, UCF101_PROMPT, 101),
    'stanfordcars':
        ('stanford_cars', 'splits/split_zhou_StanfordCars.json', 'gpt3_prompts/CuPL_prompts_stanfordcars.json',
         STANFORDCARS_CLASS_NAME, STANFORDCARS_PROMPT, 196),
    'eurosat':
        ('eurosat/2750', 'splits/split_zhou_EuroSAT.json', 'gpt3_prompts/CuPL_prompts_eurosat.json',
         EUROSAT_CLASS_NAME, EUROSAT_PROMPT, 10),
    'fgvc':
        ('fgvc-aircraft-2013b', 'splits/split_soonge_fgvc.json', 'gpt3_prompts/CuPL_prompts_fgvcaircraft.json',
         FGVC_CLASS_NAME, FGVC_PROMPT, 100),
}


class JsonDataset(VLMDataset):
    def __init__(self, root, name, split, n_shot=0, transform=None, target_transform=None):
        self.dataset_name = name
        self.dataset_path, self.json_path, self.gpt3_path, class_names, self._prompt, self.n_class = \
            json_configs[name]
        with open(DATA_DIR / self.json_path) as f:
            samples = json.load(f)[split]
            imgs = list()
            targets = list()
            for i, t, _ in samples:
                imgs.append(os.path.join(root, self.dataset_path, i))
                targets.append(t)
        with open(DATA_DIR / self.gpt3_path) as f:
            self.gpt3_prompts = json.load(f)

        super().__init__(root, imgs, targets, class_names, transform, target_transform, n_shot)

    @property
    def prompt(self):
        return self._prompt
