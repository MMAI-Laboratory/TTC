"""Extract the CLIP features of the zero-shot and domain-generalization configs into features/{zeroshot,domain_gen}.

  python -m scripts.extract --backbone ViT-B16 --data-root ./data
  python -m scripts.extract --backbone ViT-B16 --data-root ./data \
      --datasets ImageNetA ImageNetR ImageNetSketch ImageNetV2 --aug --views 10
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torchvision import transforms
from tqdm import tqdm

from ttc.clip import clip
from ttc.data.datasets import ImageNetA, ImageNetR, ImageNetSketch, ImageNetV2, JsonDataset

DATASETS = ['caltech101', 'dtd', 'eurosat', 'fgvc', 'flowers102', 'food101', 'imagenet', 'oxfordiiitpet',
            'stanfordcars', 'sun397', 'ucf101']
DG_DATASETS = {'ImageNetA': ImageNetA, 'ImageNetR': ImageNetR, 'ImageNetSketch': ImageNetSketch,
               'ImageNetV2': ImageNetV2}
CLIP_NAMES = {"RN50": "RN50", "ViT-B16": "ViT-B/16", "ViT-B32": "ViT-B/32", "ViT-L14": "ViT-L/14"}
NORMALIZE = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))


class MultiViewTransform:
    def __init__(self, n_views, size=224, scale=(0.08, 1.0)):
        self.n_views = n_views
        self.size = size
        self.base_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
        ])
        self.random_transform = transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=transforms.InterpolationMode.BICUBIC, scale=scale),
            transforms.RandomHorizontalFlip(),
        ])
        self.to_tensor = transforms.Compose([transforms.ToTensor(), NORMALIZE])

    def __call__(self, image):
        views = torch.zeros(self.n_views, 3, self.size, self.size)
        views[0] = self.to_tensor(self.base_transform(image))
        for i in range(1, self.n_views):
            views[i] = self.to_tensor(self.random_transform(image))
        return views


@torch.no_grad()
def extract_image_features(clip_model, loader, out_dir, device):
    features, labels = [], []
    for images, target in tqdm(loader, desc=str(out_dir)):
        images = images.to(device)
        if images.dim() == 5:
            b, v = images.shape[:2]
            f = clip_model.encode_image(images.flatten(0, 1)).view(b, v, -1)
        else:
            f = clip_model.encode_image(images)
        features.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        labels.append(target)
    features, labels = torch.cat(features), torch.cat(labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({'test_f': features.contiguous()}, str(out_dir / 'test_f.safetensors'))
    save_file({'test_l': labels.contiguous()}, str(out_dir / 'test_l.safetensors'))
    print(f'saved {tuple(features.shape)} -> {out_dir / "test_f.safetensors"}')


@torch.no_grad()
def extract_text_features(clip_model, dataset, device):
    text_features = []
    classnames, prompts, template = dataset.class_name, dataset.gpt3_prompts, dataset.prompt
    for classname in tqdm(classnames, desc='text'):
        classname = classname.replace('_', ' ')
        texts = [t(classname) for t in template] + prompts[classname]
        tokens = clip.tokenize(texts, truncate=True).to(device)
        class_embeddings = torch.nn.functional.normalize(clip_model.encode_text(tokens), dim=-1)
        text_features.append(torch.nn.functional.normalize(class_embeddings.mean(dim=0), dim=-1))
    return torch.stack(text_features, dim=0).cpu()


def save_text_features(text_features, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({'text_weights_cupl': text_features.contiguous()}, str(out_dir / 'text_weights_cupl.safetensors'))
    print(f'saved {tuple(text_features.shape)} -> {out_dir / "text_weights_cupl.safetensors"}')


def parse_args():
    parser = argparse.ArgumentParser(description='Extract CLIP features for TTC.')
    parser.add_argument('--backbone', default='ViT-B16', choices=list(CLIP_NAMES))
    parser.add_argument('--data-root', required=True, help='root directory of the datasets')
    parser.add_argument('--feature-root', default='features')
    parser.add_argument('--datasets', nargs='+', default=DATASETS + list(DG_DATASETS))
    parser.add_argument('--aug', action='store_true', help='also extract multi-view features (domain generalization)')
    parser.add_argument('--views', type=int, default=10, help='number of views with --aug')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, preprocess = clip.load(CLIP_NAMES[args.backbone], device=device)
    clip_model.eval()

    root = Path(args.feature_root)
    imagenet_text = None
    for name in args.datasets:
        if name not in DG_DATASETS:
            out_dir = root / 'zeroshot' / args.backbone / name
            dataset = JsonDataset(args.data_root, name, split='val' if name == 'imagenet' else 'test', transform=preprocess)
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
            extract_image_features(clip_model, loader, out_dir, device)
            save_text_features(extract_text_features(clip_model, dataset, device), out_dir)
            continue

        out_dir = root / 'domain_gen' / args.backbone / name
        if imagenet_text is None:
            imagenet_text = extract_text_features(clip_model, JsonDataset(args.data_root, 'imagenet', split='val'), device)
        save_text_features(imagenet_text, out_dir)
        dataset = DG_DATASETS[name](args.data_root, transform=preprocess)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
        extract_image_features(clip_model, loader, out_dir / '0aug', device)

        if args.aug:
            transform = MultiViewTransform(args.views)
            dataset = DG_DATASETS[name](args.data_root, transform=transform)
            batch_size = max(1, args.batch_size // transform.n_views)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=args.workers)
            extract_image_features(clip_model, loader, out_dir / f'{args.views}aug', device)


if __name__ == '__main__':
    main()
