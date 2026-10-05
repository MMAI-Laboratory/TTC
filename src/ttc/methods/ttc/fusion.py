import torch
import torch.nn.functional as F
from tqdm import tqdm


def entropy_fusion(features: torch.Tensor, text: torch.Tensor, alpha: float, eps: float = 1e-8) -> torch.Tensor:
    logits = torch.matmul(features, text.T) * 100
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    scores = -entropy

    if alpha:
        weights = F.softmax(scores[1:], dim=0).unsqueeze(1)
        weighted_features = features[0] * (1 - alpha) + torch.sum(features[1:] * weights, dim=0) * alpha
    else:
        weights = F.softmax(scores, dim=0).unsqueeze(1)
        weighted_features = torch.sum(weights * features, dim=0)

    return weighted_features


@torch.no_grad()
def feature_fusion(features, text, alpha):
    features = features.cpu()
    text = text.cpu()
    features = torch.nn.functional.normalize(features, dim=-1)

    outs = []
    for f in tqdm(features):
        outs.append(entropy_fusion(f, text, alpha))
    outs = torch.stack(outs)
    return outs
