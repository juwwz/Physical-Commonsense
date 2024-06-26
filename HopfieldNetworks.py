import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity


def retrieve_from_hopfield_network(feature_set, z, beta):
    sim_score = beta * pairwise_cosine_similarity(z, feature_set)
    sep_score = F.softmax(sim_score, dim=1)
    sep_score = F.normalize(sep_score, p=1)
    out = sep_score @ feature_set
    return out
