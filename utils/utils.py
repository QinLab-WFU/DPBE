import torch
import numpy as np
from typing import Union
from sklearn.metrics.pairwise import euclidean_distances
from torch.nn import functional as F


def gen_test_data(B, C, K, is_multi_hot=False, normalize_embeddings=True):
    """
    Args:
        B: batch size
        C: number of classes
        K: dim of embeddings
        is_multi_hot: is multi-label dataset or not
        normalize_embeddings: normalize embeddings or not
    Returns:
        embeddings: [B, K]
        singles: [B, ],  categorical ids, None if is_multi_hot
        onehots: [B, C], onehot encoded categorical ids
    """
    embeddings = torch.randn(B, K, requires_grad=True)
    if is_multi_hot:
        singles = None
        onehots = (torch.randn(B, C) > 0.8).float()
    else:
        singles = torch.randint(low=0, high=C, size=[B])  # categorical id
        onehots = F.one_hot(singles, C).float()
    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings, singles, onehots


def compute_metrics(x):
    # 取复值的原因在于cosine的值越大说明越相似，但是需要取的是前N个值，所以取符号变为增函数s
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


def calc_neighbor(a: torch.Tensor, b: torch.Tensor):
    # print(a.dtype, b.dtype)
    return (a.matmul(b.transpose(0, 1)) > 0).float()


def euclidean_similarity(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        similarity = torch.cdist(a, b, p=2.0)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        similarity = euclidean_distances(a, b)
    else:
        raise ValueError("input value must in [torch.Tensor, numpy.ndarray], but it is %s, %s"%(type(a), type(b)))
    return similarity


def euclidean_dist_matrix(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    calculate euclidean distance as inner product
    :param tensor1: a tensor with shape (a, c)
    :param tensor2: a tensor with shape (b, c)
    :return: the euclidean distance matrix which each point is the distance between a row in tensor1 and a row in tensor2.
    """
    dim1 = tensor1.shape[0]
    dim2 = tensor2.shape[0]
    multi = torch.matmul(tensor1, tensor2.t())
    a2 = torch.sum(torch.pow(tensor1, 2), dim=1, keepdim=True).expand(dim1, dim2)
    b2 = torch.sum(torch.pow(tensor2, 2), dim=1, keepdim=True).t().expand(dim1, dim2)
    dist = torch.sqrt(a2 + b2 - 2 * multi)
    return dist


def cosine_similarity(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a = a / a.norm(dim=-1, keepdim=True) if len(torch.where(a != 0)[0]) > 0 else a
        b = b / b.norm(dim=-1, keepdim=True) if len(torch.where(b != 0)[0]) > 0 else b
        return torch.matmul(a, b.t())
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a = a / np.linalg.norm(a, axis=-1, keepdims=True) if len(np.where(a != 0)[0]) > 0 else a
        b = b / np.linalg.norm(b, axis=-1, keepdims=True) if len(np.where(b != 0)[0]) > 0 else b
        return np.matmul(a, b.T)
    else:
        raise ValueError("input value must in [torch.Tensor, numpy.ndarray], but it is %s, %s"%(type(a), type(b)))

def calc_map_k(qB, rB, query_L, retrieval_L, k=None, rank=0):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    qB = torch.sign(qB)
    rB = torch.sign(rB)
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    # print("query num:", num_query)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)      # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calcHammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        print(gnd.device, ind.device)
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.to(rank)
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calcHammingDist(B1, B2):

    if len(B1.shape) < 2:
        B1.view(1, -1)
    if len(B2.shape) < 2:
        B2.view(1, -1)
    q = B2.shape[1]
    if isinstance(B1, torch.Tensor):
        distH = 0.5 * (q - torch.matmul(B1, B2.t()))
    elif isinstance(B1, np.ndarray):
        distH = 0.5 * (q - np.matmul(B1, B2.transpose()))
    else:
        raise ValueError("B1, B2 must in [torch.Tensor, np.ndarray]")
    return distH
