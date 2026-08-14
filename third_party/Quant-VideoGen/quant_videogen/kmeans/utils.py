import torch
from ..timer import time_logging_decorator

@time_logging_decorator("Level 4- density calculation")
def density_calculation(dynamic_map, q_cluster_sizes, k_cluster_sizes):
    """
    Calculate the density of the dynamic map. Currently only batch size = 1 and head size = 1 are supported.

    Input:
        dynamic_map: [cfg, num_heads, qc_num, kc_num]
        q_cluster_sizes: [cfg, num_heads, qc_num]
        k_cluster_sizes: [cfg, num_heads, kc_num]
    """
    cfg, num_heads, qc_num, kc_num = dynamic_map.shape

    # Calculate the block size of each block
    clustered_block_size = (
        q_cluster_sizes[:, :, :, None] * k_cluster_sizes[:, :, None, :]
    )
    masked_block_size = clustered_block_size * dynamic_map

    # Calculate the density of each block
    density = torch.sum(masked_block_size, dim=(2, 3)) / torch.sum(
        clustered_block_size, dim=(2, 3)
    )
    return density


# --- Functions from analyze/kmeans_rapidai.py ---


def pairwise_distance(x, y):
    """
    Computes pairwise squared Euclidean distance between two sets of points.
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = torch.clamp(
        x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1)), min=0.0
    )
    return dist


def kmeans_predict(centroids, input_tensor):  # Removed unused params argument
    """
    Predict the labels for the input tensor using the centroids.
    """
    input_tensor = input_tensor.to(torch.float32)
    dist = pairwise_distance(input_tensor, centroids)
    labels = torch.argmin(dist, dim=1)
    return labels


@time_logging_decorator("Level 4 - permute tensor by labels")
def permute_tensor_by_labels(tensor, labels, dim):
    labels = labels.to(tensor.device)
    sorted_indices = torch.argsort(labels, dim=-1)
    gather_indices = sorted_indices
    for i in range(dim + 1, tensor.dim()):
        gather_indices = gather_indices.unsqueeze(-1)
    expand_shape = list(tensor.shape)
    gather_indices = gather_indices.expand(expand_shape)
    permuted_tensor = torch.gather(tensor, dim, gather_indices)
    return permuted_tensor, sorted_indices


@time_logging_decorator("Level 4 - inverse permutation")
def apply_inverse_permutation(permuted_tensor, sorted_indices, dim):
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    gather_indices = inverse_indices
    for i in range(dim + 1, permuted_tensor.dim()):
        gather_indices = gather_indices.unsqueeze(-1)
    gather_indices = gather_indices.expand(permuted_tensor.shape)
    original_tensor = torch.gather(permuted_tensor, dim, gather_indices)
    return original_tensor


@time_logging_decorator("Level 4 - weighted softmax")
def weighted_softmax(scores, weights):
    input_dtype = scores.dtype
    scores = scores.float()
    weights = weights.float()
    max_score = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_score)
    weighted_exp = weights * exp_scores
    softmax_out = weighted_exp / torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(
        min=1e-12
    )
    return softmax_out.to(input_dtype)


@time_logging_decorator("Level 4 - identify dynamic map")
def identify_dynamic_map(
    query_centroids,
    key_centroids,
    q_cluster_sizes,
    k_cluster_sizes,
    p,
    min_kc_ratio=0,
):
    B, H, qc_num, D = query_centroids.shape
    kc_num = key_centroids.shape[2]
    device = query_centroids.device

    attn_scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (
        D**0.5
    )
    k_weights = k_cluster_sizes.unsqueeze(-2).float()

    weighted_attn_probs = weighted_softmax(attn_scores, k_weights)
    sorted_probs, sorted_indices = torch.sort(
        weighted_attn_probs, dim=-1, descending=True
    )

    cumsum_probs = torch.cumsum(
        sorted_probs.float(), dim=-1
    )  # HXI: the float() here is very important!
    remove_indices = cumsum_probs > p
    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
    remove_indices[..., 0] = False

    if min_kc_ratio > 0:
        preserve_length = int(min_kc_ratio * kc_num)
        remove_indices[..., :preserve_length] = False

    sorted_clusters_to_keep = ~remove_indices

    dynamic_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
    dynamic_map.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    return dynamic_map
