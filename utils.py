import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

def reduce_superpixels_via_cosine_merge(label_map, kc_flat, kx, ky, target_P):
    """
    Merge superpixels iteratively based on cosine similarity until only target_P remain.

    Args:
        label_map (np.ndarray): HxW label map (values 0 to N-1)
        kc_flat (np.ndarray): NxC array of color vectors (e.g., LAB or RGB) per superpixel
        kx (np.ndarray): spatial X mean per superpixel (for optional updates)
        ky (np.ndarray): spatial Y mean per superpixel
        target_P (int): target number of superpixels to reduce to

    Returns:
        new_label_map (np.ndarray): updated HxW label map
        new_kc_flat, new_kx, new_ky (np.ndarray): reduced superpixel features
    """
    H, W = label_map.shape
    current_labels = np.unique(label_map)
    num_superpixels = len(current_labels)
    
    # Initial label -> index map
    label_to_index = {label: idx for idx, label in enumerate(current_labels)}
    
    # Reverse map: index -> label
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    # Reduce kc_flat, kx, ky to current label set
    kc = kc_flat[current_labels]
    kx = kx[current_labels]
    ky = ky[current_labels]

    # Build similarity heap (max-heap using cosine similarity)
    sim_heap = []
    cos_sim = cosine_similarity(kc)
    np.fill_diagonal(cos_sim, -np.inf)  # prevent self-similarity

    for i in range(num_superpixels):
        for j in range(i+1, num_superpixels):
            heapq.heappush(sim_heap, (-cos_sim[i, j], i, j))  # negate for max heap

    # Label map for remapping during merges
    label_map_flat = label_map.flatten()
    label_assignment = {i: i for i in range(num_superpixels)}  # merged label index mapping

    merged_count = 0
    while num_superpixels - merged_count > target_P and sim_heap:
        _, i, j = heapq.heappop(sim_heap)
        
        root_i = find_root(i, label_assignment)
        root_j = find_root(j, label_assignment)

        if root_i == root_j:
            continue  # already merged

        # Merge j into i (arbitrarily)
        new_kc = (kc[root_i] + kc[root_j]) / 2
        new_kx = (kx[root_i] + kx[root_j]) / 2
        new_ky = (ky[root_i] + ky[root_j]) / 2

        kc[root_i] = new_kc
        kx[root_i] = new_kx
        ky[root_i] = new_ky

        label_assignment[root_j] = root_i
        merged_count += 1

        # Update similarities with merged cluster
        for k in range(num_superpixels):
            if k == root_i or find_root(k, label_assignment) != k:
                continue
            sim = cosine_similarity([kc[root_i]], [kc[k]])[0, 0]
            heapq.heappush(sim_heap, (-sim, root_i, k))

    # Final remapping
    root_map = {}
    new_label_id = 0
    for idx in range(num_superpixels):
        root = find_root(idx, label_assignment)
        if root not in root_map:
            root_map[root] = new_label_id
            new_label_id += 1

    # Apply new labels
    new_labels = np.array([root_map[find_root(label_to_index[l], label_assignment)] for l in label_map_flat])

    new_label_map = new_labels.reshape(H, W)

    # Compute new superpixel features
    new_kc_flat = np.zeros((target_P, kc.shape[1]))
    new_kx_flat = np.zeros((target_P,))
    new_ky_flat = np.zeros((target_P,))
    counts = np.zeros((target_P,))

    for y in range(H):
        for x in range(W):
            new_label = new_label_map[y, x]
            old_label = label_map[y, x]
            old_idx = label_to_index[old_label]
            new_kc_flat[new_label] += kc[find_root(old_idx, label_assignment)]
            new_kx_flat[new_label] += x
            new_ky_flat[new_label] += y
            counts[new_label] += 1

    counts[counts == 0] = 1
    new_kc_flat /= counts[:, None]
    new_kx_flat /= counts
    new_ky_flat /= counts

    return new_label_map, new_kc_flat, new_kx_flat, new_ky_flat


def find_root(i, parent_map):
    # Union-find helper
    path = []
    while parent_map[i] != i:
        path.append(i)
        i = parent_map[i]
    for p in path:
        parent_map[p] = i
    return i


