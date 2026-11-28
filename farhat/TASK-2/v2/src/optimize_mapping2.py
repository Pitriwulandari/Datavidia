"""Find cluster assignment matching Ghani targets using integer programming (brute force on large clusters)."""
import json
import numpy as np
from itertools import combinations

# Load data
mapping = json.load(open('models/cluster_mapping_v2_deaug_diff.json'))
labels = np.load('cache/features/clusters_v2_deaug_diff.npz')['labels']

cluster_sizes = {c: int(np.sum(labels == c)) for c in range(23)}
print("Cluster sizes:", cluster_sizes)

# Ghani v3 target distribution
targets = {
    'Hurricane': 900,
    'Earthquake': 772,
    'Volcano': 589,
    'Fire': 543,
    'Tsunami': 351,
    'Explosion': 25,
}

classes = list(targets.keys())
n = 23

# Large clusters (>100): C0-C16 (17 clusters)
# Small clusters (<=100): C17-C22 (6 clusters)

large_clusters = [c for c in range(23) if cluster_sizes[c] > 100]
small_clusters = [c for c in range(23) if cluster_sizes[c] <= 100]

print(f"Large clusters: {large_clusters}")
print(f"Small clusters: {small_clusters}")

# Strategy: assign large clusters first to get close, then fill with small clusters
# Use DP/backtracking for large clusters (17 clusters, 6 classes = 6^17 too big)
# Instead: greedy with lookahead, or use MILP via scipy

# Simple approach: try all assignments for top 6 largest clusters (C0-C5), then greedy for rest
top6 = [0, 1, 2, 3, 4, 5]  # sizes: 269, 251, 205, 191, 173, 169 = 1258

best_overall = None
best_score = float('inf')

# For top 6, try all 6^6 = 46656 combinations
from itertools import product

for assign_top6 in product(classes, repeat=6):
    partial = {top6[i]: assign_top6[i] for i in range(6)}
    partial_sum = {c: 0 for c in classes}
    for c, cls in partial.items():
        partial_sum[cls] += cluster_sizes[c]
    
    # Check if any class exceeds target by too much (>200)
    if any(partial_sum[c] > targets[c] + 200 for c in classes):
        continue
    
    # Greedy assign remaining 17 clusters
    remaining = [c for c in range(23) if c not in top6]
    remaining_sorted = sorted(remaining, key=lambda x: -cluster_sizes[x])  # largest first
    
    test_assign = partial.copy()
    test_sum = partial_sum.copy()
    
    for c in remaining_sorted:
        # Find class that gets closest to target
        best_cls = None
        best_diff = float('inf')
        for cls in classes:
            new_sum = test_sum[cls] + cluster_sizes[c]
            diff = abs(new_sum - targets[cls])
            if diff < best_diff:
                best_diff = diff
                best_cls = cls
        test_assign[c] = best_cls
        test_sum[best_cls] += cluster_sizes[c]
    
    # Score
    score = sum(abs(test_sum[c] - targets[c]) for c in classes)
    if score < best_score:
        best_score = score
        best_overall = test_assign.copy()
        if score < 100:
            print(f"New best: score={score}, dist={test_sum}")

print(f"\n=== BEST (score={best_score}) ===")
for c in sorted(best_overall.keys()):
    print(f"C{c:2d} (n={cluster_sizes[c]:4d}) -> {best_overall[c]}")

final_sum = {c: 0 for c in classes}
for c, cls in best_overall.items():
    final_sum[cls] += cluster_sizes[c]
print(f"\nFinal dist: {final_sum}")
print(f"Target:     {targets}")

# Save
with open('models/cluster_mapping_v2_deaug_diff.json', 'w') as f:
    json.dump({str(k): v for k, v in best_overall.items()}, f, indent=2)
print("\nSaved!")