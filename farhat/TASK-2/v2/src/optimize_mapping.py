"""Optimize cluster-to-class mapping to match Ghani v3 target distribution."""
import json
import numpy as np
from collections import Counter

# Load data
mapping = json.load(open('models/cluster_mapping_v2_deaug_diff.json'))
labels = np.load('cache/features/clusters_v2_deaug_diff.npz')['labels']

# Cluster sizes
cluster_sizes = {c: int(np.sum(labels == c)) for c in range(23)}
print("Cluster sizes:", cluster_sizes)

# Ghani v3 target distribution (from 0.37895 submission)
targets = {
    'Hurricane': 900,
    'Earthquake': 772,
    'Volcano': 589,
    'Fire': 543,
    'Tsunami': 351,
    'Explosion': 25,
}
total_target = sum(targets.values())
print(f"Target total: {total_target}, Actual total: {len(labels)}")

# Current assignment
current = {c: mapping[str(c)] for c in range(23)}
current_dist = Counter(current.values())
print("Current dist:", dict(current_dist))

# We need to reassign some clusters
# Strategy: greedy - move clusters from over-target classes to under-target classes
classes = list(targets.keys())

def get_dist(assign):
    return Counter(assign.values())

def score(assign):
    dist = get_dist(assign)
    return sum(abs(dist.get(c, 0) - targets[c]) for c in classes)

best_assign = current.copy()
best_score = score(best_assign)
print(f"Initial score: {best_score}")

# Try moving each cluster to each class
improved = True
while improved:
    improved = False
    for c in range(23):
        for cls in classes:
            if best_assign[c] == cls:
                continue
            test_assign = best_assign.copy()
            test_assign[c] = cls
            s = score(test_assign)
            if s < best_score:
                best_score = s
                best_assign = test_assign
                improved = True
                print(f"Improved: C{c} -> {cls}, score={best_score}, dist={dict(get_dist(best_assign))}")

print("\n=== BEST MAPPING ===")
for c in sorted(best_assign.keys()):
    print(f"C{c:2d} (n={cluster_sizes[c]:4d}) -> {best_assign[c]}")

final_dist = get_dist(best_assign)
print(f"\nFinal dist: {dict(final_dist)}")
print(f"Target:     {targets}")
print(f"Score: {best_score}")

# Save
with open('models/cluster_mapping_v2_deaug_diff.json', 'w') as f:
    json.dump({str(k): v for k, v in best_assign.items()}, f, indent=2)
print("\nSaved to models/cluster_mapping_v2_deaug_diff.json")