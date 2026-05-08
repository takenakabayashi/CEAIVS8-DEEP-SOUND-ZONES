import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

h5_path = './Sound zones - MATLAB/simulatedData.h5'

alphas = []
total_rtfs = 0
total_receivers = 0

with h5py.File(h5_path, 'r') as f:
    for room_key in f.keys():
        alphas.append(float(f[room_key]['alpha'][()]))
        n_rcv = f[room_key]['receiver_pos'].shape[0]
        n_src = f[room_key]['source_pos'].shape[0]
        total_receivers += n_rcv
        total_rtfs += n_rcv * n_src

alphas = np.array(alphas)
counts = Counter(alphas)

print(f"Total rooms:     {len(alphas):,}")
print(f"Total receivers: {total_receivers:,}")
print(f"Total RTFs:      {total_rtfs:,}  (receivers × 8 sources)")
print(f"Avg receivers per room: {total_receivers/len(alphas):.0f}")
print()
print(f"{'Alpha':<10} {'Count':<10} {'Percentage':<10}")
print("-" * 30)
for alpha in sorted(counts.keys()):
    count = counts[alpha]
    pct = 100 * count / len(alphas)
    print(f"{alpha:<10.2f} {count:<10} {pct:.1f}%")

# bar chart
fig, ax = plt.subplots(figsize=(7, 4))
sorted_alphas = sorted(counts.keys())
sorted_counts = [counts[a] for a in sorted_alphas]
bars = ax.bar([str(a) for a in sorted_alphas], sorted_counts, color='steelblue', edgecolor='black')
ax.bar_label(bars, labels=[f"{c}\n({100*c/len(alphas):.1f}%)" for c in sorted_counts], padding=3)
ax.set_xlabel('Absorption Coefficient (α)')
ax.set_ylabel('Number of Rooms')
ax.set_title('Distribution of Absorption Coefficients Across Rooms')
ax.set_ylim(0, max(sorted_counts) * 1.15)
plt.tight_layout()
plt.show()