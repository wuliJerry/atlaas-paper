#!/usr/bin/env python3
"""Generate the MLIR lifting reduction figure for ATLAAS evaluation."""

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import seaborn as sns

plt.style.use(['science', 'ieee'])
pal = sns.color_palette("Set2")

# --- Data from lifting_results.md (total lines per module) ---
modules = [
    'PE',
    'Execute\nCtrl',
    'Load\nCtrl',
    'Store\nCtrl',
    'Tensor\nGemm',
    'Tensor\nAlu',
    'Store',
    'GenVME\nCmd',
]
before = np.array([686, 997_760, 50_004, 63_313, 29_745, 4_832, 62_807, 3_160])
after  = np.array([49,  759_360, 33_244, 43_269,  1_183, 1_465, 54_294, 2_205])
reduction = (1 - after / before) * 100

fig, ax = plt.subplots(figsize=(3.5, 2.0))

x = np.arange(len(modules))
w = 0.35

bars_b = ax.bar(x - w/2, before, w, label='Before lifting', color=pal[0], edgecolor='black', linewidth=0.3, zorder=3)
bars_a = ax.bar(x + w/2, after,  w, label='After lifting',  color=pal[1], edgecolor='black', linewidth=0.3, zorder=3)

ax.set_yscale('log')
ax.set_ylabel('MLIR Lines (log scale)')
ax.set_xticks(x)
ax.set_xticklabels(modules, fontsize=5.5)
ax.set_ylim(10, 3e6)

# Annotate reduction %
for i, (b, a, r) in enumerate(zip(before, after, reduction)):
    y_pos = max(b, a) * 1.5
    ax.text(i, y_pos, f'{r:.1f}\\%', ha='center', va='bottom', fontsize=5, fontweight='bold')

# Accelerator labels
ax.text(1.5, 1.08, 'Gemmini', transform=ax.get_xaxis_transform(),
        ha='center', va='bottom', fontsize=6, fontstyle='italic', color='#333333')
ax.text(5.5, 1.08, 'VTA', transform=ax.get_xaxis_transform(),
        ha='center', va='bottom', fontsize=6, fontstyle='italic', color='#333333')

# Separator
ax.axvline(x=3.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

ax.legend(fontsize=5.5, loc='upper right', frameon=True, fancybox=False,
          edgecolor='black', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linewidth=0.3)

fig.tight_layout(pad=0.3)
fig.subplots_adjust(top=0.88)
fig.savefig('fig/lifting_results.pdf', dpi=600, bbox_inches='tight')
print('Saved fig/lifting_results.pdf')
