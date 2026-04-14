#!/usr/bin/env python3
"""Generate the compiler backend latency comparison figure."""

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import seaborn as sns

plt.style.use(['science', 'ieee'])
pal = sns.color_palette("Set2")

# --- Data from latency_result.md ---
benchmarks = [
    'mlp3',
    'mlp2',
    'mlp1',
    'transformer\n\\_linear',
    'mlp4',
    'resnet50\n\\_chain',
    'mobilenet\n\\_struct',
]
hand = np.array([1_501, 2_571, 25_364, 29_732, 39_742, 5_544_077, 3_408_414])
act  = np.array([1_433, 2_451, 25_123, 30_302, 39_362, 5_544_303, 3_408_581])
speedup = hand / act

fig, ax = plt.subplots(figsize=(3.5, 2.0))

x = np.arange(len(benchmarks))
w = 0.35

bars_h = ax.bar(x - w/2, hand, w, label='Hand-written kernel', color=pal[2], edgecolor='black', linewidth=0.3, zorder=3)
bars_a = ax.bar(x + w/2, act,  w, label='TensorLift + ACT',    color=pal[3], edgecolor='black', linewidth=0.3, zorder=3)

ax.set_yscale('log')
ax.set_ylabel('Spike Cycles (log scale)')
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=5)
ax.set_ylim(500, 2e7)

# Annotate delta %
for i, (h, a, s) in enumerate(zip(hand, act, speedup)):
    y_pos = max(h, a) * 1.6
    ax.text(i, y_pos, f'{s:.3f}$\\times$', ha='center', va='bottom', fontsize=5, fontweight='bold')

ax.legend(fontsize=5.5, loc='upper left', frameon=True, fancybox=False,
          edgecolor='black', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linewidth=0.3)

fig.tight_layout(pad=0.3)
fig.subplots_adjust(top=0.92)
fig.savefig('fig/latency_comparison.pdf', dpi=600, bbox_inches='tight')
print('Saved fig/latency_comparison.pdf')
