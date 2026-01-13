from itertools import chain
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from dion.polarExp import *


def metrics(matrix, estimate):
    if not estimate.isfinite().all():
        return dict(
            orth_error=float('inf'),
            residual_error=float('inf'),
            psd_error=float('inf'),
            dual_obj=float('inf'),
            bound_violation=float('inf'),
        )

    estimate = estimate.to(matrix.dtype)
    H = estimate.mT @ matrix
    H = (H + H.mT) / 2
    I = torch.eye(estimate.shape[0], device=estimate.device, dtype=estimate.dtype)

    # print(f"relative error: {((estimate - true_polar).norm() / (true_polar).norm()).item():.6f}")
    orthogonality_error = ((estimate @ estimate.mT - I).norm() / I.norm()).item()
    residual_error = ((estimate @ H - matrix).norm() / matrix.norm()).item()
    Heigs = torch.linalg.eigvalsh(H)
    psd_error = ((Heigs[Heigs < 0]).norm() / (Heigs[Heigs > 0]).norm()).item()
    nuc = torch.linalg.matrix_norm(matrix, ord='nuc')
    dual_obj = ((nuc - torch.inner(matrix.flatten(), estimate.flatten()))/nuc).item()
    bound_violation = max((torch.linalg.matrix_norm(estimate, ord=2) - 1).item(), 0)
    return dict(
        orth_error=orthogonality_error,
        residual_error=residual_error,
        psd_error=psd_error,
        dual_obj=dual_obj,
        bound_violation=bound_violation,
    )


def facet_plot(df):
    # Reshape data for seaborn: one row per (sample, metric) with ref and ours columns
    plot_df = df.stack(level='metric', future_stack=True).reset_index(level='metric')

    g = sns.FacetGrid(plot_df, col='metric', col_wrap=3, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x='ref', y='ours')

    # Add y=x reference line to each facet
    for ax in g.axes.flat:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

    g.set_axis_labels('ref', 'ours')
    plt.tight_layout()
    plt.savefig('plots/facet_comparison.png')
    plt.show()



STEPS = 15
num_limit = 100

bad_g_dir = "bad_G"
matrix_files = sorted(
    [os.path.join(bad_g_dir, f) for f in os.listdir(bad_g_dir) if f.endswith(".pt")],
    key=lambda x: os.path.getmtime(x)
)
results = []
for matrix_path in tqdm(matrix_files[:num_limit]):
    matrix = torch.load(matrix_path)
    ref = PolarExpress(matrix, steps=STEPS)
    reference_metrics = metrics(matrix, ref)
    ours = final_appF(matrix, steps=STEPS, first_restart=0, restart_interval=5, shift_eps=1e-3)
    ours_metrics = metrics(matrix, ours)
    results.append(list(chain.from_iterable(zip(reference_metrics.values(), ours_metrics.values()))))
cols = pd.MultiIndex.from_tuples(
    [(k, suff) for k in reference_metrics.keys() for suff in ("ref", "ours")],
    names=["metric", "method"],
)
df = pd.DataFrame(results, columns=cols)
df.to_pickle("appF_full_results.pkl")
print(len(df))
print(df.head(5))
blowups = df.loc[(~np.isfinite(df)).any(axis=1)]
if len(blowups) == 0:
    print("No blowups detected.")
else:
    print("Blowups:")
    print(blowups)

facet_plot(df)

# USE TORCH_LOGS="recompiles" 