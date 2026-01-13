from functools import partial
from itertools import chain
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from dion.polarExp import final_appF, PolarExpress


def polar_accuracy_metrics(A: torch.Tensor, estimated_polar_A: torch.Tensor) -> dict[str, float]:
    metrics = dict(
        orth_error=float('inf'),
        residual_error=float('inf'),
        psd_error=float('inf'),
        dual_obj=float('inf'),
        bound_violation=float('inf'),
    )
    if not estimated_polar_A.isfinite().all():
        return metrics

    estimated_polar_A = estimated_polar_A.to(A.dtype)
    H = estimated_polar_A.mT @ A
    H = (H + H.mT) / 2
    Heigs = torch.linalg.eigvalsh(H)
    nuc = torch.linalg.matrix_norm(A, ord='nuc')
    estimate_spectral_norm = torch.linalg.matrix_norm(estimated_polar_A, ord=2)
    I = torch.eye(estimated_polar_A.shape[0], device=estimated_polar_A.device, dtype=estimated_polar_A.dtype)

    metrics["orth_error"] = ((estimated_polar_A @ estimated_polar_A.mT - I).norm() / I.norm()).item()
    metrics["residual_error"] = ((estimated_polar_A @ H - A).norm() / A.norm()).item()
    metrics["psd_error"] = ((Heigs[Heigs < 0]).norm() / (Heigs[Heigs > 0]).norm()).item()
    metrics["dual_obj"] = ((nuc - torch.inner(A.flatten(), estimated_polar_A.flatten()))/nuc).item()
    metrics["bound_violation"] = max((estimate_spectral_norm - 1).item(), 0)
    return metrics


def facet_plot(df, outpath):
    # Reshape data for seaborn: one row per (sample, metric) with ref and ours columns
    plot_df = df.stack(level='metric', future_stack=True).reset_index(level='metric')

    g = sns.FacetGrid(plot_df, col='metric', col_wrap=3, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x='reference', y='ours')

    # Add y=x reference line to each facet
    for ax in g.axes.flat:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

    g.set_axis_labels('reference', 'ours')
    plt.tight_layout()
    plt.savefig(outpath)


def spectrum2matrix(spectrum, aspect_ratio):
    n = len(spectrum)
    m = int(n * aspect_ratio)
    U, _, Vh = torch.linalg.svd(torch.randn(m, n, device=spectrum.device, dtype=spectrum.dtype), full_matrices=False)
    return U @ torch.diag(spectrum) @ Vh


def run_comparison(our_method, reference_method, steps, matrix_iter, iterlen=None):
    results = []
    for matrix in tqdm(matrix_iter, total=iterlen):
        ref = reference_method(matrix, steps=steps)
        reference_metrics = polar_accuracy_metrics(matrix, ref)
        ours = our_method(matrix, steps=steps)
        ours_metrics = polar_accuracy_metrics(matrix, ours)
        results.append(list(reference_metrics.values()) + list(ours_metrics.values()))
    cols = pd.MultiIndex.from_tuples(
        [(k, "reference") for k in reference_metrics.keys()] + [(k, "ours") for k in ours_metrics.keys()],
        names=["metric", "method"],
    )
    return pd.DataFrame(results, columns=cols)


if __name__ == "__main__":
    STEPS = 15
    num_limit = 100
    test_set = "synthetic"

    if test_set == "bad_G":
        # Load Noah's set of numerically challenging matrices
        bad_g_dir = "bad_G"
        matrix_files = sorted(
            [os.path.join(bad_g_dir, f) for f in os.listdir(bad_g_dir) if f.endswith(".pt")],
            key=lambda x: os.path.getmtime(x)
        )
        matrix_iter = map(torch.load, matrix_files[:num_limit])
        num = min(len(matrix_files), num_limit)
    elif test_set == "synthetic":
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        n = 768
        aspect_ratio = 4
        spectra = [
            torch.logspace(0, -3, steps=n),
            torch.logspace(0, -5, steps=n),
            torch.logspace(0, -7, steps=n),
            torch.logspace(0, -9, steps=n),
            torch.cat((
                torch.logspace(0, -2, steps=n//2),
                torch.zeros(n - n//2),
            )),
            torch.cat((
                torch.logspace(0, -1, steps=10),
                torch.zeros(n - 10),
            )),
            torch.cat((
                torch.logspace(0, -0.5, steps=2),
                torch.zeros(n - 2),
            )),
            torch.cat((
                torch.logspace(0, -0.5, steps=2),
                torch.logspace(-4, -7, steps=n - 2),
            ))
        ] + [
            torch.distributions.Gamma(concentration=3, rate=0.5).sample((n,))
            for _ in range(5)
        ]
        spectra = [s.to(DEVICE) for s in spectra]
        matrix_iter = map(lambda s: spectrum2matrix(s, aspect_ratio), spectra)
        num = len(spectra)

    our_method = partial(final_appF, restart_interval=3, shift_eps=1e-3)
    df = run_comparison(our_method, PolarExpress, STEPS, matrix_iter, iterlen=num)
    df.to_pickle(f"plots/appF_{test_set}_{STEPS}_steps.pkl")
    print(len(df))
    print(df.head(5))
    blowups = df.loc[(~np.isfinite(df)).any(axis=1)]
    if len(blowups) == 0:
        print("No blowups detected.")
    else:
        print("Blowups:")
        print(blowups)

    facet_plot(df, f'plots/appF_{test_set}_{STEPS}_steps.png')
