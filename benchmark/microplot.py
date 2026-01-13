from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plots_dir = Path("plots")
pattern = re.compile(r"^(?P<prefix>.+?)_batch(?P<batch>[^_]+)_expansion(?P<expansion>[^.]+)\.csv$")

run_dfs = []
for path in plots_dir.glob("*.csv"):
    m = pattern.match(path.name)
    if m:
        one_run_df = pd.read_csv(path)
        one_run_df = one_run_df.assign(
            batch=int(m.group("batch")),
            expansion=float(m.group("expansion")),
        )
        run_dfs.append(one_run_df)

all_runs_df = pd.concat(run_dfs, ignore_index=True)
id_vars=('dim', 'batch', 'expansion',)
all_runs_df = all_runs_df.melt(
    id_vars=id_vars,
    value_vars=[c for c in all_runs_df.columns if c not in id_vars],
    var_name="method",
    value_name="runtime (ms)",
)
all_runs_df["runtime / mn^2 (ms)"] = all_runs_df["runtime (ms)"] / (all_runs_df["expansion"] * all_runs_df["dim"] ** 3)


g = sns.FacetGrid(all_runs_df.query("dim > 512"), col="dim", col_wrap=4, sharex=True, sharey=False, height=4)
g.map_dataframe(
    sns.lineplot,
    x="expansion",
    y="runtime (ms)",
    hue="method",
    style="method",
    markers=True,
)
g.add_legend(title="method")
for ax in g.axes.flatten():
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

g.tight_layout()
g.savefig(plots_dir / "microplot_by_dim.png")
