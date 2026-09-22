"""One row per training job: what it ran, on what code, and how it came out.

The slurm logs alone do not pin a run down -- they print only the hyperparameters
the YAML or CLI set explicitly, never the defaults, and the flags that live on
cli_args (Polar Express, Gram Newton-Schulz, no_triton, fs_size) appear only
inside the "train args" line. Nor do they record the commit. So this joins each
log to its wandb run directory, which carries the resolved config and the git SHA.

    python experiments/make_manifest.py > manifest.csv
    python experiments/make_manifest.py --logs ./slurm_logs --wandb ./wandb

Facts that no record captures -- the commit for runs that skipped wandb, say --
go in OVERRIDES below.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

END = re.compile(r"Peak memory consumption|ChildFailedError|CANCELLED AT|DUE TO TIME LIMIT|oom_kill|slurmstepd: error|OutOfMemoryError")
FIELDS = ["job", "name", "date", "host", "status", "config", "optimizer", "lr", "mu",
          "ortho_fraction", "steps", "tokens", "seq_len", "batch", "grad_clip", "mlp_type",
          "attn_gate", "norm_gain", "polar_express", "gns", "seed", "warmup_ratio",
          "final_val_loss", "last_step", "peak_mem_gib", "wandb_run", "git_commit",
          "commit_source", "note", "args"]

# Facts about a run that no log or wandb file records, asserted by hand. Any
# manifest field can be set; these win over what was parsed, and commit_source
# marks a commit as asserted rather than recorded, so the two are never confused.
#
# The --no_wandb runs have no wandb directory and so no recorded commit, and for
# these four the timestamps would give the wrong one: they started ~15:34 on
# 2026-09-10 against an uncommitted working tree, and the fixes they depended on
# (the LambdaLR fast-forward in train.py, the load_state_dict fix in
# dion/megabatch_base.py) were only committed at 16:11 as 9360a0a. So the nearest
# committed state is 9360a0a, which is after they started, not before.
OVERRIDES = {
    "17329691": {"git_commit": "9360a0a", "commit_source": "asserted",
                 "note": "no wandb (--no_wandb resume); ran uncommitted tree, nearest commit 9360a0a; died on the DCP KeyError"},
    "17329692": {"git_commit": "9360a0a", "commit_source": "asserted",
                 "note": "no wandb (--no_wandb resume); ran uncommitted tree, nearest commit 9360a0a; died on the DCP KeyError"},
    "17330992": {"git_commit": "9360a0a", "commit_source": "asserted",
                 "note": "no wandb (--no_wandb resume); ran uncommitted tree, nearest commit 9360a0a; resumed 17309781 at step 6000"},
    "17330993": {"git_commit": "9360a0a", "commit_source": "asserted",
                 "note": "no wandb (--no_wandb resume); ran uncommitted tree, nearest commit 9360a0a; resumed 17309782 at step 6000"},
}


def wandb_index(wandb_dir):
    """job id -> (run id, git commit, resolved config), from each run's files/."""
    index = {}
    for meta_path in glob.glob(os.path.join(wandb_dir, "*", "files", "wandb-metadata.json")):
        run_dir = os.path.dirname(os.path.dirname(meta_path))
        try:
            meta = json.load(open(meta_path))
        except (OSError, ValueError):
            continue
        args = meta.get("args", [])
        job = None
        for flag in ("--wandb_job_name", "--checkpoint_dir"):
            if flag in args:
                m = re.search(r"-(\d+)$", args[args.index(flag) + 1].rstrip("/"))
                if m:
                    job = m.group(1)
                    break
        if not job:
            continue
        cfg = {}
        cfg_path = os.path.join(run_dir, "files", "config.yaml")
        if os.path.exists(cfg_path):
            for line in open(cfg_path, errors="ignore"):
                m = re.match(r"^(\w+):\s*$", line) or re.match(r"^\s+value: (.*)$", line)
                if m and m.re.pattern.startswith("^(\\w+)"):
                    key = m.group(1)
                elif m:
                    cfg[key] = m.group(1).strip()
        index[job] = (os.path.basename(run_dir).split("-")[-1], (meta.get("git") or {}).get("commit", ""), cfg)
    return index


def row(path, index):
    text = open(path, errors="ignore").read().replace("\r", "\n")
    job = re.search(r"-(\d+)\.out$", path).group(1)
    args = (re.search(r"train args:(.*)", text) or [None, ""])[1].strip()
    run_id, commit, cfg = index.get(job, ("", "", {}))

    def flag(name):
        return "yes" if f"--{name}" in args else ""

    def opt(name, default=""):
        m = re.search(rf"--{name} (\S+)", args)
        if m:
            return m.group(1)
        m = re.search(rf"Setting hyperparameter {name}=(\S+)", text)
        return m.group(1) if m else cfg.get(name, default)

    steps = re.findall(r"--num_iterations (\d+)", args)          # a later flag wins
    steps = steps[-1] if steps else opt("num_iterations")
    vals = re.findall(r"step:(\d+)/(\d+) val_loss:([\d.]+)", text)
    end = END.search(text)
    seq, batch = opt("sequence_length", "1024"), opt("batch_size", "512")
    peak = re.search(r"Peak memory consumption: (\d+) MiB", text)
    record = {
        "job": job,
        "name": os.path.basename(path).rsplit("-", 1)[0],
        "date": __import__("datetime").date.fromtimestamp(os.path.getmtime(path)).isoformat(),
        "host": text.split("\n", 1)[0].split(" on ")[-1].strip(" ="),
        "status": ("ok" if vals and vals[-1][0] == vals[-1][1]
                   else (end.group(0) if end else "running/incomplete")),
        "config": (re.search(r"--config (\S+)", args) or [None, ""])[1],
        "optimizer": opt("optimizer"), "lr": opt("lr"), "mu": opt("mu"),
        "ortho_fraction": opt("ortho_fraction"),
        "steps": steps,
        "tokens": int(steps) * int(seq) * int(batch) if steps and seq and batch else "",
        "seq_len": seq, "batch": batch,
        "grad_clip": opt("grad_clip", "0"), "mlp_type": opt("mlp_type", "relu2"),
        "attn_gate": flag("attn_gate"), "norm_gain": flag("norm_gain"),
        "polar_express": flag("use_polar_express"), "gns": flag("use_gram_newton_schulz"),
        "seed": opt("seed", "0"), "warmup_ratio": opt("warmup_ratio"),
        "final_val_loss": vals[-1][2] if vals and vals[-1][0] == vals[-1][1] else "",
        "last_step": vals[-1][0] if vals else "",
        "peak_mem_gib": f"{int(peak.group(1)) / 1024:.1f}" if peak else "",
        "wandb_run": run_id, "git_commit": commit[:10],
        "commit_source": "wandb" if commit else "", "note": "",
        "args": args,
    }
    for key, value in OVERRIDES.get(job, {}).items():
        if key not in FIELDS:
            raise KeyError(f"OVERRIDES[{job!r}] sets unknown field {key!r}")
        record[key] = value
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", default=os.environ.get("DION_SLURM_LOGS", "/scratch/nia4240/dion/slurm_logs"))
    p.add_argument("--wandb", default="/scratch/nia4240/dion/wandb")
    p.add_argument("--out", default="-")
    a = p.parse_args()

    index = wandb_index(a.wandb) if os.path.isdir(a.wandb) else {}
    rows = [row(f, index) for f in sorted(glob.glob(os.path.join(a.logs, "*.out")))]
    rows.sort(key=lambda r: int(r["job"]))
    out = sys.stdout if a.out == "-" else open(a.out, "w", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
    if a.out != "-":
        out.close()
        print(f"wrote {a.out}: {len(rows)} runs, {sum(bool(r['git_commit']) for r in rows)} with a git commit",
              file=sys.stderr)


if __name__ == "__main__":
    main()
