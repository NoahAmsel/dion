#!/usr/bin/env bash
# NorMuon vs Dion3 (f=1/8) across model size and token budget, matched to the
# external 1B/20B runs so all three sizes sit on one axis.
#
# Four parents, each 38147 steps = 20.0B tokens, lr 0.01, one GPU each:
#
#   normuon 225m   dion3 225m      configs/normuon_225m.yaml       ~14.5h
#   normuon 418m   dion3 418m      configs/normuon_418m_20b.yaml   ~33h
#
# Not configs/normuon_418m.yaml: that one is tied, 5% warmup, 20% warmdown, and
# stages 1-10 of sweep_normuon_418m_lr.sh depend on it staying that way.
#
# Each parent keeps permanent checkpoints where a shorter run's cooldown would
# start, and a branch job resumes from each to produce a properly annealed
# endpoint at that budget. Reading a long run's curve at 5B tokens is not a
# 5B-token run; these are.
#
#   budget   branch step   total steps   cooldown
#   1.25B      1788           2384          596
#   2.5B       3576           4768         1192
#   5B         7153           9537         2384
#   10B       14305          19073         4768
#   20B          --          38147         9537   (the parent's own cooldown)
#
# Two more 225m runs at lr 0.005 probe the one axis we have never measured at a
# long budget: whether the optimum sits below 0.01. Everything we have (418M at
# 10B, external 1B at 20B) only brackets 0.01-0.02.
#
#   ./launch_size_budget.sh                 # 6 parents/probes + 16 branches
#   SKIP_PROBES=1 ./launch_size_budget.sh   # no lr probes
#   SKIP_BRANCHES=1 ./launch_size_budget.sh # parents only, branch later
#   MODELS=225m ./launch_size_budget.sh     # one size
#   ./launch_size_budget.sh --seed 1        # extra args go to every run
#
# Branches are submitted immediately with --dependency=afterok on their parent,
# so slurm starts them once the parent finishes (and cancels them if it fails).
# They read the parent's checkpoint directory, which submit_train.slurm names
# after the parent's job id -- hence --parsable to capture it here.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

# 20B tokens needs FineWeb100B; FineWeb10B has 10.26B and the loader wraps
# silently, which would make 20B the only multi-epoch point on the budget axis.
DATA_DIR="${DATA_DIR:-/scratch/$USER/fineweb100B}"
shopt -s nullglob; shards=("$DATA_DIR"/fineweb_train_*.bin); shopt -u nullglob
if [[ ${#shards[@]} -lt 210 || ! -e "$DATA_DIR/fineweb_val_000000.bin" ]]; then
    echo "error: '$DATA_DIR' has ${#shards[@]}/210 train shards (and needs the val shard)." >&2
    echo "       Get them with: sbatch submit_download.slurm 210" >&2
    exit 1
fi
# submit_train.slurm puts each job's checkpoints in $RUN_ROOT/checkpoints/<name>-<id>.
RUN_ROOT="${RUN_ROOT:-$(sed -n 's/^RUN_ROOT="\${RUN_ROOT:-\(.*\)}"$/\1/p' submit_train.slurm)}"

MODELS="${MODELS:-225m 418m}"
BUDGETS=("1p25b:1788:2384" "2p5b:3576:4768" "5b:7153:9537" "10b:14305:19073")
declare -A CONFIG=([225m]="configs/normuon_225m.yaml" [418m]="configs/normuon_418m_20b.yaml")
declare -A PARENT_TIME=([225m]="20:00:00" [418m]="40:00:00")
declare -A BRANCH_TIME=([225m]="04:00:00" [418m]="08:00:00")
FRACTION=0.125

# dion3 only; the configs already turn on Polar Express + GNS for both arms.
dion_flags() { [[ "$1" == dion3 ]] && echo "--ortho_fraction $FRACTION"; }

submit() {  # submit <jobname> <time> <tags> <args...>; echoes the job id
    local name="$1" time="$2" tags="$3"; shift 3
    local id
    id=$(WANDB_TAGS="$tags" sbatch --parsable \
        --job-name="$name" --time="$time" \
        submit_train.slurm --data_dir "$DATA_DIR" "$@" | tail -1)
    echo "${id%%;*}"   # --parsable prints "jobid" or "jobid;cluster"
}

launch_parent() {  # launch_parent <model> <optimizer> <lr> <role> <extra args...>
    local model="$1" opt="$2" lr="$3" role="$4"; shift 4
    local name="${opt}_${model}" id
    id=$(submit "$name" "${PARENT_TIME[$model]}" "$opt,$model,20b-tokens,size-budget,f$FRACTION,lr$lr,$role" \
        --config "${CONFIG[$model]}" --optimizer "$opt" --lr "$lr" \
        --checkpoint_freq 2000 --checkpoint_steps 1788,3576,7153,14305 \
        $(dion_flags "$opt") "$@")
    echo "=== $name lr=$lr ($role): job $id ===" >&2   # stdout carries the id alone
    echo "$id"
}

launch_branches() {  # launch_branches <model> <optimizer> <lr> <parent job id>
    local model="$1" opt="$2" lr="$3" parent="$4"
    local ckpt="$RUN_ROOT/checkpoints/${opt}_${model}-${parent}"
    for spec in "${BUDGETS[@]}"; do
        IFS=: read -r label branch total <<<"$spec"
        local id
        # afterok: the checkpoint exists long before the parent ends, but waiting
        # costs only queue time and means a failed parent cancels its branches.
        # --fresh_wandb_run: the checkpoint carries the parent's wandb id, and
        # resuming it would overwrite the parent's curve from the branch on.
        id=$(submit "${opt}_${model}_${label}" "${BRANCH_TIME[$model]}" \
            "$opt,$model,${label}-tokens,size-budget,f$FRACTION,lr$lr,branch" \
            --dependency="afterok:$parent" \
            --config "${CONFIG[$model]}" --optimizer "$opt" --lr "$lr" \
            --num_iterations "$total" --checkpoint_freq 0 --fresh_wandb_run \
            `# warmup_ratio applies to the branch's own num_iterations, but the` \
            `# branch starts past it, so the warmup the model saw is the parent's 381`  \
            --checkpoint_dir "$ckpt" --checkpoint_name "checkpoint-$branch" \
            $(dion_flags "$opt") "${@:5}")
        echo "    branch $label (resume $branch -> $total steps): job $id" >&2
    done
}

for model in $MODELS; do
    for opt in normuon dion3; do
        parent=$(launch_parent "$model" "$opt" 0.01 parent "$@" | tail -1)
        [[ -n "${SKIP_BRANCHES:-}" ]] || launch_branches "$model" "$opt" 0.01 "$parent" "$@"
    done
done

if [[ -z "${SKIP_PROBES:-}" ]]; then
    echo "=== lr probes: 225m at lr 0.005, no branches ==="
    for opt in normuon dion3; do
        launch_parent 225m "$opt" 0.005 lr-probe "$@" >/dev/null
    done
fi
