#!/usr/bin/env bash
# Grid the microbench profiler over {model sizes} x {kernel/algorithm variants},
# all on configs/1b_baseline.yml. Model sizes are passed as CLI overrides
# (model_dim/n_layer/n_head), which take precedence over the config -- so no
# per-size config files are needed. Each run writes its active_step_metrics.json
# into results/<size>/<family>/<variant>/ -- one folder per (size, family), with
# a subfolder per condition, so different optimizer families never mix.
#
# Run this from an interactive session INSIDE the container (where torch is
# available). For a batch job, use submit_variants.slurm, which enters the
# container and calls this script.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# GPU visibility comes from the environment: enter via singro-bash (interactive)
# or run via sbatch, both of which land in the SLURM job cgroup so
# CUDA_VISIBLE_DEVICES is already scoped to exactly the GPUs you were allocated.
# A size's "--fs_size N" then just sets how many of those GPUs a run uses.
CONFIG="configs/1b_baseline.yml"

# Model sizes (ordered). Each entry: "name|<override flags>".
# FILL IN real model_dim/n_layer/n_head values.
SIZES=(
    "1b|--model_dim 1536 --n_layer 24 --n_head 16"
    "3b|--model_dim 2304 --n_layer 32 --n_head 18"
    "4b|--model_dim 2560 --n_layer 36 --n_head 32"
    "7b|--model_dim 4096 --n_layer 32 --n_head 32"
    # 8-GPU (FSDP) sizes: add --fs_size 8 and run() uses 8 GPUs. Needs a whole node.
    "1b-8gpu|--model_dim 1536 --n_layer 24 --n_head 16 --fs_size 8"
    "3b-8gpu|--model_dim 2304 --n_layer 32 --n_head 18 --fs_size 8"
    "4b-8gpu|--model_dim 2560 --n_layer 36 --n_head 32 --fs_size 8"
    "7b-8gpu|--model_dim 4096 --n_layer 32 --n_head 32 --fs_size 8"
    "14b-8gpu|--model_dim 5120 --n_layer 40 --n_head 40 --fs_size 8"
)

# Args:
#   $1 = size selector: "all-1gpu" (default) = every single-GPU size,
#        "all-8gpu" = every --fs_size (FSDP) size, or a specific size name.
#   $2 = optimizer family: "muon-family" (default) or "normuon-family"
SIZE_ARG="${1:-all-1gpu}"
FAMILY="${2:-muon-family}"

# The family picks the two optimizers substituted into the variants below:
#   UNFILTERED = full-orthogonalization optimizer (variants 3-6)
#   FILTERED   = low-rank / fractional optimizer   (variants 7-8)
case "$FAMILY" in
    muon-family)    UNFILTERED=muon;    FILTERED=dion2 ;;
    normuon-family) UNFILTERED=normuon; FILTERED=nordion2 ;;
    *) echo "Unknown family '$FAMILY'. Valid: muon-family, normuon-family" >&2; exit 1 ;;
esac

# Variants (ordered). Each entry: "name|<flags>". The optimizer is substituted
# from the selected family via $UNFILTERED / $FILTERED, expanded here at
# definition time -- no post-hoc string rewriting.
VARIANTS=(
    "1-plain|--optimizer $UNFILTERED --no_triton"
    "2-triton|--optimizer $UNFILTERED"
    "3-baseline|--optimizer $UNFILTERED --use_gns_package --no_triton"
    "4-cutlass|--optimizer $UNFILTERED --use_gns_package"
    "5-gns|--optimizer $UNFILTERED --use_gns_package --no_triton --use_gns_alg"
    "6-gns-cutlass|--optimizer $UNFILTERED --use_gns_package --use_gns_alg"
    "7-dion2-0.5|--optimizer $FILTERED --use_gns_package --use_gns_alg --ortho_fraction 0.5"
    "8-dion2-0.25|--optimizer $FILTERED --use_gns_package --use_gns_alg --ortho_fraction 0.25"
    "9-adamw|--optimizer adamw"
)

run() {
    # run <out_dir> [extra flags...]
    # nproc_per_node = N from a "--fs_size N" flag (FSDP), else 1. GPU visibility
    # is inherited from the SLURM allocation (see header) -- not set here.
    local out="$1"; shift
    mkdir -p "$out"
    local nproc=1 prev=""
    for a in "$@"; do
        [[ "$prev" == "--fs_size" ]] && nproc="$a"
        prev="$a"
    done
    echo "  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}, nproc_per_node=$nproc)"
    torchrun --standalone --nproc_per_node="$nproc" \
        profile_training_step.py --config "$CONFIG" --profile_out "$out" \
        --no_profile --profile_wait 0 --profile_warmup 50 --profile_active 25 \
        "$@"
}

selected=()
for s in "${SIZES[@]}"; do
    case "$SIZE_ARG" in
        all-1gpu) [[ "${s#*|}" != *--fs_size* ]] && selected+=("$s") ;;
        all-8gpu) [[ "${s#*|}" == *--fs_size* ]] && selected+=("$s") ;;
        *)        [[ "${s%%|*}" == "$SIZE_ARG" ]] && selected+=("$s") ;;
    esac
done
if [[ ${#selected[@]} -eq 0 ]]; then
    echo "Unknown selector '$SIZE_ARG'. Use all-1gpu, all-8gpu, or one of: ${SIZES[*]%%|*}" >&2
    exit 1
fi
SIZES=("${selected[@]}")

echo "Optimizer family: $FAMILY  (unfiltered=$UNFILTERED, filtered=$FILTERED)"

# A crashing variant must not take the rest of the sweep with it: under `set -e`
# one bad run aborted the whole job and cost us the six variants queued behind it
# (jobs 14931328 / 14931329 died at 2-triton with only 1-plain collected).
# Failures are collected and re-reported at the end, and the script still exits
# non-zero so SLURM marks the job failed.
failures=()

for s in "${SIZES[@]}"; do
    sname="${s%%|*}"; sflags="${s#*|}"
    for v in "${VARIANTS[@]}"; do
        vname="${v%%|*}"; vflags="${v#*|}"
        echo "=== ${sname} / ${FAMILY} / ${vname}  (${sflags} ${vflags}) ==="
        # sflags/vflags are intentionally word-split into separate CLI args.
        # A failing `run` in an `if` condition does not trip `set -e`.
        # shellcheck disable=SC2086
        if ! run "results/${sname}/${FAMILY}/${vname}" $sflags $vflags; then
            echo "!!! FAILED: ${sname}/${FAMILY}/${vname} -- continuing with the remaining variants" >&2
            failures+=("${sname}/${FAMILY}/${vname}")
        fi
    done
done

if (( ${#failures[@]} > 0 )); then
    echo "Done, but ${#failures[@]} variant(s) FAILED:" >&2
    printf '  %s\n' "${failures[@]}" >&2
    echo "Metrics for the variants that did run are in results/<size>/<family>/<variant>/" >&2
    exit 1
fi

echo "Done. Metrics in results/<size>/<family>/<variant>/active_step_metrics_<ts>.json"
