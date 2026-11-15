# Adding optimizers muon, dion2 to size_budget sweep at 418m
MODELS=418m OPTIMIZERS="muon dion2" SKIP_PROBES=1 ./experiments/launch_size_budget.sh

# adding dion to the size budget sweep, after retuning learning rate
MODELS=418m OPTIMIZERS=dion LR=0.028 PARENT_WALLTIME=41:00:00 SKIP_PROBES=1 ./experiments/launch_size_budget.sh
