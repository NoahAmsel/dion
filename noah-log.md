# Training
- Running default with configs/dion_160m.yaml. 
    - Running on an a100 with 80 gb. config uses `device_batch_size: 32` and the memory usage is about 20 gb. This was with one GPU only. 
    - this is job noahamselsteam/dion-debug/run-nin9plb5-history:v0
    - runs at 5.1 sec / iter. estimates about 4.25 hours for all 3000 iterations
- Running default with configs/dion_160m.yaml except 4x: `device_batch_size: 128`
    - I got this working thru slurm
    - https://wandb.ai/noahamselsteam/dion-debug/runs/iiqsjcy3
    - it's not going much faster. still 4+ hrs
    - answer: we're always doing 3000 iters regardless of thteir size?? but that doesn't make sense because I confirmed that the number of gradient accumulation steps went down by a factor of 4