#!/bin/bash
# submit the whole pipeline so it runs in order while i sleep:
#   setup (env + test) -> prepare (cache embeddings) -> train (the 6 sae runs)
# each step only starts if the previous one finished ok.
set -e

cd "$(dirname "$0")"   # run from the jobs folder so the .job names resolve
mkdir -p logs   # slurm needs the log dir to exist before the jobs start

sid=$(sbatch --parsable setup.job)
pid=$(sbatch --parsable --dependency=afterok:$sid prepare.job)
tid=$(sbatch --parsable --dependency=afterok:$pid train.job)

echo "setup   = $sid"
echo "prepare = $pid  (after setup)"
echo "train   = $tid  (after prepare)"
