#!/bin/bash
# submit the attribution sweep. the training is already done, so this only needs
# the trained sae.pt files in output/ and the cached embeddings in embeddings/.
# setup runs first to make sure the env has captum and scikit-image installed.
set -e

cd "$(dirname "$0")"   # run from the jobs folder so the .job names resolve
mkdir -p logs   # slurm needs the log dir to exist before the jobs start

sid=$(sbatch --parsable setup.job)
aid=$(sbatch --parsable --dependency=afterok:$sid attribute.job)
echo "setup     = $sid"
echo "attribute = $aid  (24 array tasks, after setup)"
