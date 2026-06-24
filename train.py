"""

Train a sparse autoencoder on image model embeddings, following

"Interpretable and Testable Vision Features via Sparse Autoencoders"
https://arxiv.org/pdf/2502.06755

Pipeline:

ImageNet image
-> model.forward_to_embedding   embedding            (n, d_input)   [precomputed once, cached]
-> normalize                    x (mean-sub, unit)   (batch, d_input)
-> SAE.encode                   sparse features f    (batch, n_features)
-> SAE.decode                   reconstruction x_hat (batch, d_input)
-> loss = ||x - x_hat||^2 + lambda * |f|_1
-> backward
-> update SAE       (decoder columns renormalised after each step)

Embeddings are computed once per (model, split, n) and cached under embeddings/,
so the expensive CNN runs a single time and training is cheap.

Every training run is saved in output/<run-name>/ where the name encodes the config.
Rerunning an existing config stops immediately. Each run folder contains:

config.json     the exact hyperparameters
sae.pt          the trained SAE (state dict + dims + config + mean)
metrics.json    per-epoch training history and final validation metrics
*.png           loss / sparsity plots

Examples:
    python train.py --model efficientnet --l1-coefficient 8e-4 --seed 0
    python train.py --model mobilenet --optimizer adamw --seed 1
    python train.py --test-run            # quick smoke test, writes nothing

"""

from __future__ import annotations

import json
import random
import argparse
from typing import Literal, Callable, cast
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")  # no display on the cluster compute nodes
import matplotlib.pyplot as plt

from embedding import MODELS, ModelName, get_embeddings
from dataset import Split
from sae import SparseAutoEncoder

OptimizerName = Literal["adam", "adamw"]

OUTPUT_DIR = Path("output")
PLOT_DPI = 120
WARMUP_STEPS = 500  # paper: linearly warm up the LR from 0 over this many steps
TRAIN_SPLIT: Split = "train"
VAL_SPLIT: Split = "validation"

# Overrides applied by --test-run.
TEST_RUN: dict[str, int] = dict(
    expansion_factor=2,
    num_epochs=1,
    batch_size=8,
    train_samples=16,
    val_samples=16,
)


@dataclass
class Config:
    """Everything that defines (and reproduces) a run. Use the CLI args to set these."""

    model: ModelName
    optimizer: OptimizerName
    l1_coefficient: float  # lambda in L = ||x - x_hat||^2 + lambda * |f|_1
    learning_rate: float
    expansion_factor: int  # dictionary size = expansion_factor * embedding dim
    num_epochs: int
    batch_size: int  # SAE minibatch (the CNN uses its own EMBED_BATCH_SIZE)
    train_samples: int  # subset of ImageNet train (full is 1.28M)
    val_samples: int
    seed: int

    @property
    def run_name(self) -> str:
        return (
            f"{self.model}_{self.optimizer}_l1{self.l1_coefficient:g}"
            f"_lr{self.learning_rate:g}_x{self.expansion_factor}"
            f"_e{self.num_epochs}_b{self.batch_size}_n{self.train_samples}"
            f"_seed{self.seed}"
        )


@dataclass
class EvalMetrics:
    recon: float  # mean reconstruction loss ||x - x_hat||^2
    sparsity: float  # mean L1 norm of the sparse code f
    total: float  # mean total loss (recon + lambda * sparsity)
    l0: float  # mean number of active features per example


@dataclass
class TrainHistory:
    train_total: list[float] = field(default_factory=list)
    train_recon: list[float] = field(default_factory=list)
    train_sparsity: list[float] = field(default_factory=list)
    val_total: list[float] = field(default_factory=list)
    val_recon: list[float] = field(default_factory=list)
    val_l0: list[float] = field(default_factory=list)


@dataclass
class Metrics:
    history: TrainHistory
    final: EvalMetrics


@dataclass
class SAECheckpoint:
    config: Config
    d_input: int
    n_features: int
    mean: torch.Tensor
    state_dict: dict[str, torch.Tensor]


# EVALUATION

@torch.no_grad()
def evaluate(
    sae: SparseAutoEncoder,
    embeddings: torch.Tensor,
    l1_coefficient: float,
    batch_size: int,
    device: str,
) -> EvalMetrics:
    """Mean reconstruction loss, sparsity loss, total loss and L0 on a split."""
    n = 0
    recon = sparsity = total = l0 = 0.0
    for (emb,) in DataLoader(TensorDataset(embeddings), batch_size=batch_size):
        x, _ = sae.normalize(emb.to(device))  # (batch, d_input)
        out = sae(x)  # features (batch, n_features)
        loss = SparseAutoEncoder.loss(out, l1_coefficient)  # scalars
        batch = out.features.shape[0]  # int (this batch's size)
        recon += loss.reconstruction.item() * batch
        sparsity += loss.sparsity.item() * batch
        total += loss.total.item() * batch
        l0 += (out.features > 0).float().sum().item()  # mask (batch, n_features) -> float
        n += batch
    return EvalMetrics(recon / n, sparsity / n, total / n, l0 / n)

# TRAINING

def train(
    config: Config,
    train_emb: torch.Tensor,
    val_emb: torch.Tensor,
    device: str,
    generator: torch.Generator,
) -> tuple[SAECheckpoint, Metrics]:
    """Train a single SAE on precomputed embeddings."""
    d_input = train_emb.shape[1]  # int
    n_features = config.expansion_factor * d_input

    sae = SparseAutoEncoder(d_input, n_features).to(device)
    # Paper preprocessing: the normalization mean is the train mean embedding.
    sae.mean = train_emb.mean(dim=0).to(device)  # (d_input,)
    # Paper init: b_dec is the mean of the normalized train embeddings.
    normalized, _ = sae.normalize(train_emb.to(device))  # (n, d_input)
    sae.b_dec.data = normalized.mean(dim=0)  # (d_input,)

    optimizers: dict[OptimizerName, Callable[..., torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    optimizer = optimizers[config.optimizer](sae.parameters(), lr=config.learning_rate)

    train_loader = DataLoader(
        TensorDataset(train_emb), batch_size=config.batch_size,
        shuffle=True, generator=generator,
    )

    # Paper LR schedule: linear warmup from 0 to the max. The paper warms up over
    # WARMUP_STEPS, but with cached embeddings our runs have far fewer steps, so we
    # cap warmup at a small fraction of the total so training actually happens.
    total_steps = config.num_epochs * len(train_loader)
    warmup = max(1, min(WARMUP_STEPS, total_steps // 20))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )

    history = TrainHistory()
    for epoch in range(config.num_epochs):
        sae.train()
        totals = torch.zeros(3)  # (3,) running total, recon, sparsity
        steps = 0
        for (emb,) in train_loader:
            x, _ = sae.normalize(emb.to(device))  # (batch, d_input)
            loss = SparseAutoEncoder.loss(sae(x), config.l1_coefficient)  # scalars

            optimizer.zero_grad()
            loss.total.backward()
            sae.remove_parallel_decoder_grads()  # paper: drop grads along W_dec cols
            optimizer.step()
            sae.normalize_decoder()  # paper: unit-norm W_dec columns
            scheduler.step()  # advance the LR warmup

            totals += torch.tensor(
                [loss.total.item(), loss.reconstruction.item(), loss.sparsity.item()]
            )  # (3,)
            steps += 1

        sae.eval()
        val = evaluate(sae, val_emb, config.l1_coefficient, config.batch_size, device)
        ep_total, ep_recon, ep_sparsity = (totals / steps).tolist()  # 3 floats
        history.train_total.append(ep_total)
        history.train_recon.append(ep_recon)
        history.train_sparsity.append(ep_sparsity)
        history.val_total.append(val.total)
        history.val_recon.append(val.recon)
        history.val_l0.append(val.l0)
        print(
            f"epoch {epoch + 1:2d}/{config.num_epochs}  train {ep_total:.4f}  "
            f"val recon {val.recon:.4f}  val L0 {val.l0:.1f}"
        )

    final = evaluate(sae, val_emb, config.l1_coefficient, config.batch_size, device)
    checkpoint = SAECheckpoint(config, d_input, n_features, sae.mean, sae.state_dict())
    return checkpoint, Metrics(history, final)


# OUTPUT

def save_run(run_dir: Path, checkpoint: SAECheckpoint, metrics: Metrics) -> None:
    """Save the config, metrics, weights and plots to the output folder."""
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(checkpoint.config), indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2))
    # Save a plain dict (tensors + primitives) so the SAE can be reloaded without
    # importing this module: build SparseAutoEncoder(d_input, n_features),
    # load_state_dict(state_dict), then set sae.mean = normalizer_mean.
    torch.save(
        {
            "config": asdict(checkpoint.config),
            "d_input": checkpoint.d_input,
            "n_features": checkpoint.n_features,
            "normalizer_mean": checkpoint.mean.cpu(),
            "state_dict": {k: v.cpu() for k, v in checkpoint.state_dict.items()},
        },
        run_dir / "sae.pt",
    )

    epochs = range(1, len(metrics.history.train_total) + 1)
    fig, (ax_loss, ax_l0) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(epochs, metrics.history.train_total, label="train total")
    ax_loss.plot(epochs, metrics.history.val_total, label="val total")
    ax_loss.plot(epochs, metrics.history.val_recon, label="val recon")
    ax_loss.set(title="Loss", xlabel="epoch", ylabel="loss")
    ax_loss.legend()
    ax_l0.plot(epochs, metrics.history.val_l0)
    ax_l0.set(title="Sparsity", xlabel="epoch", ylabel="mean L0 (active features)")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curves.png", dpi=PLOT_DPI)
    plt.close(fig)


# MAIN

def main(config: Config, device: str, test_run: bool, prepare_only: bool) -> None:
    print(
        f"device: {device}  run: {config.run_name}"
        f"{'  [TEST RUN]' if test_run else ''}{'  [PREPARE ONLY]' if prepare_only else ''}"
    )

    run_dir = OUTPUT_DIR / config.run_name
    if not test_run and not prepare_only and run_dir.exists():
        print(f"run already exists, nothing to do: {run_dir}")
        return

    # Seed every source of randomness so the same config is fully reproducible.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    model = MODELS[config.model](device=device)
    train_data = get_embeddings(
        model, config.model, TRAIN_SPLIT, config.train_samples, use_cache=not test_run,
    )
    val_data = get_embeddings(
        model, config.model, VAL_SPLIT, config.val_samples, use_cache=not test_run,
    )
    print(
        f"embeddings: train {tuple(train_data.embeddings.shape)}  "
        f"val {tuple(val_data.embeddings.shape)}"
    )

    # --prepare-only just fills the embedding cache, so a parallel lambda sweep can
    # then read it without every task recomputing (and racing on) the same file.
    if prepare_only:
        print("embeddings cached; exiting (prepare only)")
        return

    checkpoint, metrics = train(
        config, train_data.embeddings, val_data.embeddings, device, generator
    )

    if test_run:
        print(f"[test-run] dimensions OK; would write -> {run_dir}/")
        return
    save_run(run_dir, checkpoint, metrics)
    print(f"saved -> {run_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one SAE on image embeddings.")
    p.add_argument("--model", choices=list(MODELS), default="efficientnet")
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--l1-coefficient", type=float, default=8e-4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--expansion-factor", type=int, default=32)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--train-samples", type=int, default=50_000)
    p.add_argument("--val-samples", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    p.add_argument(
        "--test-run",
        action="store_true",
        help="quick smoke test: tiny data, 1 epoch, writes nothing",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="only compute and cache the embeddings, then exit (no training)",
    )
    return p.parse_args()


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    args = parse_args()
    config = Config(
        model=cast(ModelName, args.model),
        optimizer=cast(OptimizerName, args.optimizer),
        l1_coefficient=args.l1_coefficient,
        learning_rate=args.learning_rate,
        expansion_factor=args.expansion_factor,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
    )
    if args.test_run:
        config = replace(config, **TEST_RUN)
    main(config, resolve_device(args.device), args.test_run, args.prepare_only)
