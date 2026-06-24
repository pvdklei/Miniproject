"""

Attribution analysis on a trained sparse autoencoder, following idea2.md.

It runs one analysis per call, like train.py runs one training per call. There are
two kinds:

image_to_sparse:  input is the image pixels, target is a sparse dimension. Shows
                  which parts of the image activate that dimension (heatmaps).
sparse_to_logits: input is the sparse code, target is the predicted class logit.
                  Shows which sparse dimensions drive the classification.

Results are written under the analyzed run, in a config named folder so re-running
the same analysis just prints where it is and stops:

output/<run>/<kind>/<analysis-name>/
    config.json
    dimensions/<dim>/      results grouped by sparse dimension
    classes/<label>_<name>/   same results grouped by class
    images/<i>/            same results grouped by image

For image_to_sparse, every image that activates a dimension is recorded under that
dimension (dimensions/<dim>/activators.json and a gallery image), so you can see
all of its activators. The expensive pixel attribution only runs on each image's
top-k dimensions, and those heatmaps go in dimensions/<dim>/attributions/.

The cached embeddings from train.py are reused, so the CNN is not re-run for the
sparse_to_logits kind, and only for the actual heatmaps in image_to_sparse.

Examples:

python attribute.py --run efficientnet_..._seed0 --kind sparse_to_logits --num-images 256
python attribute.py --run efficientnet_..._seed0 --kind image_to_sparse --num-images 64 --top-k 5
python attribute.py --run efficientnet_..._seed0 --kind sparse_to_logits --test-run

"""

from __future__ import annotations

import json
import shutil
import random
import argparse
from typing import Literal, cast
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import PIL.Image
import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import captum.attr as captum
from skimage.segmentation import quickshift

from embedding import (
    ImageNetClassifier, MODELS, ModelName, EmbeddingData, get_embeddings, EMBED_CACHE_DIR,
)
from dataset import ImageNetDataset, Split
from sae import SparseAutoEncoder
from train import Config, OUTPUT_DIR

Kind = Literal["image_to_sparse", "sparse_to_logits"]
Method = Literal[
    "integrated_gradients", "saliency", "input_x_gradient",
    "lime", "kernel_shap", "shapley_sampling", "feature_ablation",
]
Device = Literal["cpu", "cuda", "mps"]
ImageBaseline = Literal["white", "random"]

METHODS: dict[Method, type[captum.Attribution]] = {
    "integrated_gradients": captum.IntegratedGradients,
    "saliency": captum.Saliency,
    "input_x_gradient": captum.InputXGradient,
    "lime": captum.Lime,
    "kernel_shap": captum.KernelShap,
    "shapley_sampling": captum.ShapleyValueSampling,
    "feature_ablation": captum.FeatureAblation,
}

# gradient methods read the model derivatives, the rest perturb groups of inputs
GRADIENT_METHODS = {"integrated_gradients", "saliency", "input_x_gradient"}

# quickshift superpixel defaults, from the week 2 vision workshop
SUPERPIXEL_KERNEL_SIZE = 4
SUPERPIXEL_MAX_DIST = 200
SUPERPIXEL_RATIO = 0.2

# how many dimensions to keep in the various summaries
TOP_PER_IMAGE = 25
TOP_PER_CLASS = 40
TOP_OVERALL = 200
BAR_K = 20

# image_to_sparse: how many of a dimension's activators to show in its gallery
GALLERY_SIZE = 16

# overrides applied by --test-run
TEST_RUN: dict[str, int] = dict(
    num_images=2, top_k=2, n_steps=4, baseline_trials=2, n_samples=8
)


@dataclass
class AttributeConfig:
    """Everything that defines (and reproduces) an analysis. All from the CLI."""

    run: str  # the output/<run> folder whose sae.pt is analyzed
    kind: Kind  # image_to_sparse or sparse_to_logits (see the two functions below)
    method: Method  # which captum attribution method to use
    split: Split  # which imagenet split the images come from
    num_images: int  # how many images to analyze
    top_k: int  # image_to_sparse: how many of each image's top dimensions to attribute
    baseline: ImageBaseline  # image_to_sparse: integrated gradients baseline
    baseline_trials: int  # image_to_sparse: random baselines to average over
    n_steps: int  # integrated gradients steps
    n_samples: int  # perturbation samples for lime, kernel_shap and shapley_sampling
    save_pt: bool  # also save the raw attribution tensors next to the png heatmaps
    seed: int  # random seed

    @property
    def analysis_name(self) -> str:
        common = f"{self.method}_{self.split}_n{self.num_images}_seed{self.seed}"
        if self.method == "integrated_gradients":
            common += f"_steps{self.n_steps}"
        elif self.method in ("lime", "kernel_shap", "shapley_sampling"):
            common += f"_samples{self.n_samples}"
        if self.kind == "image_to_sparse":
            base = "white" if self.baseline == "white" else f"random{self.baseline_trials}"
            return f"{common}_k{self.top_k}_{base}"
        return common


# LOADING

def load_sae(run_dir: Path, device: Device) -> tuple[SparseAutoEncoder, ModelName]:
    """Load the trained autoencoder from a run folder (the plain sae.pt dict)."""
    ckpt = torch.load(run_dir / "sae.pt", weights_only=True)
    run_config = Config(**ckpt["config"])  # rebuild the typed training config
    sae = SparseAutoEncoder(int(ckpt["d_input"]), int(ckpt["n_features"]))
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.mean = ckpt["normalizer_mean"].to(device)  # set the mean on the device too
    sae.eval()
    return sae, run_config.model


def cache_size(cache_path: Path) -> int:
    """The number in a cache file name like efficientnet_validation_n10000.pt."""
    return int(cache_path.stem.split("_n")[-1])


def load_cached(
    model: ImageNetClassifier, model_name: ModelName, split: Split, num_images: int,
) -> EmbeddingData:
    """Reuse the cached embeddings, then take the first num_images of them."""
    # find the biggest cache train.py already wrote for this model and split,
    # so we do not have to run the cnn again
    caches = sorted(EMBED_CACHE_DIR.glob(f"{model_name}_{split}_n*.pt"), key=cache_size)
    cache_n = cache_size(caches[-1]) if caches else num_images

    # load it (this hits the cache) and keep only the first num_images
    data = get_embeddings(model, model_name, split, cache_n, use_cache=True)
    n = min(num_images, data.embeddings.shape[0])
    return EmbeddingData(data.embeddings[:n], data.labels[:n])


def output_name(model: ImageNetClassifier, output_index: int) -> str:
    """Name of a raw model output index (mobilenet has a background class at 0)."""
    return model.id2label().get(output_index, str(output_index))


def class_name(model: ImageNetClassifier, label: int) -> str:
    """Name of an imagenet label 0-999."""
    labels = model.id2label()
    # mobilenet has 1001 classes with a background class at 0, so shift by one
    index = label + 1 if len(labels) == 1001 else label
    return labels.get(index, str(label))


def folder_name(class_label: str) -> str:
    """Turn a class name into a safe folder name (first word, no spaces or slashes)."""
    return class_label.split(",")[0].strip().replace(" ", "_").replace("/", "_")


def class_folder(model: ImageNetClassifier, label: int) -> str:
    """The class folder name for a label, like '91_coucal'."""
    return f"{label}_{folder_name(class_name(model, label))}"


# CAPTUM STUFF

class PixelToSparse(nn.Module):
    """pixel tensor -> embedding -> normalize -> encoder, returns the sparse code.

    captum varies the pixels, so the embedding cnn is run on the pixel tensor here.
    """

    def __init__(self, model: ImageNetClassifier, sae: SparseAutoEncoder) -> None:
        super().__init__()
        self.embedder = model.model.base_model  # the cnn up to the embedding
        self.sae = sae

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        emb = self.embedder(pixel_values=pixel_values).pooler_output.flatten(1)  # (B, d_input)
        normalized, _ = self.sae.normalize(emb)
        return self.sae.encode(normalized)  # (B, n_features)


class SparseToLogit(nn.Module):
    """sparse -> decode -> unnormalize -> classifier head, returns logits."""

    def __init__(self, model: ImageNetClassifier, sae: SparseAutoEncoder) -> None:
        super().__init__()
        self.classifier = model.forward_from_embedding
        self.sae = sae

    def forward(self, sparse: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        emb = self.sae.unnormalize(self.sae.decode(sparse), scale)  # (B, d_input)
        return self.classifier(emb)  # (B, classes)


def run_attr(
    method: captum.Attribution, inputs: torch.Tensor, target: int,
    config: AttributeConfig, baselines: torch.Tensor | None = None,
    additional: tuple[torch.Tensor, ...] | None = None,
    feature_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one captum method, giving each kind of method only the args it accepts.

    The gradient methods read the model derivatives. The perturbation methods
    (lime, kernel_shap, shapley_sampling, feature_ablation) turn groups of inputs
    on and off; feature_mask says which inputs form a group (a superpixel, or one
    sparse dimension) and n_samples is how many perturbations they draw.
    """
    if isinstance(method, captum.IntegratedGradients):
        out = method.attribute(inputs, target=target, baselines=baselines,
                               n_steps=config.n_steps, additional_forward_args=additional)
    elif isinstance(method, (captum.Saliency, captum.InputXGradient)):
        out = method.attribute(inputs, target=target, additional_forward_args=additional)
    elif isinstance(method, (captum.Lime, captum.KernelShap, captum.ShapleyValueSampling)):
        out = method.attribute(inputs, target=target, baselines=baselines,
                               additional_forward_args=additional, feature_mask=feature_mask,
                               n_samples=config.n_samples)
    elif isinstance(method, captum.FeatureAblation):
        out = method.attribute(inputs, target=target, baselines=baselines,
                               additional_forward_args=additional, feature_mask=feature_mask)
    else:
        raise ValueError(f"unsupported method {type(method).__name__}")
    return cast(torch.Tensor, out)


def image_baselines(model: ImageNetClassifier, config: AttributeConfig) -> list[torch.Tensor]:
    """The integrated gradients baselines for image_to_sparse, as pixel tensors.

    A random baseline averaged over a few trials gave the best results in the
    week 2 vision workshop on baselines and integrated gradients, so it is the
    default. The other option is a plain white image.
    """
    if config.baseline == "white":
        white = PIL.Image.new("RGB", (224, 224), (255, 255, 255))
        return [model.images_to_tensor([white])]
    baselines = []
    for _ in range(config.baseline_trials):
        noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        baselines.append(model.images_to_tensor([PIL.Image.fromarray(noise, "RGB")]))
    return baselines


def superpixel_mask(image: PIL.Image.Image, pixel_values: torch.Tensor) -> torch.Tensor:
    """Group the pixels of an image into superpixels with quickshift.

    The result is a (1, 1, H, W) tensor of segment ids that captum uses as the
    feature_mask, so the perturbation methods toggle whole superpixels at a time.
    """
    array = np.asarray(image.resize((pixel_values.shape[-1], pixel_values.shape[-2]))) / 255.0
    segments = quickshift(array, kernel_size=SUPERPIXEL_KERNEL_SIZE,
                          max_dist=SUPERPIXEL_MAX_DIST, ratio=SUPERPIXEL_RATIO, channel_axis=-1)
    return torch.tensor(segments, device=pixel_values.device).view(1, 1, *segments.shape)


def active_dim_mask(sparse: torch.Tensor) -> torch.Tensor:
    """Group the sparse code so each active dimension is its own feature.

    There are 40k dimensions but only a fraction are active, so perturbing every
    dimension is wasteful. We give each active dimension its own group and put all
    inactive ones in a single shared group, which captum then barely perturbs.
    """
    mask = torch.zeros_like(sparse, dtype=torch.long)  # (1, n_features) all in group 0
    active = (sparse[0] > 0).nonzero().flatten()
    mask[0, active] = torch.arange(1, len(active) + 1, device=sparse.device)
    return mask


def attribute_pixels(
    method: captum.Attribution, pixel_values: torch.Tensor, dim: int,
    config: AttributeConfig, baselines: list[torch.Tensor], feature_mask: torch.Tensor,
) -> torch.Tensor:
    """Attribute a sparse dimension to the image pixels, averaging over the baselines.

    Gradient methods ignore the baseline and the superpixel mask, the perturbation
    methods use both. Integrated gradients averages over the random baselines.
    """
    attrs = [run_attr(method, pixel_values, dim, config, baselines=b, feature_mask=feature_mask)
             for b in baselines]
    return torch.stack(attrs).mean(0)


# SAVING OUTPUT

def save_overlay(
    targets: list[tuple[Path, str]], image: PIL.Image.Image,
    heatmap: np.ndarray, attr: torch.Tensor, save_pt: bool,
) -> None:
    """Render the image with the heatmap once, then copy it to every target dir.

    targets is a list of (dir, basename). The raw attribution tensor is only
    written when save_pt is set; the png already shows the heatmap.
    """
    first_dir, first_name = targets[0]
    first_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.imshow(heatmap, cmap="jet", alpha=0.5)
    ax.axis("off")
    png = first_dir / f"{first_name}.png"
    fig.savefig(png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    if save_pt:
        torch.save(attr, first_dir / f"{first_name}.pt")
    for d, name in targets[1:]:
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(png, d / f"{name}.png")
        if save_pt:
            shutil.copy(first_dir / f"{first_name}.pt", d / f"{name}.pt")


def bar_plot(path: Path, vec: torch.Tensor, title: str, k: int = BAR_K) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = vec.topk(min(k, vec.numel()))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(top.values)), top.values.tolist())
    ax.set_xticks(range(len(top.values)))
    ax.set_xticklabels([str(i) for i in top.indices.tolist()], rotation=90, fontsize=6)
    ax.set(title=title, xlabel="dimension", ylabel="attribution")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_top_dims(
    out_dir: Path, vec: torch.Tensor, k: int, title: str, extra: dict,
) -> None:
    """Write the top-k dimensions of a score vector as a json and a bar plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    top = vec.topk(min(k, vec.numel()))
    data = {**extra, "top_dims": top.indices.tolist(),
            "scores": [float(v) for v in top.values]}
    (out_dir / "top_dims.json").write_text(json.dumps(data, indent=2))
    bar_plot(out_dir / "top_dims.png", vec, title)


def save_gallery(path: Path, images: list[PIL.Image.Image], titles: list[str]) -> None:
    """Save a grid of the images that activate a dimension, strongest first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# THE TWO ANALYSES

@dataclass
class Activator:
    """An image that activated a sparse dimension, kept for the gallery."""

    activation: float
    image_index: int
    label: int
    image: PIL.Image.Image


def write_activators(
    model: ImageNetClassifier, sae: SparseAutoEncoder, data: EmbeddingData,
    split: Split, device: Device, out_dir: Path,
) -> None:
    """Record, for each sparse dimension, all the images that activate it.

    This only depends on the sparse code, not on the attribution method, so it is
    written once per run and shared by every method's analysis. Each dimension
    gets an activators.json (all its activators, strongest first) and a gallery.
    """
    emb = data.embeddings.to(device)
    with torch.no_grad():
        normalized, _ = sae.normalize(emb)
        sparse = sae.encode(normalized)  # (N, n_features)

    activators: dict[int, list[Activator]] = {}
    images = ImageNetDataset(split, max_samples=emb.shape[0])
    for i, (image, _label) in enumerate(images):
        label = int(data.labels[i])
        resized = image.resize((224, 224))
        for d in (sparse[i] > 0).nonzero().flatten().tolist():
            activators.setdefault(d, []).append(Activator(float(sparse[i, d]), i, label, resized))

    for d, found in activators.items():
        found.sort(key=lambda a: a.activation, reverse=True)
        dim_dir = out_dir / "dimensions" / str(d)
        dim_dir.mkdir(parents=True, exist_ok=True)
        (dim_dir / "activators.json").write_text(json.dumps(
            [{"image_index": a.image_index, "label": a.label,
              "label_name": class_name(model, a.label), "activation": a.activation}
             for a in found], indent=2))
        shown = found[:GALLERY_SIZE]
        save_gallery(dim_dir / "activators.png", [a.image for a in shown],
                     [f"img{a.image_index} {a.activation:.2f}" for a in shown])


def attribute_image_to_sparse(
    config: AttributeConfig, model: ImageNetClassifier, sae: SparseAutoEncoder,
    data: EmbeddingData, analysis_dir: Path, device: Device, test_run: bool,
) -> None:
    """Attribute each image's top-k active dimensions back to the image pixels.

    The heatmaps are saved per dimension, per class and per image. The activator
    galleries are written separately (see write_activators) so they are not
    duplicated across methods."""
    method = METHODS[config.method](PixelToSparse(model, sae))
    baselines = image_baselines(model, config)  # integrated gradients baselines

    emb = data.embeddings.to(device)
    with torch.no_grad():
        normalized, _ = sae.normalize(emb)
        sparse = sae.encode(normalized)  # (N, n_features)

    images = ImageNetDataset(config.split, max_samples=emb.shape[0])
    for i, (image, _label) in enumerate(images):
        label = int(data.labels[i])
        acts = sparse[i]
        active = (acts > 0).nonzero().flatten().tolist()
        if not active:
            continue

        pixel_values = model.images_to_tensor([image])  # (1, 3, H, W)
        resized = image.resize((pixel_values.shape[-1], pixel_values.shape[-2]))
        if test_run:
            continue

        # attribute only the strongest top-k dimensions back to the image pixels
        # (the perturbation methods toggle whole superpixels via the feature mask)
        feature_mask = superpixel_mask(image, pixel_values)
        for d in acts.topk(min(config.top_k, len(active))).indices.tolist():
            attr = attribute_pixels(method, pixel_values, d, config, baselines, feature_mask)
            channels = attr.abs().sum(1).squeeze(0)  # (H, W)
            heatmap = (channels / channels.max().clamp(min=1e-8)).detach().cpu().numpy()
            act = float(acts[d])
            # save the heatmap grouped by dimension and by image (no classes/ copy:
            # for image_to_sparse it was an exact duplicate, browse by images instead)
            targets = [
                (analysis_dir / "dimensions" / str(d) / "attributions", f"act{act:.3f}_img{i}"),
                (analysis_dir / "images" / str(i), f"dim{d}_act{act:.3f}"),
            ]
            save_overlay(targets, resized, heatmap, attr.squeeze(0).detach().cpu(), config.save_pt)


def attribute_sparse_to_logits(
    config: AttributeConfig, model: ImageNetClassifier, sae: SparseAutoEncoder,
    data: EmbeddingData, analysis_dir: Path, device: Device, test_run: bool,
) -> None:
    """For every image, take its sparse code and ask captum which sparse dimensions
    pushed the model to its predicted class. We save the scores per image, and add
    them up per dimension and per class. After the loop we write those summaries."""
    method = METHODS[config.method](SparseToLogit(model, sae))
    emb = data.embeddings.to(device)
    n_features = sae.n_features

    # running totals so we can also report by dimension and by class at the end
    dim_sum = torch.zeros(n_features)
    class_sum: dict[int, torch.Tensor] = {}
    class_count: dict[int, int] = {}

    for i in range(emb.shape[0]):
        # normalize the embedding, get its sparse code and the predicted class,
        # then attribute that logit to the sparse dimensions
        x = emb[i:i + 1]  # (1, d_input)
        normalized, scale = sae.normalize(x)
        with torch.no_grad():
            sparse = sae.encode(normalized)  # (1, n_features)
            pred = int(model.forward_from_embedding(x).argmax(-1))
        # the perturbation methods only toggle the active dimensions, the rest
        # share one group; gradient methods ignore the mask
        attr = run_attr(method, sparse, pred, config, baselines=torch.zeros_like(sparse),
                        additional=(scale,), feature_mask=active_dim_mask(sparse))
        attr = attr.squeeze(0).detach().cpu()  # (n_features,)
        label = int(data.labels[i])

        # save this image's top dimensions
        if not test_run:
            out = analysis_dir / "images" / str(i)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(attr, out / "scores.pt")
            save_top_dims(out, attr, TOP_PER_IMAGE, f"image {i}: top dims for the prediction",
                          extra={"predicted": pred, "predicted_name": output_name(model, pred),
                                 "label": label, "label_name": class_name(model, label)})

        # add this image into the running totals
        dim_sum += attr
        if label not in class_sum:
            class_sum[label] = torch.zeros(n_features)
            class_count[label] = 0
        class_sum[label] += attr
        class_count[label] += 1

    if test_run:
        return

    # all images done, now write the summaries.
    # per dimension: only the strongest dimensions overall get a folder
    mean_dim = dim_sum / emb.shape[0]
    for d in mean_dim.topk(min(TOP_OVERALL, n_features)).indices.tolist():
        out = analysis_dir / "dimensions" / str(d)
        out.mkdir(parents=True, exist_ok=True)
        (out / "score.json").write_text(json.dumps(
            {"mean_attribution": float(mean_dim[d])}, indent=2))
    bar_plot(analysis_dir / "dimensions" / "summary.png", mean_dim, "top dims over all images")

    # per class: the dimensions that matter most for each class
    for label, total in class_sum.items():
        mean = total / class_count[label]
        out = analysis_dir / "classes" / class_folder(model, label)
        save_top_dims(out, mean, TOP_PER_CLASS, f"top dims for {out.name}",
                      extra={"count": class_count[label]})


# MAIN

def main(config: AttributeConfig, device: Device, test_run: bool) -> None:
    """Load the trained autoencoder and its model, reuse the cached embeddings, run
    the chosen analysis and save it. If the same analysis was already done, just say
    where it is and stop."""
    print(f"device: {device}  run: {config.run}  kind: {config.kind}"
          f"{'  [TEST RUN]' if test_run else ''}")

    run_dir = OUTPUT_DIR / config.run
    if not (run_dir / "sae.pt").exists():
        raise SystemExit(f"no sae.pt found in {run_dir}")

    analysis_dir = run_dir / config.kind / config.analysis_name
    if not test_run and (analysis_dir / "config.json").exists():
        print(f"analysis already done, see {analysis_dir}")
        return

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    sae, model_name = load_sae(run_dir, device)
    model = MODELS[model_name](device=device)
    data = load_cached(model, model_name, config.split, config.num_images)
    print(f"analyzing {data.embeddings.shape[0]} images from {model_name} {config.split}")

    if config.kind == "image_to_sparse":
        # the activator galleries depend only on the sparse code, not the method, so
        # write them once into a shared folder that every method's analysis reuses
        # (the first run fills it; later runs with more images do not rebuild it)
        activators_dir = run_dir / "image_to_sparse" / "activators"
        if not test_run and not activators_dir.exists():
            write_activators(model, sae, data, config.split, device, activators_dir)
        attribute_image_to_sparse(config, model, sae, data, analysis_dir, device, test_run)
    elif config.kind == "sparse_to_logits":
        attribute_sparse_to_logits(config, model, sae, data, analysis_dir, device, test_run)
    else:
        raise Exception("Unknown kind of attribution targets")

    if test_run:
        print(f"[test-run] dimensions OK; would write -> {analysis_dir}")
        return
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    print(f"saved -> {analysis_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attribution analysis on a trained SAE.")
    p.add_argument("--run", required=True,
                   help="the output/<run> folder to analyze (its sae.pt and model)")
    p.add_argument("--kind", choices=["image_to_sparse", "sparse_to_logits"], required=True,
                   help="image_to_sparse (pixel heatmaps) or sparse_to_logits (dim scores)")
    p.add_argument("--method", choices=list(METHODS), default="integrated_gradients",
                   help="captum attribution method")
    p.add_argument("--split", choices=["train", "validation"], default="validation",
                   help="which imagenet split the images come from")
    p.add_argument("--num-images", type=int, default=256,
                   help="how many images to analyze")
    p.add_argument("--top-k", type=int, default=5,
                   help="image_to_sparse: how many top dimensions per image to attribute")
    p.add_argument("--baseline", choices=["white", "random"], default="random",
                   help="image_to_sparse: integrated gradients baseline")
    p.add_argument("--baseline-trials", type=int, default=5,
                   help="image_to_sparse: random baselines to average over")
    p.add_argument("--n-steps", type=int, default=32,
                   help="integrated gradients steps")
    p.add_argument("--n-samples", type=int, default=200,
                   help="perturbation samples for lime, kernel_shap and shapley_sampling")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                   help="where to run (auto picks cuda, then mps, then cpu)")
    p.add_argument("--no-pt", action="store_true",
                   help="do not save the raw attribution tensors, only the png heatmaps")
    p.add_argument("--test-run", action="store_true",
                   help="quick smoke test: tiny sizes, writes nothing")
    return p.parse_args()


def pick_device(choice: str) -> Device:
    """Turn the --device argument into a real device, picking one for 'auto'."""
    if choice == "cpu" or choice == "cuda" or choice == "mps":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    args = parse_args()
    config = AttributeConfig(
        run=args.run, kind=cast(Kind, args.kind), method=cast(Method, args.method),
        split=cast(Split, args.split), num_images=args.num_images, top_k=args.top_k,
        baseline=cast(ImageBaseline, args.baseline), baseline_trials=args.baseline_trials,
        n_steps=args.n_steps, n_samples=args.n_samples, save_pt=not args.no_pt, seed=args.seed,
    )
    if args.test_run:
        config = replace(config, **TEST_RUN)
    main(config, pick_device(args.device), args.test_run)
