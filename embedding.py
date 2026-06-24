"""

We need some model that can produce embeddings (some vector of numbers)
from images. ViT CLS token or CNN perhaps. Because these are small and
have a good performance I chose

google/efficientnet-b0
google/mobilenet_v2_1.0_224

The model has some methods:

images_to_tensor(images)     -> pixel tensor (batch, 3, 224, 224)
forward_to_embedding(images) -> embedding    (batch, hidden_dim)
forward_from_embedding(emb)  -> logits       (batch, classes)
forward(images)              -> logits       (batch, classes), the whole model
id2label()                   -> {output index: class name}

get_embeddings caches the embeddings of an imagenet split to disk so the cnn
only runs once per (model, split, n).

"""

from pathlib import Path
from typing import Literal
from dataclasses import dataclass

import numpy as np
import torch
import PIL.Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from dataset import ImageNetDataset, Split

ImageBatch = list[PIL.Image.Image] | torch.Tensor | np.ndarray

EMBED_CACHE_DIR = Path("embeddings")
EMBED_BATCH_SIZE = 256  # how many images go through the cnn at once (keep small)


class ImageNetClassifier:
    """Base class: a pretrained ImageNet-1k classifier split at its embedding.

    Subclasses only set `model_id`. The embedding backbone and classifier head
    are reached generically, so the same methods work for every model.
    """

    model_id: str

    def __init__(self, device: torch.device | str):
        assert device is not None
        self.device = torch.device(device)
        self.preprocessor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def images_to_tensor(self, images: ImageBatch) -> torch.Tensor:
        """images -> preprocessed pixel tensor (batch, 3, 224, 224) on the device."""
        return self.preprocessor(images, return_tensors="pt").to(self.device)["pixel_values"]

    def forward_to_embedding(self, images: ImageBatch) -> torch.Tensor:
        """images -> pooled embedding (batch, hidden_dim), the second to last layer."""
        backbone = self.model.base_model(pixel_values=self.images_to_tensor(images))
        return backbone.pooler_output.flatten(1)  # (batch, hidden_dim)

    def forward_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """embedding (batch, hidden_dim) -> class logits (batch, classes)."""
        return self.model.classifier(embedding)  # (batch, classes)

    def forward(self, images: ImageBatch) -> torch.Tensor:
        """images -> class logits (batch, classes), the whole model."""
        return self.forward_from_embedding(self.forward_to_embedding(images))

    def id2label(self) -> dict[int, str]:
        """the map from a model output index to its class name."""
        return self.model.config.id2label


class EfficientNetB0(ImageNetClassifier):
    """google/efficientnet-b0

    Params:    ~5.3M
    Embedding: 1280-dim
    Speed:     ~0.39 GFLOPs/image (about 100s img/s).
    """

    model_id = "google/efficientnet-b0"


class MobileNetV2(ImageNetClassifier):
    """google/mobilenet_v2_1.0_224

    Params:    ~3.5M  (the smallest / fastest here)
    Embedding: 1280-dim
    Speed:     ~0.30 GFLOPs/image (100+ images per second)

    Note: this model has 1001 output classes, not 1000. Index 0 is an extra
    background class, so imagenet class i sits at logit index i + 1. To compare
    with imagenet labels, subtract 1 from the predicted index (argmax - 1).
    """

    model_id = "google/mobilenet_v2_1.0_224"


ModelName = Literal["efficientnet", "mobilenet"]

MODELS: dict[ModelName, type[ImageNetClassifier]] = {
    "efficientnet": EfficientNetB0,
    "mobilenet": MobileNetV2,
}


@dataclass
class EmbeddingData:
    """The embeddings of a split, together with each image's class label."""

    embeddings: torch.Tensor  # (n, d_input)
    labels: torch.Tensor  # (n,) imagenet class index 0-999


@torch.no_grad()
def get_embeddings(
    model: ImageNetClassifier,
    model_name: ModelName,
    split: Split,
    n_samples: int,
    use_cache: bool,
) -> EmbeddingData:
    """Embed the first n_samples images and keep their labels.

    Images go through the cnn in small batches (EMBED_BATCH_SIZE) so they fit in
    gpu memory. The result is cached so the cnn only runs once per (model, split, n).
    """
    cache = EMBED_CACHE_DIR / f"{model_name}_{split}_n{n_samples}.pt"
    if use_cache and cache.exists():
        saved = torch.load(cache)
        return EmbeddingData(saved["embeddings"], saved["labels"])

    images: list = []
    labels: list[int] = []
    chunks: list[torch.Tensor] = []
    for image, label in ImageNetDataset(split, max_samples=n_samples):
        images.append(image)
        labels.append(label)
        if len(images) == EMBED_BATCH_SIZE:
            chunks.append(model.forward_to_embedding(images).cpu())  # (batch, d_input)
            images = []
    if images:
        chunks.append(model.forward_to_embedding(images).cpu())  # (rest, d_input)
    embeddings = torch.cat(chunks)  # (n_samples, d_input)
    label_tensor = torch.tensor(labels)  # (n_samples,)

    if use_cache:
        EMBED_CACHE_DIR.mkdir(exist_ok=True)
        torch.save({"embeddings": embeddings, "labels": label_tensor}, cache)
    return EmbeddingData(embeddings, label_tensor)
