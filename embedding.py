"""

We need some model that can produce embeddings (some vector of numbers)
from images. ViT CLS token or CNN perhaps. Because these are small and 
have a good performance I chose 

google/efficientnet-b0
google/mobilenet_v2_1.0_224

Each model exposes three methods:

forward(image)              -> logits     (batch, 1000)
forward_to_embedding(image) -> embedding  (batch, hidden_dim)
forward_from_embedding(emb) -> logits     (batch, 1000)

`forward_to_embedding` runs the model up to (but not including) the final
classifier. `forward_from_embedding` runs only that classifier.

"""

import numpy as np
import torch
import PIL.Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

ImageBatch = list[PIL.Image.Image] | torch.Tensor | np.ndarray


class ImageNetClassifier:
    """Base class: a pretrained ImageNet-1k classifier split at its embedding.

    Subclasses only set `model_id`. The backbone and classifier head are
    accessed generically (`self.model.base_model` is the backbone; the head is
    the model's `classifier`), so the same three methods work for every model.
    """

    model_id: str

    def __init__(self, device: torch.device | str):
        assert device is not None
        self.device = torch.device(device)
        self.preprocessor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, images: ImageBatch) -> dict:
        # returns {"pixel_values": (batch, 3, 224, 224)} on self.device
        return self.preprocessor(images, return_tensors="pt").to(self.device)

    def forward(self, images: ImageBatch) -> torch.Tensor:
        """batch of images -> logits of shape (batch, 1000)."""

        return self.model(**self._preprocess(images)).logits  # (batch, 1000)

    def forward_to_embedding(self, images: ImageBatch) -> torch.Tensor:
        """batch of images -> pooled embeddings of shape (batch, hidden_dim)."""

        outputs = self.model.base_model(**self._preprocess(images)) # (batch, hidden_dim, 1, 1)
        return outputs.pooler_output.flatten(1)  # (batch, hidden_dim)

    def forward_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """embedding -> logits of shape (batch, 1000).

        Runs only the final classifier head on a precomputed embedding.
        """
        # embedding: (batch, hidden_dim)
        return self.model.classifier(embedding)  # (batch, 1000)


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
