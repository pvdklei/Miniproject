"""

Dataset for training the sparse autoencoder.

ImageNet-1k on hugging face is private: run `huggingface-cli login` and accept the
dataset terms first.

The full train split is ~150GB, so we stream it and only ever hold one batch of
images in memory at a time. Each item is an image with its class label. 

"""

from typing import Literal, Iterator

from torch.utils.data import IterableDataset
import PIL.Image

from datasets import load_dataset

Split = Literal["train", "validation"]
DATASET_ID = "ILSVRC/imagenet-1k"


class ImageNetDataset(IterableDataset):
    """Streams the first `max_samples` ImageNet-1k (image, label) pairs."""

    def __init__(
        self,
        split: Split = "train",
        dataset_id: str = DATASET_ID,
        max_samples: int | None = None,
    ):
        self.split = split
        self.dataset_id = dataset_id
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[tuple[PIL.Image.Image, int]]:
        stream = load_dataset(self.dataset_id, split=self.split, streaming=True)
        if self.max_samples is not None:
            stream = stream.take(self.max_samples)
        for example in stream:
            # some imagenet images are greyscale or cmyk, so force rgb
            image = example["image"].convert("RGB")
            yield image, example["label"]
