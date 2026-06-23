# Miniproject

We'll be training a SAE on embeddings produces by pre trained (classification) models, so that we can interpet these sparse monosemantic embeddings.

And we'll also use these to...

## Setup

You will need to install the requirements 

```sh 

pip install -r requirements.txt

```

And if you need to use the imagenet data (`dataset.py`) you will need to login with the huggingface cli 

```sh

huggingface-cli login

``` 

And then just follow instructions: make an access token and paste it into the terminal.

Also go to the imagenet page and accept the terms of conditions.

[https://huggingface.co/datasets/ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

## Using the imagenet data

`dataset.py` streams imagenet images and their labels from huggingface. You need to be logged in first (see Setup). To grab one image you can pick an offset with a seed so you always get the same one.

```python

import random
from datasets import load_dataset

offset = random.Random(0).randint(0, 199)   # the seed makes it repeatable
stream = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True).skip(offset).take(1)
example = next(iter(stream))
image, label = example["image"].convert("RGB"), example["label"]

```

For many images use the dataset class, it gives `(image, label)` pairs.

```python

from dataset import ImageNetDataset

for image, label in ImageNetDataset(split="train", max_samples=1000):
    ...

```

## Using an embedding model

`embedding.py` has the classification models, split into a backbone and a classifier head. There is `EfficientNetB0` and `MobileNetV2`. The image is a PIL image.

```python

from embedding import EfficientNetB0

model = EfficientNetB0(device="cpu")   # or "cuda" or "mps"

logits = model.forward([image])                   # full model: image -> class logits
embedding = model.forward_to_embedding([image])   # image -> embedding (the pooled second to last layer)
logits = model.forward_from_embedding(embedding)  # embedding -> class logits

```

So `forward(x)` is the same as `forward_from_embedding(forward_to_embedding(x))`. To turn a class index into a name use the label map from the model.

```python

names = model.model.config.id2label
print(names[logits.argmax(-1).item()])

```

## Training a SAE

To train an a sparse autoencoder on some models embeddings (second to last layer, `pooled_output` attribute) run something like. 

```sh 

python train.py --model efficientnet --l1-coefficient 8e-4 --seed 0
python train.py --model mobilenet --optimizer adamw --seed 1
python train.py --test-run            # quick smoke test, writes nothing

```

The model params and stats will be saved to `./output/<run>`, the embeddings produced by the model will be cached to `./embeddings`.

## Using a SAE

Every run saves a `sae.pt` file. It is a plain dictionary, so you can load it without importing `train.py`. You only need `sae.py`.

```python

import torch
from sae import SparseAutoEncoder

ckpt = torch.load("output/<run>/sae.pt", weights_only=True)
sae = SparseAutoEncoder(ckpt["d_input"], ckpt["n_features"])
sae.load_state_dict(ckpt["state_dict"])
sae.eval()

mean = ckpt["normalizer_mean"]   # the mean embedding from training

```

The autoencoder works on normalized embeddings. So before you encode you subtract the mean and scale to unit norm, the same way training did. After you decode you put the normalization back.

```python

# x is a batch of embeddings, shape (batch, d_input)
centered = x - mean
norm = centered.norm(dim=-1, keepdim=True)
x_norm = centered / norm

sparse = sae.encode(x_norm)    # sparse features, shape (batch, n_features)
recon = sae.decode(sparse)     # reconstruction, still normalized
x_back = recon * norm + mean   # back to the normal embedding space

```

See `sae.ipynb` for a full example that runs one image through the model and the autoencoder and compares the predictions.