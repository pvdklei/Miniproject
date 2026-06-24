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

names = model.id2label()
print(names[logits.argmax(-1).item()])

```

One thing to watch out for: mobilenet has 1001 output classes, not 1000. Index 0 is an extra background class, so imagenet class `i` sits at index `i + 1`. So to compare a mobilenet prediction with an imagenet label you subtract one (`argmax - 1`). Efficientnet has the normal 1000 classes and needs no shift.

## Training a SAE

To train an a sparse autoencoder on some models embeddings (second to last layer, `pooled_output` attribute) run something like. 

```sh 

python train.py --model efficientnet --l1-coefficient 8e-4 --seed 0
python train.py --model mobilenet --optimizer adamw --seed 1
python train.py --test-run            # quick smoke test, writes nothing

```

The model params and stats will be saved to `./output/<run>`, the embeddings produced by the model will be cached to `./embeddings`.

## Trained SAE's

We trained six autoencoders, three lambda values for each of the two models. All runs use the same config: AdamW, learning rate 1e-3, expansion factor 32 (so 40960 sparse features), 200 epochs, batch size 4096, 100000 train images, 10000 validation images, seed 0.

These are the scores on the validation set.

| model | lambda | recon error | L1 | active (L0) | acc with sae | acc without |
| --- | --- | --- | --- | --- | --- | --- |
| efficientnet | 4e-4 | 0.0088 | 50.6 | 1698 | 75.8% | 75.7% |
| efficientnet | 8e-4 | 0.0149 | 41.9 | 1713 | 75.7% | 75.7% |
| efficientnet | 1.6e-3 | 0.0385 | 28.2 | 1957 | 75.5% | 75.7% |
| mobilenet | 4e-4 | 0.0128 | 49.1 | 1653 | 71.5% | 71.6% |
| mobilenet | 8e-4 | 0.0154 | 42.7 | 1673 | 71.5% | 71.6% |
| mobilenet | 1.6e-3 | 0.0391 | 29.5 | 1867 | 71.5% | 71.6% |

`recon error` is the reconstruction error in the normalized space. `L1` is the sum of the feature sizes and `L0` is how many features are active (out of 40960), both measure sparsity. `acc with sae` is the imagenet top-1 accuracy when you push the reconstructed embedding through the classifier head, `acc without` is the same but with the original embedding. They are almost equal, so the autoencoder keeps the classification.

The lowest lambda (4e-4) was best for both models: lower reconstruction error and fewer active features.

## Using a trained SAE

Every run saves a `sae.pt` file. It is a plain dictionary, so you can load it without importing `train.py`. You only need `sae.py`.

```python

import torch
from sae import SparseAutoEncoder

ckpt = torch.load("output/<run>/sae.pt", weights_only=True)
sae = SparseAutoEncoder(ckpt["d_input"], ckpt["n_features"])
sae.load_state_dict(ckpt["state_dict"])
sae.mean = ckpt["normalizer_mean"]   # the normalization mean from training
sae.eval()

```

The autoencoder works on normalized embeddings, and it does the normalization for you. `normalize` subtracts the mean and scales to unit norm, and returns the length it removed so `unnormalize` can put the embedding back after decoding.

```python

# x is a batch of embeddings, shape (batch, d_input)
x_norm, scale = sae.normalize(x)        # subtract mean, scale to unit norm
sparse = sae.encode(x_norm)             # sparse features, shape (batch, n_features)
recon = sae.decode(sparse)              # reconstruction, still normalized
x_back = sae.unnormalize(recon, scale)  # back to the normal embedding space

```

See `sae.ipynb` for a full example that runs one image through the model and the autoencoder and compares the predictions.

## Attribution analysis

`attribute.py` runs attribution methods (with captum) on a trained autoencoder to find out what the sparse dimensions mean. There are two kinds, set with `--kind`.

`image_to_sparse` uses the image pixels as input and a sparse dimension as target, so you get heatmaps showing which parts of an image activate that dimension.

`sparse_to_logits` uses the sparse embeddings as input and the predicted class logit as target, so you get a score for how much each sparse dimension drives the classification.

```sh

python attribute.py --run efficientnet_..._seed0 --kind sparse_to_logits --num-images 256
python attribute.py --run efficientnet_..._seed0 --kind image_to_sparse --num-images 64 --top-j 5

```

The `--run` is a folder name in `./output` (its `sae.pt` and model are used). Results are saved under `./output/<run>/<kind>/<analysis-name>/`, where the analysis name encodes the config so a repeated analysis is skipped. Inside, the same results are grouped three ways so you can browse them however you like: by sparse `dimensions`, by `classes`, and by `images`. Each folder has plots, heatmaps and the raw attribution data.

Pick the attribution method with `--method`. There are gradient methods (`integrated_gradients`, `saliency`, `input_x_gradient`) and perturbation methods (`lime`, `kernel_shap`, `shapley_sampling`, `feature_ablation`). They all work for both kinds. The perturbation methods need groups of inputs to turn on and off: for images these are superpixels (from quickshift), for the sparse code each active dimension is its own group. They are slower, so `--num-images` should be smaller for them. `--n-samples` sets how many perturbations lime, kernel_shap and shapley_sampling draw.

For `image_to_sparse` with integrated gradients you can set the baseline with `--baseline` (`white` or `random`). The default is `random`, averaged over `--baseline-trials` random images, which gave the best results in the week 2 vision workshop.

## Research Question 1 - do attribution methods agree

Do different attribution methods identify the same sparse features as important for a classifier's prediction? This follows the agreement paradigm from `literature/agree.pdf`, which argues that good attribution methods should agree: if two methods disagree, at least one is underperforming (though agreement alone does not prove they are right). That paper asks how well attention based explanations correlate with feature attribution methods for NLP. We ask the same kind of question, but for the sparse dimensions: we take the `sparse_to_logits` scores, rank the dimensions per method, and measure how much the rankings overlap between every pair of methods.

We use four overlap measures, in `agree.py`: Kendall's tau (rank correlation), Overlap@k (fraction of the top-k shared), Jaccard@k (intersection over union of the top-k), and rank biased overlap (overlap of the top of the ranking, weighted so earlier ranks count more). The notebook `rq1_agreement.ipynb` averages these over many images and draws an agreement matrix per measure and per model, plus a demo that shows how all methods ranked a single image.

For `sparse_to_logits` the result is a bit degenerate, in an informative way. The path from the sparse code to a class logit is `decode` (a linear layer) followed by the classifier head, which for both our models is a single linear layer with no nonlinearity. So the whole path is one affine map: the logit is `M @ sparse + c`, where `M` is the single matrix you get by multiplying the decoder and classifier weights. On a linear function, input-times-gradient, integrated gradients, feature ablation and shapley sampling all reduce to the same formula, `attribution_d = sparse_d * M[target, d]`, so they give identical rankings and agree perfectly. Saliency differs (it uses the raw gradient and drops the `sparse_d` factor) and lime and kernel_shap differ (they fit a sampled surrogate).

Because four methods collapse together there, the notebook also runs the same analysis on `image_to_sparse`. There the input is the image pixels and the path runs through the whole non-linear CNN backbone, so the methods no longer collapse. We compare how the methods rank the pixels of an image for the same (dimension, image) target. These matrices are more interesting: the gradient methods (input-times-gradient and saliency) agree most, while lime is almost orthogonal to the rest.

## Research Question 2 - do sparse features activate on similar images

Do sparse autoencoder features only activate on similar images? In `literature/sae.pdf` they motivate the SAE as an interpretability tool by showing dimensions with a clear meaning: their figure 1 has a dimension that fires on cigarettes, one on ships, one on bridges, and the images that activate each one all share that theme. We check whether this holds in general for our smaller efficientnet and mobilenet SAEs, or whether those are hand picked cases. In practice most dimensions fire on a fairly random mix of images, and only some have a nameable theme.

The notebook `rq2_features.ipynb` uses the `image_to_sparse` analysis. The activations matter most here (which images turn a dimension on), but the attribution heatmaps add insight into what part of each image caused the activation. It has a demo that shows, for a chosen dimension, the images that activate it most and their attribution maps, so you can judge whether they share a theme. The final figure is an illustrative diagram with two panels: one where the activating images share a clear theme (an interpretable feature) and one where they do not (a non-interpretable feature), with placeholders for the percentages we find when qualitatively checking a sample of dimensions.


