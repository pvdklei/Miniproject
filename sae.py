"""

Sparse Autoencoder (SAE) as used in

"Interpretable and Testable Vision Features via Sparse Autoencoders"
https://arxiv.org/pdf/2502.06755

encode:   h = W_enc (x - b_dec) + b_enc
          f = ReLU(h)
decode:   x_hat = W_dec f + b_dec

It is trained on the embeddings produced by the image model so that the
sparse embeddings can be interpreted as (close to) monosemantic features.

"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    sae_in: torch.Tensor  # (batch, d_input)    input, for convenience
    features: torch.Tensor  # (batch, n_features) sparse code f = ReLU(h)
    reconstruction: torch.Tensor  # (batch, d_input)    x_hat


@dataclass
class SAELoss:
    total: torch.Tensor  # reconstruction + l1_coefficient * sparsity
    reconstruction: torch.Tensor  # ||x - x_hat||_2^2
    sparsity: torch.Tensor  # L1 norm of the sparse code f


class SparseAutoEncoder(nn.Module):
    def __init__(self, d_input: int, n_features: int):
        """

        d_input:    dimensionality of the embeddings fed in (e.g. 1280 for
                    efficientnet-b0, 768 for the ViT used in the paper)
        n_features: size of the sparse dictionary. The paper uses a 32x
                    expansion factor, i.e. n_features = 32 * d_input.

        """
        super().__init__()

        self.d_input = d_input
        self.n_features = n_features

        # Encoder maps d_input -> n_features, decoder maps n_features -> d_input.
        self.W_enc = nn.Parameter(torch.empty(n_features, d_input))  # (n_features, d_input)
        self.b_enc = nn.Parameter(torch.zeros(n_features))  # (n_features,)

        self.W_dec = nn.Parameter(torch.empty(d_input, n_features))  # (d_input, n_features)
        self.b_dec = nn.Parameter(torch.zeros(d_input))  # (d_input,)

        self.mean = torch.zeros(d_input)  # (d_input,) set after training or loading

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Initialise the decoder columns to unit norm, and tie the encoder to
        # the decoder transpose (a common, stable initialisation for SAEs).
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)  # (d_input, n_features)
            self.W_enc.data = self.W_dec.data.t().clone()  # (n_features, d_input)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d_input) -> sparse features f: (batch, n_features)."""
        h = F.linear(x - self.b_dec, self.W_enc, self.b_enc)  # (batch, n_features)
        return F.relu(h)  # (batch, n_features)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, n_features) -> reconstruction: (batch, d_input)."""
        return F.linear(features, self.W_dec, self.b_dec)  # (batch, d_input)

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """raw embedding -> (mean-subtracted unit-norm embedding, the length removed).

        The length is returned so unnormalize can put the embedding back later.
        """
        centered = x - self.mean  # (batch, d_input)
        scale = centered.norm(dim=-1, keepdim=True)  # (batch, 1)
        return centered / scale.clamp(min=1e-8), scale  # (batch, d_input), (batch, 1)

    def unnormalize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """inverse of normalize: scale back up and add the mean."""
        return x * scale + self.mean  # (batch, d_input)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        features = self.encode(x)  # (batch, n_features)
        reconstruction = self.decode(features)  # (batch, d_input)
        return SAEOutput(sae_in=x, features=features, reconstruction=reconstruction)

    @classmethod
    def loss(cls, output: SAEOutput, l1_coefficient: float) -> SAELoss:
        """SAE training loss from the paper:

            L = ||x - x_hat||_2^2 + lambda * S(f)

        The reconstruction term is the mean squared error and the sparsity
        term S uses the L1 norm of the sparse code f (L0 is used only for
        model selection, not training). `l1_coefficient` is lambda.
        """
        # mean over batch, summed over feature dimensions
        reconstruction = (
            (output.sae_in - output.reconstruction).pow(2).sum(dim=-1).mean()
        )  # scalar
        sparsity = output.features.abs().sum(dim=-1).mean()  # scalar
        total = reconstruction + l1_coefficient * sparsity  # scalar
        return SAELoss(total=total, reconstruction=reconstruction, sparsity=sparsity)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Project decoder columns back to unit norm.

        Call after every optimiser step (the paper normalises the columns of
        W_dec to unit length after every gradient update).
        """
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)  # (d_input, n_features)

    @torch.no_grad()
    def remove_parallel_decoder_grads(self) -> None:
        """Remove the gradient component parallel to each decoder column.

        Call after backward() and before the optimiser step. Together with
        normalize_decoder() this keeps W_dec exactly on the unit sphere so the
        unit-norm constraint does not interfere with the gradient signal.
        """
        if self.W_dec.grad is None:
            return
        parallel = (self.W_dec.grad * self.W_dec.data).sum(dim=0, keepdim=True)  # (1, n_features)
        self.W_dec.grad -= parallel * self.W_dec.data  # (d_input, n_features)
