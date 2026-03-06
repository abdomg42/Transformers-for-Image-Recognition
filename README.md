# Vision Transformer (ViT) - PyTorch Implementation

A PyTorch implementation of the **Vision Transformer (ViT)** from the paper:

> **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**  
> Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.  
> 📄 [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

---

## Architecture

![Vision Transformer Architecture](architecture.png)

The model works as follows:
1. **Patch Extraction**: The input image is split into fixed-size patches (16×16 pixels).
2. **Linear Projection**: Each patch is flattened and projected into a latent embedding space.
3. **Class Token**: A learnable `[CLS]` token is prepended to the sequence of patch embeddings.
4. **Positional Embedding**: Positional embeddings are added to encode spatial information.
5. **Transformer Encoder**: The sequence is passed through a stack of encoder blocks, each containing:
   - Layer Normalization
   - Multi-Head Self-Attention
   - Residual Connection
   - Layer Normalization
   - MLP (Feed-Forward Network)
   - Residual Connection
6. **MLP Head**: The `[CLS]` token output is passed to an MLP head for classification.

---

## Model Configuration (ViT-Base)

| Parameter        | Value  |
|------------------|--------|
| Patch Size       | 16×16  |
| Latent Size      | 768    |
| Num Heads        | 12     |
| Num Encoders     | 12     |
| Dropout          | 0.1    |
| Image Size       | 224×224|
| Num Classes      | 10     |

---

## Requirements

```bash
pip install torch torchvision einops torchsummary tqdm
```

---

## Usage

```python
import torch
from ViT import Vit

model = Vit().to('cuda')
test_input = torch.randn((1, 3, 224, 224)).to('cuda')
output = model(test_input)
print(output.size())  # torch.Size([1, 10])
```

---

## Reference

- [Official ViT Repository (Google Research)](https://github.com/google-research/vision_transformer)
- [Paper: arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
