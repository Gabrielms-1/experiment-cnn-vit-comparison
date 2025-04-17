# In progress...

# Dataset Analysis

## Train Split:
Total images: 2604
Class distribution:
- Squirtle: 495 images (19.01%)
- Pikachu: 543 images (20.85%)
- Charmander: 540 images (20.74%)
- Mewtwo: 513 images (19.70%)
- Bulbasaur: 513 images (19.70%)

## Test Split:
Total images: 131
Class distribution:
- Squirtle: 25 images (19.08%)
- Pikachu: 32 images (24.43%)
- Charmander: 23 images (17.56%)
- Mewtwo: 28 images (21.37%)
- Bulbasaur: 23 images (17.56%)

## Valid Split:
Total images: 255
Class distribution:
- Squirtle: 55 images (21.57%)
- Pikachu: 45 images (17.65%)
- Charmander: 55 images (21.57%)
- Mewtwo: 59 images (23.14%)
- Bulbasaur: 41 images (16.08%)


# Experiment Setup

## Reproducibility (Seeding)

To ensure reproducibility across experiments for both ResNet-50 and ViT models, a consistent seeding strategy is employed, applied similarly in both the CNN and ViT training pipelines:

*   **Global Seeding:** A fixed seed value (configurable, e.g., via `train.yaml`) is set at the beginning of the main training script (`train.py`) using a utility function (`utils.set_seed`). This typically initializes the random number generators for Python's `random` module, NumPy, and PyTorch (CPU and GPU) to ensure consistent weight initialization and other stochastic processes.
*   **DataLoader Seeding:** To guarantee reproducible data loading, shuffling, and augmentation order, the `DataLoader` instances are explicitly seeded using the same fixed seed. This is achieved by:
    *   Passing a seeded `torch.Generator` to the `DataLoader` responsible for shuffling the training data.
    *   Using a custom `worker_init_fn` (`utils.seed_worker`) to ensure that each parallel data loading worker is also initialized with a unique but deterministic seed derived from the main seed.

This comprehensive seeding approach ensures that experiments run with the same seed value will have identical data splitting, shuffling, augmentation sequences, and model initializations, allowing for a fair and reproducible comparison between the different architectures.

# Model Architectures

## ResNet-50

The ResNet-50 model implemented follows the standard architecture using Bottleneck blocks. It's built from scratch without pre-trained weights.

**Architecture Details:**

*   **Input:** Accepts images with 3 color channels (e.g., RGB).
*   **Initial Convolution (Conv1):**
    *   Convolution: 64 filters, kernel size 7x7, stride 2, padding 3.
    *   Batch Normalization.
    *   ReLU activation.
    *   Max Pooling: kernel size 3x3, stride 2, padding 1.
*   **Residual Blocks (Bottleneck):** The core of the network consists of sequences of Bottleneck residual blocks. Each block has three convolutional layers (1x1, 3x3, 1x1) with Batch Normalization and ReLU activation after the first two convolutions. A skip connection adds the input of the block to the output before the final ReLU. The `expansion` factor for the number of output channels in the last convolution of the block is 4.
    *   **Layer 1 (Conv2\_x):** 3 Bottleneck blocks, `mid_channels=64`. Output channels: 256.
    *   **Layer 2 (Conv3\_x):** 4 Bottleneck blocks, `mid_channels=128`, stride 2. Output channels: 512.
    *   **Layer 3 (Conv4\_x):** 6 Bottleneck blocks, `mid_channels=256`, stride 2. Output channels: 1024.
    *   **Layer 4 (Conv5\_x):** 3 Bottleneck blocks, `mid_channels=512`, stride 2. Output channels: 2048.
*   **Final Layers:**
    *   Adaptive Average Pooling: Reduces spatial dimensions to 1x1.
    *   Flatten.
    *   Fully Connected Layer: Maps the 2048 features to the specified number of classes (default 5).

## Vision Transformer (ViT)

The Vision Transformer (ViT) model implemented is based on the architecture described in "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". It's also built from scratch without pre-trained weights.

**Architecture Details:**

*   **Input:** Accepts images with specified dimensions (`img_size`) and channels (`n_channels`).
*   **Patch Embedding (`PatchEmbedding`):**
    *   The input image is divided into fixed-size patches (`patch_size`).
    *   Each patch is linearly projected into a vector of dimension `d_model` using a 2D convolution.
    *   The output is a sequence of patch embeddings (tokens).
*   **Positional Encoding (`PositionalEncoding`):**
    *   A learnable `[CLS]` token is prepended to the sequence of patch embeddings.
    *   Standard sinusoidal positional encodings are added to the patch embeddings and the `[CLS]` token to retain positional information. The maximum sequence length (`max_seq_length`) depends on the number of patches plus the `[CLS]` token.
*   **Transformer Encoder (`TransformerEncoder`):**
    *   The core of the ViT consists of `n_layers` identical Transformer Encoder blocks.
    *   Each block contains:
        *   Layer Normalization (`ln1`).
        *   Multi-Head Self-Attention (`MultiHeadAttention`): Calculates attention across the sequence of tokens. It uses `n_heads`, where each head (`AttentionHead`) computes scaled dot-product attention. The outputs of all heads are concatenated and linearly projected.
        *   Dropout (`dropout1`).
        *   Skip Connection: Adds the input of the MHA sub-layer to its output.
        *   Layer Normalization (`ln2`).
        *   MLP (Feed-Forward Network): A two-layer MLP with a GELU activation in between. The expansion factor for the hidden layer is `r_mlp` (default 4).
        *   Dropout (`dropout2`).
        *   Skip Connection: Adds the input of the MLP sub-layer to its output.
*   **Classifier:**
    *   The final state of the `[CLS]` token from the output sequence of the last Transformer Encoder block is used as the aggregate representation of the image.
    *   A simple Linear layer maps this `[CLS]` token representation (dimension `d_model`) to the final number of classes (`n_classes`).
