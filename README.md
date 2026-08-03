# NNsCompletelyFromScratch
Neural Networks made completely from scratch. The only external library allowed for computation is Numpy. Ie, no pytorch/huggingface/sklearn (besides downloading datasets), etc.

Currently capabilities:
- Solve XOR (MSE, Cross Entropy)
- Linear regression through feed forward neural network
- Classification problems

Current Features
- He Kaiming, Xavier Glorot, and LeCun weights initializations
  - Uses Box Muller Gaussian with option to truncate tails (+ variance correction if this is chosen)
- Minmax/Z-score normalization
- ERF, PDF, CDF vectorized functions
- Feed forward layer
- Network builder
