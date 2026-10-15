# FashionVision

FashionVision is a small PyTorch project that trains a convolutional neural network to classify 28 × 28 grayscale fashion images into 10 classes. It reads the Fashion-MNIST IDX training files included in `data/`, evaluates models using a held-out validation split, and writes predictions for the supplied test images to `y_test.npy`.

**Repository:** [github.com/LordJohan-9/FashionVision](https://github.com/LordJohan-9/FashionVision)

## Features

- A compact CNN with batch normalization, dropout, and a 10-class output layer.
- A reproducible 55,000/5,000 training-validation split and fixed random seed.
- GPU training when CUDA is available, with CPU fallback.
- Best-validation-accuracy checkpoint selection during training.
- NumPy prediction output with one integer class label per test image.

## Tech stack

- Python
- PyTorch
- NumPy

## Project files

```text
.
├── data/
│   ├── train-images-idx3-ubyte.gz   # 60,000 training images
│   └── train-labels-idx1-ubyte.gz  # 60,000 training labels
├── train_predict.py                 # Training and prediction entry point
├── requirements.txt                 # Python dependencies
├── LICENSE                           # MIT License
├── x_test_a87ff6.npy                # 9,000 unlabeled test images (28 × 28)
└── y_test.npy                       # Generated predictions (created on run)
```

The input data files are part of this project and are required by the script. `y_test.npy` is an output file and is excluded from Git so it can be regenerated.

## Requirements

- Python 3.9 or later
- `pip`
- Optional: a CUDA-compatible GPU and suitable PyTorch installation for GPU acceleration

## Getting started

### 1. Get the project

```bash
git clone https://github.com/LordJohan-9/FashionVision.git
cd FashionVision
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU use, install the PyTorch build matching your CUDA setup by following the current instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Train and generate predictions

Run from the project root:

```bash
python train_predict.py
```

The script trains for 15 epochs, prints the device and per-epoch training loss and validation accuracy, then saves the best validation model's predictions to `y_test.npy`. Existing output is overwritten.

## How it works

1. Reads the compressed IDX images and labels from `data/`.
2. Converts image values to floating point, scales them to `[0, 1]`, and standardizes them using the training data mean and standard deviation.
3. Uses a seeded random permutation to reserve 5,000 examples for validation and train on the remaining 55,000.
4. Trains a convolutional neural network using cross-entropy loss with label smoothing, AdamW, and a cosine learning-rate schedule.
5. Keeps the model state with the highest validation accuracy.
6. Loads `x_test_a87ff6.npy`, applies the same scaling and normalization, and predicts a class for each image.
7. Saves predictions as a one-dimensional `int64` NumPy array in `y_test.npy`.

The expected test input shape is `(9000, 28, 28)`; the script checks that images are 28 × 28 and validates that the output contains one class index in `[0, 9]` per input image.

## Customization

Training settings are defined near the top of `train_predict.py`:

- `SEED`: random seed used for Python, NumPy, and PyTorch.
- `EPOCHS`: number of training epochs.
- `TRAIN_IMAGES` / `TRAIN_LABELS`: paths to the compressed IDX training files.
- `TEST_IMAGES`: path to the unlabeled NumPy test array.
- `OUTPUT`: path where predictions are saved.

The architecture is defined in the `FashionCNN` class.

## Troubleshooting

- **Missing IDX file:** Confirm both compressed files exist in `data/` with the expected names.
- **Missing test array:** Confirm `x_test_a87ff6.npy` is in the project root.
- **Import error for `torch` or `numpy`:** Activate the virtual environment and reinstall dependencies with `python -m pip install -r requirements.txt`.
- **Slow training:** Training runs on CPU if CUDA is unavailable. A supported CUDA-enabled PyTorch installation can substantially speed up training.

## License

This project is licensed under the [MIT License](LICENSE).
