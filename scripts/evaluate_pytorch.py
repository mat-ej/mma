# ---
# jupyter:
#   ploomber:
#     injected_manually: true
# ---

# + tags=["parameters"]
upstream = None
product = None
pytorch_config = None
target = None

# + tags=["injected-parameters"]
# Parameters
target = ["WINNER"]
pytorch_config = {}
upstream = {
    "split-train-test": {
        "train": "/home/m/repo/mma/products/data/train.csv",
        "test": "/home/m/repo/mma/products/data/test.csv",
    },
    "fit-pytorch": {
        "nb": "/home/m/repo/mma/products/reports/fit_pytorch.ipynb",
        "model": "/home/m/repo/mma/products/models/pytorch.pt",
    },
}
product = {"nb": "/home/m/repo/mma/products/reports/evaluate_pytorch.ipynb"}

