## FLOE for Sequential Kernel Two-Sample and Independence Testing

Implementation of ***F**ollow the **L**eave-one-**O**ut **E**power* betting strategy proposed in (link to paper).
The code is written in jax and all necessary packages can be installed with `pip install -r requirements.txt`.


### Data Preprocessing
Certain datasets (CIFAR-10 vs 10.1, RatInABox, HIGGS) require you to first preprocess the data.


CIFAR-10 vs 10.1:
1. Download [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) and move the cifar-10-batches-py folder into `src/data/cache/cifar10/raw/`
2. Download [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1) version `v6` (cifar10.1_v6_data.npy) and move it to `src/data/cache/cifar10/raw/`
3. `cd src/data/real` and run `python cifar10.py`

RatInABox:
1. `cd src/data/real` and run `python riab.py`

HIGGS:
1. Download [HIGGS](https://archive.ics.uci.edu/dataset/280/higgs) and move the uncompressed csv file into `src/data/cache/higgs/`
2. `cd src/data/real` and run `python higgs.py`


### Running Tests
