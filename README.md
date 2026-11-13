## FLOE for Sequential Kernel Two-Sample and Independence Testing

Implementation of ***F**ollow the **L**eave-one-**O**ut **E**power* betting strategy proposed in (link to paper).
The code is written in jax and all necessary packages can be installed with `pip install -r requirements.txt`.


### Data Preprocessing
Certain datasets (CIFAR-10 vs 10.1, RatInABox, HIGGS) require you to first preprocess the data.

CIFAR-10 vs 10.1:
1. Download [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) and move the cifar-10-batches-py folder into `src/data/cache/cifar10/raw/`
2. Download [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1) version `v6` (cifar10.1_v6_data.npy) and move it to `src/data/cache/cifar10/raw/`
3. run `python data/real/cifar10.py`

RatInABox:
1. run `python data/real/riab.py`

HIGGS:
1. Download [HIGGS](https://archive.ics.uci.edu/dataset/280/higgs) and move the uncompressed csv file into `src/data/cache/higgs/`
1. run `python data/real/higgs.py`


### Running Tests
Simply run `python seq-mmd.py` or `python seq-hsic.py` with user specified arguments. Relavent arguments include:
- `dataset`: path to the dataset yaml config file (these can be found in config/dataset). The datasets used in the experiments of the paper are:
    - Two-Sample:
        - (x-Normal) config/dataset/xgaussian/twosample/offline/corx=0.65.cory=0.generator.yml
        - (Sinusoid) config/dataset/sinusoid/twosample/offline/dim=2.f=3.5.generator.yml
        - (Blobs) config/dataset/blobs/twosample/offline/isotropy=custom.angle=custom.generator.yml
        - (RatInABox) config/dataset/riab/twosample/offline/wall_height=0.3.n_pcells=10.size=2000.yml
        - (Higgs) config/dataset/higgs/twosample/dataset.yml
        - (Cifar10 vs 10.1) config/dataset/cifar10/twosample/resnet50-features.dataset.yml
    - Independence:
        - (x-Normal) config/dataset/xgaussian/independence/offline/cor=0.5.generator.yml
        - (Sinusoid) config/dataset/sinusoid/independence/offline/f=5.dim=1.generator.yml
        - (RatInABox) config/dataset/riab/independence/offline/n_pcells=6.n_bvcells=6.wall_height=0.3.size=5000.yml
- `pred-strat`: prediction strategy used.
- `bet-strat`: betting strategy used.
- `bet-reg`: whether to regularize bets.

Example: running sequential kernel two-sample test with FLOE-R
```bash
python seq-mmd.py \
    --seed=5123512357913 \
    --dataset=config/dataset/xgaussian/twosample/offline/corx=0.65.cory=0.generator.yml \
    --n-tests=100 \
    --n-steps=1000 \
    --bandwidth=0.5 \
    --bet-strat=floe \
    --bet-reg
```
