# DL-Resnet50 - Deep Learning Project

ResNet-50 image classification in **TensorFlow 2.21 / Keras 3.15**, trained from scratch and via
ImageNet transfer learning on two Kaggle datasets. The code is a reimplementation of the original
notebooks (`ant-bee-classification.ipynb`, `bird-species-classification.ipynb`, kept unchanged for
reference) as plain Python modules, with the bugs found in the notebooks fixed
(see [docs/MIGRATION.md](docs/MIGRATION.md)).

## Repository layout

```
DL-Resnet50/
├── train.py               # train from a YAML config
├── evaluate.py            # evaluate a trained model on an image folder
├── predict.py             # classify individual images
├── inspect_model.py       # print input shape, classes and parameter counts of a saved model
├── dlresnet/              # library code
│   ├── config.py          #   typed experiment config (YAML + --set overrides)
│   ├── layers.py          #   BottleneckBlock, CaffePreprocess (serializable Keras layers)
│   ├── model.py           #   ResNet-50 from scratch, transfer-learning model, fine-tuning helper
│   ├── data.py            #   tf.data image-folder pipeline and augmentation layers
│   ├── schedules.py       #   stateless StepDecay / WarmupCosine LR schedules, optimizer factory
│   ├── trainer.py         #   training loop, callbacks, artifacts
│   ├── evaluation.py      #   loading runs, metrics, confusion matrix
│   ├── inference.py       #   image loading and top-k prediction
│   ├── inspection.py      #   model summary for saved artifacts
│   └── plotting.py        #   loss/accuracy curves
├── configs/               # experiment configs
│   ├── ant_bee_scratch.yaml
│   ├── ant_bee_transfer.yaml
│   └── birds_scratch.yaml
├── scripts/
│   └── download_datasets.py
├── docs/MIGRATION.md      # notebook -> Python migration notes and bug analysis
├── models/                # legacy .h5 weights from the notebooks (Git LFS)
├── *.ipynb                # original notebooks
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands are run from the repository root.

## Datasets

Kaggle datasets:

- Hymenoptera (ant/bee classification): https://www.kaggle.com/datasets/thedatasith/Hymenoptera
- Birds 525 species: https://www.kaggle.com/datasets/gpiosenka/100-bird-species

Download them into `data/` (requires `pip install kagglehub` and Kaggle credentials):

```bash
python scripts/download_datasets.py all
```

Expected layout (one sub-folder per class in every split):

```
data/hymenoptera/{train,val}/{ants,bees}/*.jpg
data/100-bird-species/{train,valid,test}/<SPECIES>/*.jpg
```

## Training

```bash
python train.py --config configs/ant_bee_scratch.yaml
python train.py --config configs/ant_bee_transfer.yaml
python train.py --config configs/birds_scratch.yaml
```

Any config value can be overridden from the command line:

```bash
python train.py --config configs/ant_bee_transfer.yaml --set train.epochs=5 --set data.batch_size=8
```

Each run writes to `runs/<run_name>/`: `model.keras` (final), `best.keras` (best validation
accuracy), `class_names.json`, `config.yaml`, `history.{json,csv,png}`, and `metrics.json` when the
config has a test split.

## Evaluation, prediction and inspection

```bash
python evaluate.py runs/birds_scratch --data-dir data/100-bird-species/test --output metrics.json
```

```bash
python predict.py runs/ant_bee_transfer path/to/image.jpg --top-k 2
```

```bash
python inspect_model.py runs/ant_bee_scratch
```

Preprocessing is part of the saved model, so `predict.py` takes raw images and always matches
training. Saved models load with `keras.models.load_model(path)` after `import dlresnet`.

## Fixes compared with the notebooks

1. Correct bottleneck widths (64/128/256/512, expansion 4, blocks [3, 4, 6, 3]): 25,557,032
   trainable parameters at 224×224×3 / 1000 classes, instead of 179.8M in the notebook.
2. A single architecture definition instead of duplicated `Block` / `ResNet50` cells.
3. No ReLU before the residual addition; optional zero-initialised last BN per block.
4. Optimizers use `learning_rate=` (the notebook's `SGD(lr=0.001)` silently ran at 0.01).
5. Learning-rate schedules are stateless functions of the training step, so repeated `fit()` calls
   cannot compound the decay.
6. Input shape and number of classes are taken from the dataset, never hard-coded.
7. Preprocessing lives inside the model and class names are saved with it, so inference cannot
   use different preprocessing or labels than training.
8. The test split is evaluated; models are saved in the native `.keras` format.

The legacy `models/*.h5` files come from the notebooks. Both are 252,477,020 bytes, which matches a
21.0M-parameter model plus Adam state, so `ant-bee-25epochs.h5` may actually be a bird-architecture
model. Check it with `git lfs pull` before using it.

## References

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016.
  [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- K. He, X. Zhang, S. Ren, J. Sun. *Identity Mappings in Deep Residual Networks*. ECCV 2016.
  [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
- P. Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. 2017.
  [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- [Keras 3 migration guide](https://keras.io/guides/migrating_to_keras_3/)

## Contributors

- [Dao Duy Manh Ha](https://github.com/R1verrrr) (Leader)
- [Bui Huy Hoang](https://github.com/bhhoang)
- [Doan Dinh Dang](https://github.com/dangdd2003)
- Nguyen Nhat Anh
- Cong Thi Hong Ha
- Nguyen Le Minh Duc
