# Amazon Deforestation Detection

A satellite-image classification project that tests how pretraining, dataset size, and data augmentation change ResNet18 performance on Amazon rainforest photos.

The data is from Kaggle's [Planet: Understanding the Amazon from Space](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space) competition: **40,479** images, each 256×256 RGB. One image can have several labels at once. There are **17** labels for land use, land cover, and weather, including primary forest, agriculture, bare ground, water, and cloud cover.

Built with **Python, PyTorch, torchvision, pandas, scikit-learn, and Matplotlib** for the Data Science Distinction at The Browning School. Models were trained on an NVIDIA T4 GPU.

## Experiments

Four training setups, all ResNet18, all trained for **10 epochs** with the Adam optimizer and a starting learning rate of **1e-4**.

| Notebook | Starting weights | Training data | Image changes | Batch size |
| --- | --- | --- | --- | ---: |
| [`barebones_fraction.ipynb`](notebooks/barebones_fraction.ipynb) | Random | About one-third of the images | None | 128 |
| [`pretrained_fraction.ipynb`](notebooks/pretrained_fraction.ipynb) | ImageNet | About one-third of the images | None | 64 |
| [`pretrained_all_data.ipynb`](notebooks/pretrained_all_data.ipynb) | ImageNet | Full training set | None | 64 |
| [`augmentations_fraction.ipynb`](notebooks/augmentations_fraction.ipynb) | Random | About one-third of the images | Flips, rotation, shift, zoom, and color changes | 128 |

During training, the notebooks score with **F2**, which gives more credit for finding real labels than for avoiding false alarms. The charts use **F1**, which treats those two equally.

## Results

The ImageNet model trained on the full set had the best F1 of the four runs: **0.801** on primary forest, the most common label, and **0.342** on bare ground, a rare one. The other three models were close on primary forest (about 0.78) and lower on bare ground (about 0.29–0.31).

The full-data run and the smaller runs did not use the same validation set, so these numbers describe each run. They are not a clean test of "more data" by itself. Rare labels also show why one overall score is not enough: a model can look strong on common forest photos and still miss the uncommon ones.

[`analysis/plot_results.py`](analysis/plot_results.py) draws the charts from these recorded scores. The spreadsheet is [`results/distinction_sets.xlsx`](results/distinction_sets.xlsx). Making the charts again does not retrain the models.

## Run the notebooks

1. Open a notebook in `notebooks/` in **Google Colab**.
2. Turn on a **GPU**.
3. Run the cells from top to bottom. The notebook will ask to connect Google Drive and will download the images.
4. Check the data paths and the save location before training starts.

The satellite images are not in this repo. The notebooks download them from Google Drive and save checkpoints to `/content/drive/MyDrive/`. You need access to those Drive files to train.

Colab already has most of the libraries. Anything else is listed in [`requirements.txt`](requirements.txt).

## Make the charts on your computer

From the project folder:

```bash
python -m pip install -r requirements.txt
python analysis/plot_results.py
```

This writes:

- `results/model_comparison.png`
- `results/confusion_shares.png`

## What is in the repo

```text
notebooks/          Four training notebooks
analysis/           Chart script
results/            Score spreadsheet and charts
requirements.txt    Python libraries
```

## What this project does not do

The models label what is in a single satellite photo. They do not measure forest loss over time, and they do not decide whether an activity is illegal. Most images are ordinary forest, the photos are only from the Amazon, and the four runs did not all share one validation set.

## Credits

Project by **Jonathan Guo**.

The notebooks are adapted from course material by **Indu Panigrahi and Carlos Jimenez** and from [Vincent Maladière's ResNet18 notebook](https://www.kaggle.com/code/vincentmaladiere/pytorch-resnet18-0-93).
