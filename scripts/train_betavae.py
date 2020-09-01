import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader

from dii.pipeline.datautils import CompositeH5Dataset
from dii.models.base import VAE, BaseEncoder, BaseDecoder
from dii.utils import checkpoint_callback, load_yaml

config = load_yaml("betavae.yml")

N_WORKERS = config["n_workers"]
BATCH_SIZE = config["batch_size"]
TRAIN_SEED = np.random.seed(config["train_seed"])
TEST_SEED = np.random.seed(config["test_seed"])

DROPOUT = config["dropout"]

if torch.cuda.is_available():
    GPU = 1
else:
    GPU = 0


vae = VAE(BaseEncoder(dropout=DROPOUT), BaseDecoder(dropout=DROPOUT), **config)

with h5py.File("../data/raw/ion_images.h5", "r") as h5_file:
    train_indices = np.array(h5_file["train"])
    test_indices = np.array(h5_file["test"])

# Load up the datasets; random seed is set for the training set
train_dataset = CompositeH5Dataset(
    "../data/raw/ion_images.h5", "true", indices=train_indices, seed=TRAIN_SEED
)
test_dataset = CompositeH5Dataset(
    "../data/raw/ion_images.h5", "true", indices=test_indices, seed=TEST_SEED
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)

logger = pl.loggers.WandbLogger(name="beta-vae", project="deep-ion-image")
logger.watch(vae, log="all")
logger.log_hyperparams(config)

trainer = pl.Trainer(logger=logger, max_epochs=config["max_epochs"], gpus=GPU, accumulate_grad_batches=config["accumulate_grad_batches"])

trainer.fit(vae, train_loader, test_loader)
