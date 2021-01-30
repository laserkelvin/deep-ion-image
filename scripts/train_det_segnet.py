import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader

from dii.pipeline.datautils import CompositeH5Dataset
from dii.models.base import UNetSegAE
from dii.utils import checkpoint_callback, load_yaml

config = load_yaml("unet-seg.yml")

N_WORKERS = config["n_workers"]
BATCH_SIZE = config["batch_size"]
TRAIN_SEED = np.random.seed(config["train_seed"])
TEST_SEED = np.random.seed(config["test_seed"])

# model settings
NUM_SEG = config.get("num_segs", 6)
PRECISION = config.get("precision", 16)

# wandb.init(config=config)

if torch.cuda.is_available():
    GPU = 1
else:
    GPU = 0

model = UNetSegAE(**config)

with h5py.File("../data/raw/ion_images.h5", "r") as h5_file:
    train_indices = np.array(h5_file["dev"])
    test_indices = np.array(h5_file["test"])

# Load up the datasets; random seed is set for the training set
train_dataset = CompositeH5Dataset(
    "../data/raw/ion_images.h5",
    "projection",
    indices=train_indices,
    seed=TRAIN_SEED,
    max_composites=NUM_SEG,
)
test_dataset = CompositeH5Dataset(
    "../data/raw/ion_images.h5",
    "projection",
    indices=test_indices,
    seed=TEST_SEED,
    max_composites=NUM_SEG,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=False
)

logger = pl.loggers.WandbLogger(name="unet-seg", project="deep-ion-image")
logger.watch(model, log="all")
logger.log_hyperparams(config)

trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    gpus=GPU,
    accumulate_grad_batches=config["accumulate_grad_batches"],
    precision=PRECISION,
    num_sanity_val_steps=0,
    logger=logger,
)

trainer.fit(model, train_loader, test_loader)
