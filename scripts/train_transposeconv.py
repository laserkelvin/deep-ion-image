import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader

from dii.pipeline.datautils import CompositeH5Dataset
from dii.pipeline.transforms import default_pipeline
from dii.models.base import AutoEncoder, BaseEncoder, TransposeDecoder
from dii.utils import checkpoint_callback


N_WORKERS = 8
BATCH_SIZE = 24

if torch.cuda.is_available():
    GPU = 1
else:
    GPU = 0


autoencoder = AutoEncoder(BaseEncoder(), TransposeDecoder())

with h5py.File("../data/raw/ion_images.h5", "r") as h5_file:
    train_indices = np.array(h5_file["train"])
    test_indices = np.array(h5_file["test"])
    dev_indices = np.array(h5_file["dev"])

# Load up the datasets
train_dataset = CompositeH5Dataset(
    "../data/raw/ion_images.h5", "true", default_pipeline, indices=train_indices
)
test_dataset = CompositeH5Dataset(
    "../data/raw/ion_images.h5", "true", default_pipeline, indices=test_indices
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)

logger = pl.loggers.WandbLogger(name="baseline-trans", project="deep-ion-image")
logger.watch(autoencoder, log="all")

trainer = pl.Trainer(logger=logger, max_epochs=30, gpus=GPU, accumulate_grad_batches=4)

trainer.fit(autoencoder, train_loader, test_loader)
