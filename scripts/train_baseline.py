import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dii.pipeline.datautils import CompositeH5Dataset
from dii.pipeline.transforms import default_pipeline
from dii.models.base import AutoEncoder, BaseEncoder, BaseDecoder


N_WORKERS = 8
BATCH_SIZE = 64


autoencoder = AutoEncoder(BaseEncoder(), BaseDecoder())
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
dev_dataset = CompositeH5Dataset(
    "../data/raw/ion_images.h5", "true", default_pipeline, indices=dev_indices
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)

logger = pl.loggers.WandbLogger(project="deep-ion-image")

trainer = pl.Trainer()

trainer.fit()