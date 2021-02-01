import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import argparse
from torch.utils.data import DataLoader

from dii.pipeline.datautils import CompositeH5Dataset
from dii.models.base import AutoEncoder, Encoder, Decoder
from dii.utils import load_yaml

config = load_yaml("baseline.yml")

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--in_channels', type=int, default=1, metavar='N',
                    help='Number of input channels (default: 1)')
parser.add_argument('--out_channels', type=int, default=1, metavar='N',
                    help='Number of otuput channels (default: 1)')
parser.add_argument('--latent_dim', type=int, default=128, metavar='N',
                    help='Dimensionality of the latent vector (default: 1)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--n_workers', type=int, default=8, metavar='N',
                    help='Number of CPUs for dataloading (default: 9)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--train_seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test_seed', type=int, default=1923, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--grad_clip', type=float, default=0., metavar='GC',
                    help='Gradient clipping value (default: 0)')
parser.add_argument('--activation', type=str, default="relu", metavar='GC',
                    help='Name of activation function (default: relu)')
parser.add_argument('--half', type=bool, default=False, metavar='GC',
                    help='Flag for using half precision (default: False)')
parser.add_argument('--dropout', type=float, default=0., metavar='GC',
                    help='Dropout probability (default: 0)')
parser.add_argument('--weight_decay', type=float, default=0., metavar='GC',
                    help='L2 regularization (default: 0)')
args = parser.parse_args()

if args.half:
    PRECISION = 16
else:
    PRECISION = 32

wandb.init(config=config)
wandb.config.update(args)

if torch.cuda.is_available():
    GPU = 1
else:
    GPU = 0


model = AutoEncoder(**vars(args))

with h5py.File("../data/raw/ion_images.h5", "r") as h5_file:
    train_indices = np.array(h5_file["train"])
    test_indices = np.array(h5_file["test"])

# Load up the datasets; random seed is set for the training set
train_dataset = CompositeH5Dataset(
    "../data/raw/128_ion_images.h5", "projection", indices=train_indices, seed=args.train_seed
)
test_dataset = CompositeH5Dataset(
    "../data/raw/128_ion_images.h5", "projection", indices=test_indices, seed=args.test_seed
)

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=False
)

logger = pl.loggers.WandbLogger(name="baseline", project="deep-ion-image")
logger.watch(model, log="all")
logger.log_hyperparams(config)

trainer = pl.Trainer(
    max_epochs=args.epochs,
    gpus=GPU,
    accumulate_grad_batches=config["accumulate_grad_batches"],
    gradient_clip_val=args.grad_clip,
    precision=PRECISION,
    logger=logger,
)

trainer.fit(model, train_loader, test_loader)
