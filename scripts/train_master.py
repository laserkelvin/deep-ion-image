import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import argparse
from dii.models import base

parser = argparse.ArgumentParser(description='Deep ion image training script')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--grad_clip', type=float, default=0., metavar='GC',
                    help='Gradient clipping value (default: 0)')
parser.add_argument('--half', type=bool, default=False, metavar='GC',
                    help='Flag for using half precision (default: False)')
parser.add_argument('--cpu', type=bool, default=False, help="Force CPU model (default: False")
parser.add_argument('--model', type=str, default="baseline", help="Model specification, refer to base.py")
parser.add_argument('--sweep', type=bool, default=False, help="If using wandb for a sweep (default: False")
# this grabs the model choice without running parse_args
temp_args, _ = parser.parse_known_args()
model_choice = base.valid_models.get(temp_args.model)

# grab the necessary hyperparameters for a given model
parser = model_choice.add_model_specific_args(parser)
args = parser.parse_args()

if args.half:
    PRECISION = 16
else:
    PRECISION = 32

# Defaults to GPU use, unless otherwise forced with the cpu flag
if not torch.cuda.is_available() or args.cpu:
    GPU = 0
else:
    GPU = 1


# load the indices
with h5py.File(args.h5_path, "r") as h5_file:
    train_indices = np.array(h5_file["train"])
    test_indices = np.array(h5_file["test"])

if args.sweep:
    wandb.init(entity="team-brazil", config=args)

model = model_choice(train_indices=train_indices, test_indices=test_indices, **vars(args))

logger = pl.loggers.WandbLogger(project="deep-ion-image", entity="team-brazil")
logger.watch(model, log="all")
logger.log_hyperparams(vars(args))

trainer = pl.Trainer(
    max_epochs=args.epochs,
    gpus=GPU,
    gradient_clip_val=args.grad_clip,
    precision=PRECISION,
    logger=logger,
)
trainer.fit(model)

# save the final weights
torch.save(model.cpu().state_dict(), f"../models/{args.model}.pt")
with open(f"../models/{args.model}.md", "w+") as write_file:
    write_file.write(logger.version)
