import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from dii.models import base

model = base.VAE(base.BaseEncoder(), base.BaseDecoder())
chk = torch.load("deep-ion-image/2qqreezg/checkpoints/epoch=8.ckpt", map_location="cpu")
model.load_state_dict(chk["state_dict"])
torch.save(model.state_dict(), "../models/betavae.pt")

model = base.UNetAutoEncoder()
chk = torch.load("deep-ion-image/yirli8lf/checkpoints/epoch=9.ckpt", map_location="cpu")
model.load_state_dict(chk["state_dict"])
torch.save(model.state_dict(), "../models/unet-skim.pt")
