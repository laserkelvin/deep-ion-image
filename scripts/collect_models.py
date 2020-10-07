import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from dii.models import base
from dii.utils import load_yaml

betavae_config = load_yaml("betavae.yml")

model = base.VAE(base.BaseEncoder(), base.BaseDecoder(), **betavae_config)
chk = torch.load("deep-ion-image/firto7nh/checkpoints/epoch=8.ckpt", map_location="cpu")
model.load_state_dict(chk["state_dict"])
torch.save(model.state_dict(), "../models/betavae.pt")

unet_config = load_yaml("unet-skim.yml")

model = base.UNetAutoEncoder(**unet_config)
chk = torch.load("deep-ion-image/yirli8lf/checkpoints/epoch=9.ckpt", map_location="cpu")
model.load_state_dict(chk["state_dict"])
torch.save(model.state_dict(), "../models/unet-skim.pt")
