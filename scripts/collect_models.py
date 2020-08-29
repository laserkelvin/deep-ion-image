import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from dii.models import base

#model = AutoEncoder(BaseEncoder(), BaseDecoder())
#chk = torch.load("deep-ion-image/2z1gqfo5/checkpoints/epoch=10.ckpt", map_location="cpu")
#model.load_state_dict(chk["state_dict"])
#torch.save(model.state_dict(), "../models/upsample_baseline.pt")

#model = AutoEncoder(BaseEncoder(), TransposeDecoder())
#chk = torch.load("deep-ion-image/3mroe9qe/checkpoints/epoch=6.ckpt", map_location="cpu")
#model.load_state_dict(chk["state_dict"])
#torch.save(model.state_dict(), "../models/transconv_baseline.pt")
#
#model = VAE(BaseEncoder(), TransposeDecoder())
#chk = torch.load("deep-ion-image/2wgnmlwu/checkpoints/epoch=9.ckpt", map_location="cpu")
#model.load_state_dict(chk["state_dict"])
#torch.save(model.state_dict(), "../models/betavae.pt")

model = base.UNetAutoEncoder()
chk = torch.load("deep-ion-image/ne6bq38e/checkpoints/epoch=10.ckpt", map_location="cpu")
model.load_state_dict(chk["state_dict"])
torch.save(model.state_dict(), "../models/unet-skim.pt")
