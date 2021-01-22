import torch
from torchsummary import summary
from dii.models.base import VAE, BaseEncoder, BaseDecoder

img_size = 500

encoder = BaseEncoder()
decoder = BaseDecoder()
vae = VAE(encoder, decoder, encoding_imgsize=2, encoding_filters=256, latent_dim=128).cuda()

#summary(encoder, (1, img_size, img_size), device="cuda")
#summary(decoder, (128, 9, 9), device="cuda")

summary(vae, (1, img_size, img_size), batch_size=48, device="cuda")

data = torch.rand(48, 1, img_size, img_size, device="cuda")

print(vae(data).shape)
