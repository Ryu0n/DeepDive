import json
import torch
import torch.nn.functional as F
from itertools import chain
from encoder import Encoder
from decoder import Decoder
from dataloader import load_dataloader


with open("config.json", "r") as f:
    config = json.load(f)
    
config_trainer = config.get("trainer")
encoder = Encoder()
decoder = Decoder()

# def kl_divergence(z, mu1, sigma1, mu2=0, sigma2=1):
#     """
#     실제 분포 q_{\phi}(z|x)
#     mu1, sigma1
    
#     approximation distribution (가우시안 분포) p(z)
#     mu2, sigma2 : z-표준화 정규 분포 파라미터
#     """
#     q = torch.distributions.Normal(mu1, sigma1)
#     p = torch.distributions.Normal(mu2, sigma2)
    
#     log_qzx = q.log_prob(z)
#     log_qz = p.log_prob(z)
    
#     return log_qzx - log_qz


def train():
    train_dataloader, test_dataloader = load_dataloader(config_trainer.get("batch_size"))
    optimizer = torch.optim.Adam(
        chain(
            *[
                encoder.parameters(),
                decoder.parameters()
            ]
        ),
        lr=config_trainer.get("learning_rate"),
        betas=(config_trainer.get("beta1"), config_trainer.get("beta2"))
    )
    
    for epoch in range(config_trainer.get("n_epochs")):
        
        # Optimization
        for i, (x, _) in enumerate(train_dataloader):
            x = x.view(-1, x.shape[-2]*x.shape[-1])
            z, mu, logvar = encoder(x)
            x_reconstruct = decoder(z)
            
            reconstruct_loss = F.binary_cross_entropy(x_reconstruct, x, reduction="sum")
            kl_divergence = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

            loss = reconstruct_loss + kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Validation
        