import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from tqdm.notebook import tqdm
from IPython.display import clear_output

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###########################
#### HELPER FUNCTIONS #####
###########################

def show_images(pred, real, log, num_images=25, size=(1, 28, 28)):
    pred_unf = pred.detach().cpu().view(-1, *size)
    real_unf = real.detach().cpu().view(-1, *size)
    pred_grid = make_grid(pred_unf[:num_images], nrow=5)
    real_grid = make_grid(real_unf[:num_images], nrow=5)
    fig = plt.figure()
    ax1, ax2 = fig.subplots(1, 2)
    plt.title(log)
    ax1.imshow(pred_grid.permute(1, 2, 0).squeeze())
    ax2.imshow(real_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
##################################
###### LinearBnReLU block ########
##################################

class LinearBnReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block=nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.block(x)
    
##################################
###### LinearLeakyReLU block #####
##################################

class LinearLeakyReLU(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.block=nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(alpha))
        
    def forward(self, x):
        return self.block(x)
    
##################################
########### GENERATOR ############
##################################
    
class Generator(nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            LinearBnReLU(in_features, hid_features),
            LinearBnReLU(hid_features, 2*hid_features),
            LinearBnReLU(2*hid_features, 4*hid_features),
            LinearBnReLU(4*hid_features, 8*hid_features),
            nn.Linear(8*hid_features, out_features),
            nn.Sigmoid())
        
    def forward(self, x):
        return self.model(x)
    
##################################
######### DISCRIMINATOR ##########
##################################

class Discriminator(nn.Module):
    def __init__(self, in_features, hid_features):
        super().__init__()
        self.model = nn.Sequential(
            LinearLeakyReLU(in_features, 4*hid_features),
            LinearLeakyReLU(4*hid_features, 2*hid_features),
            LinearLeakyReLU(2*hid_features, hid_features),
            nn.Linear(hid_features, 1))
        
    def forward(self, x):
        return self.model(x)
    
##################################
############ TRAINER #############
##################################

class Trainer():
    def __init__(self, Generator, Discriminator, G_optimizer, D_optimizer,
                 criterion, device=device):
        self.G = Generator.to(device)
        self.D = Discriminator.to(device)
        self.G_optim = G_optimizer
        self.D_optim = D_optimizer
        
        self.criterion=criterion.to(device)
        self.results={'G_loss':[], 'D_loss':[]}
        
    def fit(self, generator, epochs=30, device=device):
        for epoch in range(1, epochs+1):
            G_losses=[]
            D_losses=[]

            log = f'::::: Epoch {epoch}/{epochs} :::::'
            for real, _ in tqdm(generator):
                real = real.view(real.size(0), -1).to(device)
                # DISCRIMINATOR
                self.D_optim.zero_grad()
                # DISCRIMINATOR`s LOSS
                noise = torch.randn(real.size(0), 64).to(device)
                # Prediction on FAKE image
                fake = self.G(noise).detach()
                fake_pred = self.D(fake)
                fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
                # Prediction on REAL image
                real_pred = self.D(real)
                real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
                D_loss = (fake_loss + real_loss)/2

                D_losses.append(D_loss.item())
                D_loss.backward(retain_graph=True)
                self.D_optim.step()
                # GENERATOR
                self.G_optim.zero_grad()
                # GENERATOR`s LOSS
                fake = self.G(noise)
                fake_pred = self.D(fake)
                G_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))

                G_losses.append(G_loss.item())
                G_loss.backward()
                self.G_optim.step()
                template = f'::: Generator Loss: {G_loss.item():.3f} | Discriminator Loss: {D_loss.item():.3f} :::'

            self.results['G_loss'].append(np.mean(G_losses))
            self.results['D_loss'].append(np.mean(D_losses))
            clear_output(wait=True)
            show_images(fake, real, log+template)