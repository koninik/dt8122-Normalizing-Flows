import torch
import torch.nn as nn
from realnvp import RealNVP
import numpy as np
from torch.utils.data import DataLoader
from datasets import datasets_dict
from utils import subfig_plot
from torch.distributions import MultivariateNormal
from sklearn import datasets
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def train(model, train_loader, optimizer, epochs, dataset_name, device):
    print('Training...')
    model = model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        for i, x in enumerate(tqdm(train_loader)):
            x = x.to(device)
            loss = -model.log_prob(x).mean()
            #loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('iter %s:' % i, 'loss = %.3f' % loss)
            loss_sum += loss.detach().cpu().item()
        print('Epoch: {}/{}, Loss: {:.3f}'.format(epoch+1, epochs, loss_sum/len(train_loader)))
        torch.save(model.state_dict(), f"./models/real_nvp_{dataset_name}.pt")
    


def inference(model, test_loader, dataset):
    model.eval()
    test_loss = 0
    x_all = np.array([[]]).reshape(0,2)
    z_all = np.array([[]]).reshape(0,2)
    with torch.no_grad():
        for i, x in enumerate(tqdm(test_loader)):
            x = x.to(device)
            z, _ = model.f(x)
            
            x_all = np.concatenate((x_all,x.numpy()))
            z_all = np.concatenate((z_all,z.numpy()))
        
        subfig_plot(3, x_all, -1.5, 2.5, -1, 1.5,'X ~ p(x) (dataset)', 'RealNVP', 'r', f'real_nvp_{dataset}', dataset)
        subfig_plot(1, z_all, -3, 3, -3,3,'z = f(x) (Inverse transform)', 'RealNVP', 'b', f'real_nvp_{dataset}', dataset)
        
def sample(model, prior_z, dataset):
    "sampling"
    model.eval()
    
    with torch.no_grad():
        z, x = model.sample(1000)
        
        z = z.numpy()
        x = x.numpy()
        
        subfig_plot(2, z, -3, 3, -3, 3, 'z ~ p(z) (Samples from prior)', 'RealNVP', 'b', f'real_nvp_{dataset}', dataset) #OK
        subfig_plot(4, x, -1.5, 2.5, -1, 1.5,' X = g(z) (forward transform)', 'RealNVP', 'r', f'real_nvp_{dataset}', dataset)
        plt.close()        
        
    
if __name__=="__main__":
    
    '''Main function'''
    parser = argparse.ArgumentParser(description='Planar Flows')
    parser.add_argument('--dataset_idx', type=int, help='0: two_moons, 1: two_blobs, 2: boomerang', default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--K', type=int, help='Number of planar flows: 2, 8, 32', default=32)
    parser.add_argument('--train_flag', type=bool, default=True)
    
    args = parser.parse_args()
    epochs = 300
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    
    #nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())
    #nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2))
    
    nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2), nn.Tanh())
    nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2))
    
    
    masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    model = RealNVP(nets, nett, masks, prior)
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if args.dataset_idx == 0:
        dataset_name = 'Two_Moons'
    elif args.dataset_idx == 1:
        dataset_name = 'Two_Blobs'
    else:
        dataset_name = 'Boomerang'
        
    csv_file = datasets_dict[args.dataset_idx]
    train_data = np.loadtxt(open(csv_file, "rb"), dtype=np.float32, delimiter=",", skiprows=1)
    test_data = np.loadtxt(open(csv_file, "rb"), dtype=np.float32, delimiter=",", skiprows=1)
    
    #noisy_moons = datasets.make_blobs(n_samples=1000, noise=.05)[0].astype(np.float32)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad==True], lr=1e-4)
    
    if args.train_flag == True:
        train(model, train_loader, optimizer, epochs, dataset_name, device)
    else:
        model_path = f'./models/real_nvp_{dataset_name}.pt'
        model.load_state_dict(torch.load(model_path))
        inference(model, test_loader, dataset_name)
        sample(model, prior, dataset_name)
        
    


    