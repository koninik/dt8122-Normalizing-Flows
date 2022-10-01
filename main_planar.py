from importlib.metadata import distribution
import torch
import torch.nn
from planar_flows import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from torch.distributions import MultivariateNormal, Distribution
from sklearn import datasets
from datasets import datasets_dict
import argparse
from utils import subfig_plot



def train(model, training_data, prior_z, optimizer, epochs, batch_num, device, args):    
    print('Training')
    loss_sum = 0
    for epoch in range(epochs):
        model.train()
        for batch_idx, data in enumerate(tqdm(training_data)):
            
            samples = torch.zeros(size=(128, 2)).normal_(mean=0, std=1)
            data = data.to(device)
            z, logdet_J = model(samples)
            
            loss = - torch.mean(prior_z.log_prob(z) + logdet_J)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().cpu().item()
            #if i % 1000 == 0:
             #   print(f"(batch_num {i:05d}/{batch_num}) loss: {loss}")
        print('Epoch: {}/{}, Loss: {:.3f}'.format(epoch+1, epochs, loss_sum/len(train_loader)))
        torch.save(model.state_dict(), f"./models/model_planar.pt")
            

def inference(model, test_loader, prior_z, epochs, dataset):
    model.eval()
    test_loss_sum = 0
    x_all = np.array([[]]).reshape(0,2)
    z_all = np.array([[]]).reshape(0,2)
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            samples = torch.zeros(size=(128, 2)).normal_(mean=0, std=1)
            z, logdet_J = model(data)
            
            x_all = np.concatenate((x_all,data.numpy()))
            z_all = np.concatenate((z_all,z.numpy()))
        
        
        if args.dataset_idx == 2:
            subfig_plot(3, x_all, -5, 50, -15, 15,'X ~ p(x) (dataset)', 'Planar','r', f'planar_{dataset}', dataset)   #OK
        else:
            subfig_plot(3, x_all, -1.5, 2.5, -1, 1.5,'X ~ p(x) (dataset)', 'Planar','r', f'planar_{dataset}', dataset)   #OK
        subfig_plot(1, z_all, -3, 3, -3,3,'z = f(x) (Inverse transform)', 'Planar','b', f'planar_{dataset}', dataset)
        
def sample(model, prior_z, dataset):
    x_all = np.array([[]]).reshape(0,2)
    z_all = np.array([[]]).reshape(0,2)
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            z = prior_z.sample((1000,))
            x, p  = model(data)
            #print(x)
            z = z.numpy()
            x = x.numpy()
        
        if args.dataset_idx == 2:
            subfig_plot(4, x, -5, 50, -15, 15,'X = g(z) (forward transform)', 'Planar','r', f'planar_{dataset}', dataset)   #OK
        else:
            subfig_plot(4, x, -1.5, 2.5, -1, 1.5,' X = g(z) (forward transform)', 'Planar', 'r', f'planar_{dataset}', dataset)
        subfig_plot(2, z, -3, 3, -3, 3, 'z ~ p(z) (Samples from prior)', 'Planar', 'b', f'planar_{dataset}', dataset) #OK
        
        #subfig_plot(1, x, -3, 3, -3,3,'z = f(x) (Inverse transform)', 'Planar','b', f'planar_{dataset}', dataset)


    
    
if __name__== "__main__":
    
    '''Main function'''
    parser = argparse.ArgumentParser(description='Planar Flows')
    parser.add_argument('--dataset_idx', type=int, help='0: two_moons, 1: two_blobs, 2: boomerang', default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--K', type=int, help='Number of planar flows: 2, 8, 32', default=32)
    
    args = parser.parse_args()
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    if args.dataset_idx == 0:
        dataset_name = 'Two_Moons'
    elif args.dataset_idx == 1:
        dataset_name = 'Two_Blobs'
    else:
        dataset_name = 'Boomerang'
        
    csv_file = datasets_dict[args.dataset_idx]
    train_data = np.loadtxt(open(csv_file, "rb"), dtype=np.float32, delimiter=",", skiprows=1)
    test_data = np.loadtxt(open(csv_file, "rb"), dtype=np.float32, delimiter=",", skiprows=1)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    batch_num = 500
    epochs = 100
    prior_z = MultivariateNormal(torch.zeros(2), torch.eye(2))
    model = PlanarFlowModel(prior_z, dim = 2, K = args.K)
    
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3) # or Adagrad
    train(model, test_loader, prior_z, optimizer, epochs, batch_num, device, args)
    model_path = './models/model_planar.pt'
    model.load_state_dict(torch.load(model_path))
    
    inference(model, test_loader, prior_z, epochs, dataset_name)
    sample(model, prior_z, dataset_name)