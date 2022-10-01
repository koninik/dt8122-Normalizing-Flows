import torch
import numpy as np
from datasets import datasets_dict
from cnf import CNF
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
from datasets import datasets_dict
import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *

def get_batch(num_samples, dataset_idx):
    if dataset_idx is None:
        points,_ = make_moons(n_samples=num_samples, noise=0.06)
    else:
        csv_file = datasets_dict[dataset_idx]
        points = np.loadtxt(open(csv_file, "rb"), dtype=np.float32, delimiter=",", skiprows=1)
    
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return(x, logp_diff_t1)

def train(func, optimizer, dataset_idx, device, args):
    try:
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()

            x, logp_diff_t1 = get_batch(args.num_samples, args.dataset_idx)

            z_t, logp_diff_t = odeint(
                func,
                (x, logp_diff_t1),
                torch.tensor([t1, t0]).type(torch.float32).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

            logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
            loss = -logp_x.mean(0)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())

            print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))
        

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))


def visualize(t0, t1, viz_samples, viz_timesteps, target_sample, dataset_name):
    with torch.no_grad():
        # Generate evolution of samples
        z_t0 = p_z0.sample([viz_samples]).to(device)
        logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

        z_t_samples, _ = odeint(
            func,
            (z_t0, logp_diff_t0),
            torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )

        # Generate evolution of density - This should be adjusted according to dataset
        x = np.linspace(-1.5, 2.5, 100)
        y = np.linspace(-1.5, 2.5, 100)
        #x = np.linspace(-5, 15, 100)
        #y = np.linspace(-10, 10, 100)
        points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

        z_t1 = torch.tensor(points).type(torch.float32).to(device)
        logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

        z_t_density, logp_diff_t = odeint(
            func,
            (z_t1, logp_diff_t1),
            torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )

        # Create plots for each timestep
        for (t, z_sample, z_density, logp_diff) in zip(
                np.linspace(t0, t1, viz_timesteps),
                z_t_samples, z_t_density, logp_diff_t
        ):
            
            #Target Distribution
            
            plt.subplot(2, 2, 3)
            plt.scatter(*target_sample.detach().cpu().numpy().T, color = 'r', s=1)
            plt.title('X ~ p(x) (dataset)')
            if args.dataset_idx == 2:
                plt.xlim(-5, 50)
                plt.ylim(-15,15)
            else:
                plt.xlim(-1.5, 2.5)
                plt.ylim(-1,1.5)
            plt.tight_layout()
            
            
            if t==0:
                z_prior = z_sample.detach().cpu().numpy().T
            
            plt.subplot(2, 2, 2)
            plt.scatter(*z_prior, color = 'b', s=1)
            plt.title('z ~ p(z) (Samples from prior)')
            #if args.dataset_idx == 2:
             #   plt.xlim(-5, 50)
              #  plt.ylim(-15,15)
            #else:
            plt.xlim(-1.5, 2.5)
            plt.ylim(-1,1.5)
            plt.tight_layout()
            
            
            logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
            plt.subplot(2, 2, 4)
            plt.scatter(*z_sample.detach().cpu().numpy().T, color = 'r', s=1)
            plt.title('X = g(z) (forward transform)')
            if args.dataset_idx == 2:
                plt.xlim(-5, 50)
                plt.ylim(-15,15)
            else:
                plt.xlim(-1.5, 2.5)
                plt.ylim(-1,1.5)
            plt.tight_layout()
    
            plt.savefig(f"{dataset_name}-cnf-viz-{int(t*1000):05d}.jpg")
            
            
            plt.close()
            
            if args.gif == True:
                #Plot from Normal to Data Distribution to create the GIF
                plt.tricontourf(*z_t1.detach().cpu().numpy().T, np.exp(logp.detach().cpu().numpy()), 200)
                plt.savefig(f'./results/{dataset_name}_cnf_{int(t*1000):05d}.png')
                
        if args.gif == True:
            img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join('./results', f"{dataset_name}_cnf_*.png")))]
            img.save(fp=os.path.join('./results', f"cnf-viz-{dataset_name}.gif"), format='GIF', append_images=imgs,
                        save_all=True, duration=250, loop=0)

        print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "cnf-viz-{dataset_name}.gif")))


def inference(t0, t1, dataset_idx, dataset_name, viz_timesteps, target_sample, device, args):
    with torch.no_grad():
        #for itr in range(1, 500):
        x, logp_diff_t1 = get_batch(args.num_samples, args.dataset_idx)

        z_t, logp_diff_t = odeint(
            func,
            (x, logp_diff_t1),
            torch.tensor([t1, t0]).type(torch.float32).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )

        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

        logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
        
        
        # Generate evolution of density - This should be adjusted according to dataset
        x = np.linspace(-1.5, 2.5, 100)
        y = np.linspace(-1.5, 2.5, 100)
        #x = np.linspace(-5, 15, 100)
        #y = np.linspace(-10, 10, 100)
        points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

        z_t1 = torch.tensor(points).type(torch.float32).to(device)
        logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

        z_t_density, logp_diff_t = odeint(
            func,
            (z_t1, logp_diff_t1),
            torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )
        
        for (t, z_sample, z_density, logp_diff) in zip(
                np.linspace(t0, t1, viz_timesteps),
                z_t, z_t_density, logp_diff_t
        ):
            #print(z_sample)
            #if t==0.5:
            #plt.subplot(2, 1, 1)
            plt.scatter(*z_sample.detach().cpu().numpy().T,  color = 'b', s=1)
            plt.title('z = f(x) (Inverse transform)')
            if dataset_idx == 2:
                plt.xlim(-5, 50)
                plt.ylim(-15,15)
            else:
                plt.xlim(-1.5, 2.5)
                plt.ylim(-1,1.5)
            plt.tight_layout()
            
            
            plt.savefig(f"CNF_{dataset_name}_{t}.jpg")
        plt.close()
        
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--viz', type=bool, default=True)
    parser.add_argument('--gif', type=bool, default=True)
    parser.add_argument('--train_flag', type=bool, default=True)
    parser.add_argument('--inference', type=bool, default=True)
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train_dir', type=str, default='./models/')
    parser.add_argument('--results_dir', type=str, default="./results")
    parser.add_argument('--dataset_idx', type=int, help='0: two_moons, 1: two_blobs, 2: boomerang', default=2)
    args = parser.parse_args()

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
    
    t0 = 0.0
    t1 = 10.0
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    # Load datasets 
    csv_file = datasets_dict[args.dataset_idx]
    data = np.loadtxt(open(csv_file, "rb"), dtype=np.float32, delimiter=",", skiprows=1)
    if args.dataset_idx == 0:
        dataset_name = 'Two_Moons'
    elif args.dataset_idx == 1:
        dataset_name = 'Two_Blobs'
    else:
        dataset_name = 'Boomerang'
    
    # model
    func = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width).to(device)
    
    optimizer = optim.Adam(func.parameters(), lr=args.lr)
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device)
    )
    loss_meter = RunningAverageMeter()
    
    if args.train_flag:
        train(func, optimizer, args.dataset_idx, device, args)
    
    if args.inference:
        viz_samples = 1000
        viz_timesteps = 41
        target_sample, _ = get_batch(viz_samples, args.dataset_idx)
        inference(t0, t1, args.dataset_idx, dataset_name, viz_timesteps, target_sample, device, args)
        
    
    if args.viz:
        viz_samples = 1000
        viz_timesteps = 41
        target_sample, _ = get_batch(viz_samples, args.dataset_idx)

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        visualize(t0, t1, viz_samples, viz_timesteps, target_sample, dataset_name)
    