#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from scipy.stats import norm
import sys
sys.path.insert(0,'..')
import importlib
import algorithms.EarlyStopping as ES
import torch.utils.data as data

# ---
# ====================== Training functions ====================== #
# ---
class AE_Algorithm():
    def __init__(self, opt):
        # Save the whole config
        self.opt = opt

        # Training parameters
        self.batch_size = opt['batch_size']
        self.epochs = opt['epochs']
        self.current_epoch = None
        self.loss_fn = opt['loss_fn']
        self.snapshot = opt['snapshot']
        self.console_print = opt['console_print']
        self.lr_schedule = opt['lr_schedule']
        self.init_lr_schedule = opt['lr_schedule']
        self.model = None
        self.vae_optimiser = None

        # Beta scheduling
        self.beta = opt['beta_min']
        self.beta_range = opt['beta_max'] - opt['beta_min'] + 1
        self.beta_steps = opt['beta_steps'] - 1
        self.beta_idx = 0

        # Gamma scheduling
        self.gamma_warmup = opt['gamma_warmup']
        self.gamma = 0 if self.gamma_warmup > 0 else opt['gamma_min']
        self.gamma_min = opt['gamma_min']
        self.gamma_idx = 0
        self.gamma_update_step = (opt['gamma_max'] - opt['gamma_min']) / opt['gamma_steps']
        self.gamma_update_epoch_step = (self.epochs - self.gamma_warmup - 1) / opt['gamma_steps']

        # Action loss parameters
        self.min_dist = opt['min_dist_samples']
        self.weight_dist_loss = opt['weight_dist_loss']
        self.distance_type = opt['distance_type'] if 'distance_type' in opt.keys() else '2'
        self.batch_dist_dict = {}
        self.epoch_dist_dict = {}
        self.min_epochs = opt['min_epochs'] if 'min_epochs' in opt.keys() else 499
        self.max_epochs = opt['max_epochs'] if 'max_epochs' in opt.keys() else 499

        # Other parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt['device'] = self.device
        print(' *- Chosen device: ', self.device)
        
        torch.manual_seed(opt['random_seed'])
        np.random.seed(opt['random_seed'])
        print(' *- Chosen random seed: ', self.device)
        
        if self.device == 'cuda': torch.cuda.manual_seed(opt['random_seed'])
        self.save_path = self.opt['exp_dir'] + '/' + self.opt['filename']
        self.model_path = self.save_path + '_model.pt'


    def count_parameters(self):
        """
        Counts the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)



    def plot_grid(self, images, n=5,name="dec"):
        """
        Plots an nxn grid of images of size digit_size. Used to monitor the 
        reconstruction of decoded images.
        """
        digit_size = int(np.sqrt(self.opt['input_dim']/self.opt['input_channels']))
        filename = self.save_path +name + '_checkpointRecon_{0}'.format(self.current_epoch)
        figure = np.zeros((digit_size * n, digit_size * n, self.opt['input_channels']))

        # Construct grid of latent variable values
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        # decode for each square in the grid
        counter = 0
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                digit = images[counter].permute(1,2,0).detach().cpu().numpy()
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
                counter += 1

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='bone')
        plt.savefig(filename)
        plt.clf()
        plt.close()



    def latent_mean_dist(self, mean1, mean2, action,
                         distance_type='1'):
        """
        Computed the average d distance between the action and no action pairs
        in the given batch.
        """
        sample1 = mean1
        sample2 = mean2
        dist = torch.norm(sample1 - sample2, p=float(self.distance_type), dim=1) # Batch size
        dist=dist**2
        # Distances between pairs with an action
        dist_action = torch.mul(dist, action) # Batch size
        dist_action = dist_action[dist_action.nonzero()] # num_action, 1
        dist_action_mean = torch.mean(dist_action)
        dist_action_std = torch.std(dist_action)

        # Distances between pairs without an action
        dist_no_action = torch.mul(dist, (1-action))
        dist_no_action = dist_no_action[dist_no_action.nonzero()]
        dist_no_action_mean = torch.mean(dist_no_action)
        dist_no_action_std = torch.std(dist_no_action)
        
        # Compute the action loss
        zeros = torch.zeros(dist.size()).to(self.device)
        batch_dist = (1 - action) * dist + action * torch.max(zeros, self.min_dist - dist)
        dist_loss = torch.mean(batch_dist)

        # Weight the action loss
        batch_loss = self.weight_dist_loss*self.gamma * batch_dist
        avg_batch_loss = torch.mean(batch_loss)

        # Save the result in order to compute average epoch distance
        # If training, save to training distances
        if self.model.training:
            self.epoch_action_dist.append(dist_action.cpu().detach().numpy())
            self.epoch_noaction_dist.append(dist_no_action.cpu().detach().numpy())
            
        # If evaluation, save to test distances
        else:
            self.test_action_dist.append(dist_action.cpu().detach().numpy())
            self.test_noaction_dist.append(dist_no_action.cpu().detach().numpy())

        return (avg_batch_loss, dist_loss, dist_action_mean, dist_action_std, 
                dist_no_action_mean, dist_no_action_std)


    def compute_loss(self, x, dec_mu, enc_mu):
        """
        Computes the usual VAE loss on the training batch given the criterion.
        """
        # Reconstruction loss
        loss = torch.nn.MSELoss(reduction = 'sum')
        batch_rec=loss(x,dec_mu)
       
        return batch_rec 


 

    def format_loss(self, losses_list):
        """Rounds the loss and returns an np array for logging."""
        reformatted = list(map(lambda x: round(x.item(), 2), losses_list))
        reformatted.append(int(self.current_epoch))
        return np.array(reformatted)


    def init_model(self):
        """Initialises the VAE model."""
        vae = importlib.import_module("architectures.{0}".format(self.opt['model']))
        print(' *- Imported module: ', vae)
        try:
            class_ = getattr(vae, self.opt['model'])
            instance = class_(self.opt).to(self.device)
            return instance
        except:
            raise NotImplementedError(
                    'Model {0} not recognized'.format(self.opt['model']))


    def init_optimiser(self):
        """Initialises the optimiser."""
        print(self.model.parameters())
        if self.opt['optim_type'] == 'Adam':
            print(' *- Initialised Adam optimiser.')
            vae_optim = optim.Adam(self.model.parameters(), lr=self.lr)
            return vae_optim
        else:
            raise NotImplementedError(
                    'Optimiser {0} not recognized'.format(self.opt['optim_type']))


    def update_learning_rate(self, optimiser):
        """Annealing schedule for the learning rate."""
        if self.current_epoch == self.lr_update_epoch:
            for param_group in optimiser.param_groups:
                self.lr = self.new_lr
                param_group['lr'] = self.lr
                print(' *- Learning rate updated - new value:', self.lr)
                try:
                    self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
                except:
                    print(' *- Reached the end of the update schedule.')
                print(' *- Remaning lr schedule:', self.lr_schedule)


 

    def update_gamma(self):
        """Annealing schedule for the distance term."""
        epoch_to_update = self.gamma_idx * self.gamma_update_epoch_step + self.gamma_warmup
        if (self.current_epoch + 1) > epoch_to_update:
            self.gamma = self.gamma_min + self.gamma_idx * self.gamma_update_step
            self.gamma_idx += 1
            print (' *- Gamma updated - new value:', self.gamma)


    def train(self, train_dataset, test_dataset, num_workers=0, chpnt_path=''):
        """Trains a model with given hyperparameters."""


        dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=num_workers, drop_last=True)#, sampler=train_sampler)
        n_data = len(train_dataset)
        assert(train_dataset.dataset_name == test_dataset.dataset_name)

        print(('\nPrinting model specifications...\n' +
               ' *- Path to the model: {0}\n' +
               ' *- Training dataset: {1}\n' +
               ' *- Number of training samples: {2}\n' +
               ' *- Number of epochs: {3}\n' +
               ' *- Loss criterion: {4}\n' +
               ' *- Batch size: {5}\n'
               ).format(self.model_path, train_dataset.dataset_name, n_data,
                   self.epochs, self.loss_fn, self.batch_size))

        if chpnt_path:
            # Pick up the last epochs specs
            self.load_checkpoint(chpnt_path)

        else:
            # Initialise the model
            self.model = self.init_model()
            self.start_epoch, self.lr = self.lr_schedule.pop(0)
            try:
                self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
            except:
                self.lr_update_epoch, self.new_lr = self.start_epoch - 1, self.lr
            self.vae_optimiser = self.init_optimiser()
            self.valid_losses = []
            self.epoch_losses = []
            
            # To track the average epoch action loss
            self.epoch_action_dist_list = []
            self.epoch_noaction_dist_list = []
            self.test_action_dist_list = []
            self.test_noaction_dist_list = []

            print((' *- Learning rate: {0}\n' +
                   ' *- Next lr update at {1} to the value {2}\n' +
                   ' *- Remaining lr schedule: {3}'
                   ).format(self.lr, self.lr_update_epoch, self.new_lr,
                   self.lr_schedule))

        es = ES.EarlyStopping(patience=300)
        num_parameters = self.count_parameters()
        self.opt['num_parameters'] = num_parameters
        print(' *- Model parameter/training samples: {0}'.format(
                num_parameters/len(train_dataset)))
        print(' *- Model parameters: {0}'.format(num_parameters))

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                spacing = 1
                print('{0:>2}{1}\n\t of dimension {2}'.format('', name, spacing),
                      list(param.shape))

        print('\nStarting to train the model...\n' )
        for self.current_epoch in range(self.start_epoch, self.epochs):
            # Restart the epoch distances
            self.epoch_action_dist = []
            self.epoch_noaction_dist = []
            
            # Update hyperparameters
            self.model.train()
            #self.update_beta()
            self.update_gamma()
            self.update_learning_rate(self.vae_optimiser)
            epoch_loss = np.zeros(11)
            for batch_idx, (img1, img2, action) in enumerate(dataloader):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                action = action.to(self.device)

                # VAE loss on img1
                dec_mean1,  enc_mean1 = self.model(img1)
                loss1= self.compute_loss(
                        img1, dec_mean1,  enc_mean1)

                # VAE loss on img2
                dec_mean2,  enc_mean2 = self.model(img2)
                loss2 = self.compute_loss(
                        img2, dec_mean2,  enc_mean2)

                # Average VAE loss for the pair of image batches
                loss = (loss1 + loss2) / 2

                # Action loss between the latent samples
                (w_dist_loss, pure_dist_loss,
                 dist_action_mean, dist_action_std,
                 dist_no_action_mean, dist_no_action_std)  = self.latent_mean_dist(
                     enc_mean1, enc_mean2,  action)

                # Optimise the VAE for the complete loss
                the_loss = loss + w_dist_loss
                self.vae_optimiser.zero_grad()
                the_loss.backward()
                self.vae_optimiser.step()

          
            
           
            valid_loss = np.zeros(10)

            # Check that the at least 350 epochs are done
            if (es.step(valid_loss[0]) and self.current_epoch > self.min_epochs) \
                or (self.current_epoch > self.min_epochs and self.epoch_gap > 0) \
                or self.current_epoch > self.max_epochs:
                self.plot_dist_hists(
                    self.epoch_action_dist, self.epoch_noaction_dist, 'training')
                break

            # Update the checkpoint only if no early stopping was done
            self.save_checkpoint(epoch_loss[0])

            # Print current loss values every epoch
            if (self.current_epoch + 1) % self.console_print == 0:
                print('Epoch {0}:'.format(self.current_epoch))
                print('   Train loss: {0:.3f} recon loss: {1:.3f} KL loss: {2:.3f} dist: {3:.3f}'.format(
                        epoch_loss[0], epoch_loss[1], epoch_loss[2], epoch_loss[4]))
                print('   Valid loss: {0:.3f} recon loss: {1:.3f} KL loss: {2:.3f} dist: {3:.3f}'.format(
                        valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[4]))
                print('   Beta: {0:.6e}'.format(self.beta))
                print('   Gamma: {0:.6e}'.format(self.gamma))
                print('   LR: {0:.6e}'.format(self.lr))
                print('   MD: {0:.6e}'.format(self.min_dist))
                #print('   Gap: {0:.6e}\n'.format(self.epoch_gap))
                
                
                
            # Print validation results when specified
            if (self.current_epoch + 1) % self.snapshot == 0:
                
                # Plot reconstructions
                self.plot_grid(dec_mean1)
                self.plot_grid(img1, name="input")
                self.model.eval()


        print('Training completed.')
        #self.plot_model_loss()
        self.model.eval()

        # Measure other d distances at the end of training for comparisson
        print('Calculating other distances...')
        original_distance_type = self.distance_type
        all_distance_types = ['1', '2', 'inf']
        self.batch_dist_dict = {}
        self.epoch_dist_dict = {}

        for dist_type in all_distance_types:
            self.distance_type = dist_type
            self.batch_dist_dict[dist_type] = {}
            self.epoch_dist_dict[dist_type] = {}
            print(' *- Distance type set to ', self.distance_type)
            
            after_training_train =  self.compute_test_loss(train_dataset)
            self.batch_dist_dict[dist_type]['train'] = list(map(lambda x: round(x, 3),
                                after_training_train))
            #self.epoch_dist_dict[dist_type]['train_action'] = round(self.test_action_dist / self.test_action_pairs, 2)
            self.epoch_dist_dict[dist_type]['train_noaction'] = round(self.test_noaction_dist / self.test_noaction_pairs, 2)

            after_training_test =  self.compute_test_loss(test_dataset)
            self.batch_dist_dict[dist_type]['test'] = list(map(lambda x: round(x, 3),
                                after_training_test))
            #self.epoch_dist_dict[dist_type]['test_action'] = round(self.test_action_dist / self.test_action_pairs, 2)
            self.epoch_dist_dict[dist_type]['test_noaction'] = round(self.test_noaction_dist / self.test_noaction_pairs, 2)

        self.distance_type = original_distance_type

        # Save the model
        torch.save(self.model.state_dict(), self.model_path)
        

    def save_checkpoint(self, epoch_ml, keep=False):
        """
        Saves a checkpoint during the training.
        """
        if keep:
            path = self.save_path + '_checkpoint{0}.pth'.format(self.current_epoch)
            checkpoint_type = 'epoch'
        else:
            path = self.save_path + '_lastCheckpoint.pth'
            checkpoint_type = 'last'

        training_dict = {
                'last_epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'vae_optimiser_state_dict': self.vae_optimiser.state_dict(),
                'last_epoch_loss': epoch_ml,
                'valid_losses': self.valid_losses,
                'epoch_losses': self.epoch_losses,
                'epoch_action_dist_list': self.epoch_action_dist_list,
                'epoch_noaction_dist_list': self.epoch_noaction_dist_list,
                'test_action_dist_list': self.test_action_dist_list,
                'test_noaction_dist_list': self.test_noaction_dist_list,
                'beta': self.beta,
                'beta_range': self.beta_range,
                'beta_steps': self.beta_steps,
                'beta_idx': self.beta_idx,
                'gamma_warmup': self.gamma_warmup,
                'gamma': self.gamma,
                'gamma_min': self.gamma_min,
                'gamma_idx': self.gamma_idx,
                'gamma_update_step': self.gamma_update_step,
                'gamma_update_epoch_step': self.gamma_update_epoch_step,
                'snapshot': self.snapshot,
                'console_print': self.console_print,
                'current_lr': self.lr,
                'lr_update_epoch': self.lr_update_epoch,
                'new_lr': self.new_lr,
                'lr_schedule': self.lr_schedule
                }
        torch.save({**training_dict, **self.opt}, path)
        print(' *- Saved {1} checkpoint {0}.'.format(self.current_epoch, checkpoint_type))


    def load_checkpoint(self, path, eval=False):
        """
        Loads a checkpoint and initialises the models to continue training.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model = self.init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.lr = checkpoint['current_lr']
        self.lr_update_epoch = checkpoint['lr_update_epoch']
        self.new_lr = checkpoint['new_lr']
        self.lr_schedule = checkpoint['lr_schedule']
        self.vae_optimiser= self.init_optimiser()
        self.vae_optimiser.load_state_dict(checkpoint['vae_optimiser_state_dict'])

        self.start_epoch = checkpoint['last_epoch'] + 1
        self.snapshot = checkpoint['snapshot']
        self.valid_losses = checkpoint['valid_losses']
        self.epoch_losses = checkpoint['epoch_losses']

        if 'epoch_action_dist_list' in checkpoint.keys():
            self.epoch_action_dist_list = checkpoint['epoch_action_dist_list']
            self.epoch_noaction_dist_list = checkpoint['epoch_noaction_dist_list']
            self.test_action_dist_list = checkpoint['test_action_dist_list']
            self.test_noaction_dist_list = checkpoint['test_noaction_dist_list']

        self.beta = checkpoint['beta']
        self.beta_range = checkpoint['beta_range']
        self.beta_steps = checkpoint['beta_steps']
        self.beta_idx = checkpoint['beta_idx']

        self.gamma_warmup = checkpoint['gamma_warmup']
        self.gamma = checkpoint['gamma']
        self.gamma_min = checkpoint['gamma_min']
        self.gamma_idx = checkpoint['gamma_idx']
        self.gamma_update_step = checkpoint['gamma_update_step']
        self.gamma_update_epoch_step = checkpoint['gamma_update_epoch_step']

        self.snapshot = checkpoint['snapshot']
        self.console_print = checkpoint['console_print']

        print(('\nCheckpoint loaded.\n' +
               ' *- Last epoch {0} with loss {1}.\n'
               ).format(checkpoint['last_epoch'],
               checkpoint['last_epoch_loss']))
        print(' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                self.lr, self.lr_update_epoch, self.new_lr)
              )
        if eval == False:
            self.model.train()
        else:
            self.model.eval()



