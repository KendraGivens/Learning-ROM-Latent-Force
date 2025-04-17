import lightning as L
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.colors as mcolors

class PendulumPEGPVAEPlotting(L.pytorch.callbacks.Callback):
    def __init__(self, dt):
        super().__init__() 
        self.dt = dt
        self.frequency_per_epoch = []
        self.damping_per_epoch = []
        self.current_epoch_freq = []
        self.current_epoch_damping = []

    def mse_scale(self, a, b):
        return (torch.sum(b*a))/torch.sum(b**2)

    def heatmap(self, frames, time):
        video = np.array([(t+4)*v for t,v in enumerate(frames)]) 
        return np.max(video, 0)*(1/(4+time))
    
    def plot(self, true_frames, true_trajectories, reconstructed_frames, scale, mean, var, axes):
        cmap = plt.cm.Blues
        cmap = cmap(np.linspace(0, 1, cmap.N))
        cmap[:1, -1] = 0  
        transparent_cmap = mcolors.ListedColormap(cmap)
        num_frames = true_frames.shape[1]
        time = int(num_frames * self.dt)
        for i, ax in enumerate(axes[0]):
            image = self.heatmap(true_frames[i,:,:,:], num_frames)
            ax.imshow(image, cmap=transparent_cmap, extent=[0, image.shape[1], 0, image.shape[0]])
        
        for i, ax in enumerate(axes[2]):
            ax.plot(np.linspace(0, time, len(true_trajectories[i])), true_trajectories[i])        
    
        for i, ax in enumerate(axes[1]):
            image = self.heatmap(reconstructed_frames[i,:,:,:], num_frames)
            ax.imshow(image, cmap=transparent_cmap, extent=[0, image.shape[1], 0, image.shape[0]])
    
        for i, ax in enumerate(axes[3]):
            mean_i = scale*mean[i].squeeze().cpu()
            std_i = scale*np.sqrt(var[i].squeeze().cpu())
    
            x = np.linspace(0, time, len(mean_i))
            ax.plot(x, mean_i)
            ax.fill_between(x, mean_i + 2*std_i, mean_i - 2*std_i, alpha=0.5)
        
        return axes
    
    def on_validation_batch_end(self, trainer: L.Trainer, model, outputs, batch, batch_idx):        
        loss, (gp_mean, gp_var, latent_samples, reconstructed_video, frequency, damping) = outputs
        # loss, (gp_mean, gp_var, latent_samples, reconstructed_video) = outputs

        reconstructed_video = (reconstructed_video > 0.5).int()
        frames, trajectories = batch
        frames = frames.cpu()
        trajectories = trajectories.cpu() 
        batch_size, time, px, py = frames.shape

        scale = self.mse_scale(trajectories, gp_mean.cpu())
        scaled_latent_samples = latent_samples
        scaled_var = gp_var
        scaled_mean = gp_mean
        
        num_cols = batch_size if batch_size < 4 else 4 
        fig, axes = plt.subplots(4, num_cols, figsize=(10,8))    

        print("Frequency", frequency, "\nDamping", damping)
        
        self.plot(frames, trajectories, reconstructed_video.cpu().numpy(), scale, scaled_mean, scaled_var, axes=axes)
        fig.suptitle(str(trainer.current_epoch)+' ELBO: ' + str(-loss.item()))
        # trainer.logger.experiment.log({"reconstruction": wandb.Image(fig)}) 
        plt.show()

        # Collect frequency and damping for the current batch
        self.current_epoch_freq.append(frequency.item())
        self.current_epoch_damping.append(damping.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        avg_freq = np.mean(self.current_epoch_freq)
        avg_damp = np.mean(self.current_epoch_damping)
        self.frequency_per_epoch.append(avg_freq)
        self.damping_per_epoch.append(avg_damp)

        epochs = np.arange(1, len(self.frequency_per_epoch) + 1)
        fig, (ax_freq, ax_damp) = plt.subplots(nrows=1,ncols=2,figsize=(10, 3),constrained_layout=True)

        ax_freq.plot(epochs,self.frequency_per_epoch, marker='o',markersize=4,linewidth=1,color='tab:blue',label='Frequency')
        ax_freq.set_xlabel('Epoch', fontsize=10)
        ax_freq.set_ylabel('Frequency', color='tab:blue', fontsize=10)
        ax_freq.tick_params(axis='y', labelcolor='tab:blue')
        ax_freq.set_title('Frequency Over Epochs', fontsize=12)
        ax_freq.grid(True, linewidth=0.5)
        ax_freq.legend(loc='upper left', fontsize=8)

        ax_damp.plot(epochs,self.damping_per_epoch,marker='x',markersize=4, linewidth=1, color='tab:red', label='Damping')
        ax_damp.set_xlabel('Epoch', fontsize=10)
        ax_damp.set_ylabel('Damping', color='tab:red', fontsize=10)
        ax_damp.tick_params(axis='y', labelcolor='tab:red')
        ax_damp.set_title('Damping Over Epochs', fontsize=12)
        ax_damp.grid(True, linewidth=0.5)
        ax_damp.legend(loc='upper left', fontsize=8)

        plt.show()
        plt.close(fig) 
        self.current_epoch_freq = []
        self.current_epoch_damping = []  

class ShallowWaterPEGPVAEPlotting(L.pytorch.callbacks.Callback):
    def __init__(self, dt):
        super().__init__() 
        self.dt = dt
        self.frequency_per_epoch = []
        self.damping_per_epoch = []
        self.current_epoch_freq = []
        self.current_epoch_damping = []

    def heatmap(self, frames, time):
        video = np.array([(t+4)*v for t,v in enumerate(frames)]) 
        return np.max(video, 0)*(1/(4+time))
    
    def plot(self, true_frames, reconstructed_frames, mean, var, axes):
        cmap = plt.cm.Blues
        cmap = cmap(np.linspace(0, 1, cmap.N))
        cmap[:1, -1] = 0  
        transparent_cmap = mcolors.ListedColormap(cmap)
        num_frames = true_frames.shape[1]
        time = int(num_frames * self.dt)

        batch_size = true_frames.shape[0]
        num_cols = axes.shape[1]
        rand_idxs = np.random.choice(batch_size, size=num_cols, replace=False)
        
        for i, ax in enumerate(axes[0]):
            image = true_frames[rand_idxs[i],:,:,:]
            ax.imshow(image, cmap=transparent_cmap, extent=[0, image.shape[1], 0, image.shape[0]])       
    
        for i, ax in enumerate(axes[1]):
            image = reconstructed_frames[rand_idxs[i],:,:,:]
            ax.imshow(image, cmap=transparent_cmap, extent=[0, image.shape[1], 0, image.shape[0]])
    
        for i, ax in enumerate(axes[2]):
            mean_i = mean[i].squeeze().cpu()
            std_i = np.sqrt(var[i].squeeze().cpu())
    
            x = np.linspace(0, time, len(mean_i))
            ax.plot(x, mean_i)
            ax.fill_between(x, mean_i + 2*std_i, mean_i - 2*std_i, alpha=0.5)
        
        return axes
    
    def on_validation_batch_end(self, trainer: L.Trainer, model, outputs, batch, batch_idx):        
        loss, (gp_mean, gp_var, latent_samples, reconstructed_video, frequency, damping) = outputs

        reconstructed_video = (reconstructed_video > 0.5).int()
        frames = batch
        frames = frames.cpu()
        batch_size, time, px, py = frames.shape

        scaled_latent_samples = latent_samples
        scaled_var = gp_var
        scaled_mean = gp_mean
        
        num_cols = batch_size if batch_size < 4 else 4 
        fig, axes = plt.subplots(3, num_cols, figsize=(10,8))    

        print("Frequency", frequency, "\nDamping", damping)
        
        self.plot(frames, reconstructed_video.cpu().numpy(), scaled_mean, scaled_var, axes=axes)
        fig.suptitle(str(trainer.current_epoch)+' ELBO: ' + str(-loss.item()))
        # trainer.logger.experiment.log({"reconstruction": wandb.Image(fig)}) 
        plt.show()

        # Collect frequency and damping for the current batch
        self.current_epoch_freq.append(frequency.item())
        self.current_epoch_damping.append(damping.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        avg_freq = np.mean(self.current_epoch_freq)
        avg_damp = np.mean(self.current_epoch_damping)
        self.frequency_per_epoch.append(avg_freq)
        self.damping_per_epoch.append(avg_damp)

        epochs = np.arange(1, len(self.frequency_per_epoch) + 1)
        fig, (ax_freq, ax_damp) = plt.subplots(nrows=1,ncols=2,figsize=(10, 3),constrained_layout=True)

        ax_freq.plot(epochs,self.frequency_per_epoch, marker='o',markersize=4,linewidth=1,color='tab:blue',label='Frequency')
        ax_freq.set_xlabel('Epoch', fontsize=10)
        ax_freq.set_ylabel('Frequency', color='tab:blue', fontsize=10)
        ax_freq.tick_params(axis='y', labelcolor='tab:blue')
        ax_freq.set_title('Frequency Over Epochs', fontsize=12)
        ax_freq.grid(True, linewidth=0.5)
        ax_freq.legend(loc='upper left', fontsize=8)

        ax_damp.plot(epochs,self.damping_per_epoch,marker='x',markersize=4, linewidth=1, color='tab:red', label='Damping')
        ax_damp.set_xlabel('Epoch', fontsize=10)
        ax_damp.set_ylabel('Damping', color='tab:red', fontsize=10)
        ax_damp.tick_params(axis='y', labelcolor='tab:red')
        ax_damp.set_title('Damping Over Epochs', fontsize=12)
        ax_damp.grid(True, linewidth=0.5)
        ax_damp.legend(loc='upper left', fontsize=8)

        plt.show()
        plt.close(fig) 
        self.current_epoch_freq = []
        self.current_epoch_damping = []  
