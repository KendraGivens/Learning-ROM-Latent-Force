from locale import currency
from networkx import current_flow_betweenness_centrality
import torch
import torch.nn as nn
import lightning as L
import math
from .gaussian_process import gaussian_process_ode, gaussian_process_sq
from .functions import gauss_cross_entropy, truncated_normal

class Encoder(torch.nn.Module):
    def __init__(self, px, py, embed_dim, latent_dim):
        super(Encoder, self).__init__()
        self.px = px
        self.py = py
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(self.px*self.py, self.embed_dim),
            nn.Tanh(),
            nn.Linear(self.embed_dim, 2*self.latent_dim)
        )
        
        self.apply(self.initialize_weights)

    # initialize weights with values drawn from truncated normal distribution
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data = truncated_normal(m.weight.data, std=1.0 / math.sqrt(float(m.in_features))+0.0000001)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        
        params = self.encoder(x.double().flatten(2)) 
        # reshape params: (batch, frames, 2)
        params = params.view(batch_size, -1, 2)
        mean = params[:, :, :params.shape[-1]//2]
        log_var = params[:, :, params.shape[-1]//2:]
        return mean, log_var

class Decoder(torch.nn.Module):
    def __init__(self, px, py, embed_dim, latent_dim):
        super(Decoder, self).__init__()
        self.px = px
        self.py = py
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.Tanh(),
            nn.Linear(self.embed_dim, self.px*self.py)
        )
        
        self.apply(self.initialize_weights)

    # initialize weights with values drawn from truncated normal distribution
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data = truncated_normal(m.weight.data, std=1.0 / math.sqrt(float(m.in_features))) #CHANGE +0.0000001
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        reconstructed = self.decoder(x)
        reconstructed = reconstructed.view(batch_size, -1, self.py, self.px)
        return reconstructed

class PendulumPEGPVAEModel(L.LightningModule):
    def __init__(self, encoder, decoder, dt):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dt = dt

        self.Dq = nn.Parameter((torch.tensor(0.0, dtype=torch.double)))
        self.Cq = nn.Parameter((torch.tensor(0.0, dtype=torch.double)))

    def _step(self, mode, batch):
        Dq =  nn.functional.softplus(self.Dq) # constrain to be > 0 
        Cq = nn.functional.softplus(self.Cq) # constrain to be > 0

        frequency = torch.sqrt(Dq)
        damping = Cq/(2*frequency)
        
        frame, trajectory = batch
        batch_size, time, px, py = frame.shape

        device = frame.device
        vae_mean, vae_var = self.encoder(frame.double())

        # create time for gp
        time_points = torch.arange(time, device=device) * self.dt
        batches_time = torch.cat([time_points.view(1, time) for i in range(batch_size)], 0)     
        
        # compute posterior mean and variance with the gaussian process
        # gp_mean, gp_var, gp_likelihood = gaussian_process(batches_time, vae_mean[:,:,0], vae_var[:,:,0], batches_time, length_scale, nat_freq, damp_ratio)
        gp_mean, gp_var, gp_likelihood = gaussian_process(batches_time, vae_mean[:,:,0], vae_var[:,:,0], batches_time, nat_freq=frequency, damp_ratio=damping)

        gp_mean = torch.unsqueeze(gp_mean, 2)
        gp_var = torch.unsqueeze(gp_var, 2)
        
        # compute loss function of vae
        # compute cross entropy between encoder and gp distribution predictions
        cross_entropy = torch.sum(gauss_cross_entropy(gp_mean, gp_var, vae_mean, vae_var), (1,2))
        # compute KL divergence bewteen prior distribution of gp and predicted distribution of vae
        kld = torch.mean(gp_likelihood - cross_entropy)
        
        # compute reconstruction loss terms    
        epsilon = torch.randn((batch_size, time, 1), device=device)
        latent_samples = gp_mean + epsilon * torch.sqrt(gp_var)
        reconstructed_video_logits = self.decoder(latent_samples)
        reconstructed_video = torch.sigmoid(reconstructed_video_logits)
        reconstruction_loss = (nn.BCEWithLogitsLoss(reduction='sum')(reconstructed_video_logits, frame)) / batch_size
        
        #compute total loss
        loss = reconstruction_loss - 50.*kld
        
        # log loss
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
        # return loss, (gp_mean, gp_var, latent_samples, reconstructed_video)

        return loss, (gp_mean, gp_var, latent_samples, reconstructed_video, frequency, damping)

    def training_step(self, batch):
        return self._step("training", batch)[0]

    def validation_step(self, batch):
        return self._step("validation", batch)

    def configure_optimizers(self):
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


class ShallowWaterPEGPVAEModel(L.LightningModule):
    def __init__(self, encoder, decoder, dt, kld_warmup_epochs=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dt = dt

        self.Dq = nn.Parameter((torch.tensor(0.0, dtype=torch.double)))
        self.Cq = nn.Parameter((torch.tensor(0.0, dtype=torch.double)))
        self.register_buffer("beta", torch.tensor(0.0))  
        self.kld_warmup_epochs = kld_warmup_epochs

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        self.beta.copy_(torch.tensor(min(1.0, (epoch + 1) / float(self.kld_warmup_epochs+1))))

        # self.beta.copy_(torch.tensor(min(1.0, (epoch + 1) / float(self.kld_warmup_epochs+1)*50)))
        # self.beta = min(self.beta, 50.0)

    def _step(self, mode, batch):
        Dq =  nn.functional.softplus(self.Dq) # constrain to be > 0 
        Cq = nn.functional.softplus(self.Cq) # constrain to be > 0

        frequency = torch.sqrt(Dq)
        damping = Cq/(2*frequency)
        
        frames, latents = batch
        batch_size, time, px, py = frames.shape

        device = frames.device
        vae_mean, vae_log_var = self.encoder(frames.double())
        vae_var = torch.exp(vae_log_var)

        # create time for gp
        time_points = torch.arange(time, device=device) * self.dt
        batches_time = torch.cat([time_points.view(1, time) for i in range(batch_size)], 0)     
        
        # compute posterior mean and variance with the gaussian process
        # gp_mean, gp_var, gp_likelihood = gaussian_process(batches_time, vae_mean[:,:,0], vae_var[:,:,0], batches_time, length_scale, nat_freq, damp_ratio)
        gp_mean, gp_var, gp_likelihood = gaussian_process_ode(batches_time, vae_mean[:,:,0], vae_var[:,:,0], batches_time, nat_freq=frequency, damp_ratio=damping)

        gp_mean = torch.unsqueeze(gp_mean, 2)
        gp_var = torch.unsqueeze(gp_var, 2)
        
        # compute loss function of vae
        # compute cross entropy between encoder and gp distribution predictions
        cross_entropy = torch.sum(gauss_cross_entropy(gp_mean, gp_var, vae_mean, vae_var), (1,2))
        # compute KL divergence bewteen prior distribution of gp and predicted distribution of vae
        kld = torch.mean(cross_entropy - gp_likelihood)   # = KL(q||p) ≥ 0
        # compute reconstruction loss terms    
        epsilon = torch.randn((batch_size, time, 1), device=device)
        latent_samples = gp_mean + epsilon * torch.sqrt(gp_var)
        reconstructed_video_logits = self.decoder(latent_samples)
        reconstructed_video = torch.sigmoid(reconstructed_video_logits)
        reconstruction_loss = nn.BCELoss(reduction="sum")(reconstructed_video, frames) / batch_size
        
        #compute total loss
        loss = reconstruction_loss + self.beta*kld
        
        # log loss
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
        # return loss, (gp_mean, gp_var, latent_samples, reconstructed_video)

        return loss, (gp_mean, gp_var, latent_samples, reconstructed_video, frequency, damping)

    def training_step(self, batch):
        return self._step("training", batch)[0]

    def validation_step(self, batch):
        return self._step("validation", batch)

    def configure_optimizers(self):
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer
    
class ShallowWaterRBFVAEModel(L.LightningModule):
    def __init__(self, encoder, decoder, dt, kld_warmup_epochs=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dt = dt
        self.register_buffer("beta", torch.tensor(0.0))  
        self.kld_warmup_epochs = kld_warmup_epochs

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        self.beta.copy_(torch.tensor(min(1.0, (epoch + 1) / float(self.kld_warmup_epochs+1))))

    def _step(self, mode, batch):
        frames, latents = batch
        batch_size, time, px, py = frames.shape

        device = frames.device
        vae_mean, vae_log_var = self.encoder(frames.double())
        vae_var = torch.exp(vae_log_var)

        # create time for gp
        time_points = torch.arange(time, device=device) * self.dt
        batches_time = torch.cat([time_points.view(1, time) for i in range(batch_size)], 0)     
        
        # compute posterior mean and variance with the gaussian process
        # gp_mean, gp_var, gp_likelihood = gaussian_process(batches_time, vae_mean[:,:,0], vae_var[:,:,0], batches_time, length_scale, nat_freq, damp_ratio)
        gp_mean, gp_var, gp_likelihood = gaussian_process_sq(batches_time, vae_mean[:,:,0], vae_var[:,:,0], batches_time, nat_freq=None, damp_ratio=None)

        gp_mean = torch.unsqueeze(gp_mean, 2)
        gp_var = torch.unsqueeze(gp_var, 2)
        
        # compute loss function of vae
        # compute cross entropy between encoder and gp distribution predictions
        cross_entropy = torch.sum(gauss_cross_entropy(gp_mean, gp_var, vae_mean, vae_var), (1,2))
        # compute KL divergence bewteen prior distribution of gp and predicted distribution of vae
        kld = torch.mean(cross_entropy - gp_likelihood)   # = KL(q||p) ≥ 0
        
        # compute reconstruction loss terms    
        epsilon = torch.randn((batch_size, time, 1), device=device)
        latent_samples = gp_mean + epsilon * torch.sqrt(gp_var)
        reconstructed_video_logits = self.decoder(latent_samples)
        reconstructed_video = torch.sigmoid(reconstructed_video_logits)
        reconstruction_loss = nn.BCELoss(reduction="sum")(reconstructed_video, frames) / batch_size
        
        #compute total loss
        loss = reconstruction_loss + self.beta*kld
        
        # log loss
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
        # return loss, (gp_mean, gp_var, latent_samples, reconstructed_video)

        return loss, (gp_mean, gp_var, latent_samples, reconstructed_video, None, None)

    def training_step(self, batch):
        return self._step("training", batch)[0]

    def validation_step(self, batch):
        return self._step("validation", batch)

    def configure_optimizers(self):
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer
        
        
class ShallowWaterVAEModel(L.LightningModule):
    def __init__(self, encoder, decoder, dt, kld_warmup_epochs=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.register_buffer("beta", torch.tensor(0.0))  
        self.kld_warmup_epochs = kld_warmup_epochs

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        self.beta.copy_(torch.tensor(min(1.0, (epoch + 1) / float(self.kld_warmup_epochs+1))))

    def _step(self, mode, batch):
        frames, latents = batch
        batch_size, time, px, py = frames.shape

        device = frames.device
        vae_mean, vae_log_var = self.encoder(frames.double())  
        vae_var = torch.exp(vae_log_var)

        std = torch.exp(0.5 * vae_log_var)              
        epsilon = torch.randn_like(std)
        
        latent_samples = vae_mean + epsilon * std                          

        logits = self.decoder(latent_samples)    
        reconstructed_video = torch.sigmoid(logits)
        reconstruction_loss = nn.BCELoss(reduction="sum")(reconstructed_video, frames) / batch_size

        kld = -0.5 * torch.sum(1 + vae_log_var - vae_mean.pow(2) - vae_log_var.exp(), dim=(1,2))
        kld = kld.mean()

        loss = reconstruction_loss + self.beta*kld
        
        # log loss
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
        return loss, (vae_mean, vae_var, latent_samples, reconstructed_video, None, None)

    def training_step(self, batch):
        return self._step("training", batch)[0]

    def validation_step(self, batch):
        return self._step("validation", batch)

    def configure_optimizers(self):
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer