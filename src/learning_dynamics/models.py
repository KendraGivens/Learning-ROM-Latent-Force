import torch
import torch.nn as nn
import lightning as L
from .gaussian_process import gaussian_process
from .functions import gauss_cross_entropy

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
        var = torch.exp(params[:, :, params.shape[-1]//2:])
        return mean, var

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
        
        frame = batch
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