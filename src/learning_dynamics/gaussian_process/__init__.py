import torch
from .kernels import kernel_ode, kernel_sq

def gaussian_process_ode(train_time_points, prior_mean, prior_var, test_time_points, length_scale=1, nat_freq=None, damp_ratio=None):
    device = train_time_points.device
    num_batches, train_time = train_time_points.shape
    _, test_time = test_time_points.shape
    
    # compute covariance matrix for training points
    cov_batches = kernel_ode(train_time_points, train_time_points, length_scale, nat_freq, damp_ratio).to(device)

    cov_batches += torch.eye(cov_batches.size(-1), device=device).unsqueeze(0) * 1e-4

    cholesky = torch.linalg.cholesky(cov_batches + torch.diag_embed(prior_var))

    # gp likelihood calculation
    likelihood_constant = train_time * torch.log(torch.tensor([2*torch.pi], device=device))
    likelihood_log_determinant = 2 * torch.sum(torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)), 1)

    # prior_mean: (batch_size, train_time)
    prior_mean_reshaped = prior_mean.unsqueeze(-1)  # (batch_size, train_time, 1)
    inverse_cov_mean = torch.cholesky_solve(prior_mean_reshaped, cholesky)  # (batch_size, train_time, 1)
    prior_mean_trans = prior_mean_reshaped.transpose(-1, -2)  # (batch_size, 1, train_time)
    likelihood_quadratic = torch.matmul(prior_mean_trans, inverse_cov_mean).view(num_batches)

    gp_likelihood = -0.5 * (likelihood_constant + likelihood_quadratic + likelihood_log_determinant)

    if torch.equal(train_time_points, test_time_points):
        inverse_cov_cov = torch.cholesky_solve(cov_batches, cholesky)
        diag_cov = torch.diag_embed(torch.diagonal(cov_batches, dim1=-2, dim2=-1))
        gp_var = torch.sum(diag_cov - cov_batches * inverse_cov_cov, 1)
        gp_mean = torch.matmul(cov_batches.transpose(1, 2), inverse_cov_mean).squeeze(-1)
        gp_var = gp_var + 1e-6
        return gp_mean, gp_var, gp_likelihood
    else:
        cov_star = kernel_ode(train_time_points, test_time_points, length_scale, nat_freq, damp_ratio).to(device)      # (num_batches, train_time, test_time)
        cov_star_star = kernel_ode(test_time_points, test_time_points, length_scale, nat_freq, damp_ratio).to(device) # (num_batches, test_time, test_time)
        inverse_cov_cov_star = torch.cholesky_solve(cov_star, cholesky)    # (num_batches, train_time, test_time)
        gp_mean = torch.matmul(cov_star.transpose(1, 2), inverse_cov_mean).squeeze(-1) # (num_batches, test_time)
        gp_var_full = cov_star_star - torch.matmul(cov_star.transpose(1, 2), inverse_cov_cov_star)
        gp_var = torch.diagonal(gp_var_full, dim1=1, dim2=2) + 1e-4
        return gp_mean, gp_var, gp_likelihood
    
def gaussian_process_sq(train_time_points, prior_mean, prior_var, test_time_points, length_scale=1, nat_freq=None, damp_ratio=None):
    device = train_time_points.device
    num_batches, train_time = train_time_points.shape
    _, test_time = test_time_points.shape
    
    # compute covariance matrix for training points
    cov_batches = kernel_sq(train_time_points, train_time_points, length_scale).to(device)

    cov_batches += torch.eye(cov_batches.size(-1), device=device).unsqueeze(0) * 1e-4

    cholesky = torch.linalg.cholesky(cov_batches + torch.diag_embed(prior_var))

    # gp likelihood calculation
    likelihood_constant = train_time * torch.log(torch.tensor([2*torch.pi], device=device))
    likelihood_log_determinant = 2 * torch.sum(torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)), 1)

    # prior_mean: (batch_size, train_time)
    prior_mean_reshaped = prior_mean.unsqueeze(-1)  # (batch_size, train_time, 1)
    inverse_cov_mean = torch.cholesky_solve(prior_mean_reshaped, cholesky)  # (batch_size, train_time, 1)
    prior_mean_trans = prior_mean_reshaped.transpose(-1, -2)  # (batch_size, 1, train_time)
    likelihood_quadratic = torch.matmul(prior_mean_trans, inverse_cov_mean).view(num_batches)

    gp_likelihood = -0.5 * (likelihood_constant + likelihood_quadratic + likelihood_log_determinant)

    if torch.equal(train_time_points, test_time_points):
        inverse_cov_cov = torch.cholesky_solve(cov_batches, cholesky)
        diag_cov = torch.diag_embed(torch.diagonal(cov_batches, dim1=-2, dim2=-1))
        gp_var = torch.sum(diag_cov - cov_batches * inverse_cov_cov, 1)
        gp_mean = torch.matmul(cov_batches.transpose(1, 2), inverse_cov_mean).squeeze(-1)
        gp_var = gp_var + 1e-6
        return gp_mean, gp_var, gp_likelihood
    else:
        cov_star = kernel_sq(train_time_points, test_time_points, length_scale, nat_freq, damp_ratio).to(device)      # (num_batches, train_time, test_time)
        cov_star_star = kernel_sq(test_time_points, test_time_points, length_scale, nat_freq, damp_ratio).to(device) # (num_batches, test_time, test_time)
        inverse_cov_cov_star = torch.cholesky_solve(cov_star, cholesky)    # (num_batches, train_time, test_time)
        gp_mean = torch.matmul(cov_star.transpose(1, 2), inverse_cov_mean).squeeze(-1) # (num_batches, test_time)
        gp_var_full = cov_star_star - torch.matmul(cov_star.transpose(1, 2), inverse_cov_cov_star)
        gp_var = torch.diagonal(gp_var_full, dim1=1, dim2=2) + 1e-4
        return gp_mean, gp_var, gp_likelihood