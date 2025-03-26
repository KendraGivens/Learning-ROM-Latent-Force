from __future__ import annotations
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import argparse
import os
import h5py
import numpy as np
import imageio
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def plot_data_gif(data, t, dim, channel, config, output_prefix):
    use_color = config.plot.use_color
    cmap = cm.get_cmap("viridis") if use_color else None

    frames = []
    for arr in data:
        field = arr[..., channel].T
        norm_field = (field - field.min()) / (field.max() - field.min())
        
        if use_color:
            rgba = cmap(norm_field)
            rgb = (rgba[..., :3] * 255).astype(np.uint8)
            frames.append(rgb)
        else:
            gray = (norm_field * 255).astype(np.uint8)
            frames.append(gray)

    frames = np.stack(frames)
    suffix = "_gray" if not use_color else ""
    gif_path = f"{output_prefix}{suffix}.gif"
    imageio.mimsave(gif_path, frames, duration=config.plot.duration)
    print(frames.shape)
    return frames
    
def plot_data(data_path, config, output_path, seed: str | None = None):
    with h5py.File(data_path, "r") as h5:
        seeds = list(h5.keys())
        seed = seed or seeds[0]

        latent = np.array(h5[f"{seed}/latent"])
        t_latent = np.array(h5[f"{seed}/grid/t"])
        fig_path = output_path / f"{config.name}_{seed}_latent.png"

        print("LATENT", latent.shape)
        print("T", t_latent.shape)

        plt.figure()
        plt.plot(t_latent, latent)
        plt.xlabel("time")
        plt.ylabel("latent value")
        plt.savefig(fig_path)
        plt.close()

        data = np.array(h5[f"{seed}/data"])
        t    = np.array(h5[f"{seed}/grid/t"])
        
        k = config.plot.subsample
        data = data[:-1:k]
        t    = t[:-1:k]
        
        prefix = output_path / f"{config.name}_{seed}"
        plot_data_gif(data, t, config.plot.dim, config.plot.channel_idx, config, prefix)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", "-n", type=int, default=1)
    args = p.parse_args()

    config_path = Path("/home/kendra/Code/libs/PDEBench/pdebench/data_gen/configs/constant_water_inflow.yaml")
    h5_path     = Path("/home/kendra/Code/Learning-Dynamics-Latent-Force/data/PDEs/Shallow_Water/2D_cwi_NA_NA.h5")
    output_dir  = h5_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    config: DictConfig = OmegaConf.load(config_path)

    with h5py.File(h5_path, "r") as h5:
        seeds = list(h5.keys())[: args.num]

    for seed in seeds:
        print(f"Saving GIF + latent plot for seed={seed}")
        plot_data(h5_path, config, output_dir, seed)


if __name__ == "__main__":
    main()

    