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
import imageio

def plot_data_gif(data, t, dim, channel, config, output_prefix, gif=False):
    use_color = config.plot.use_color
    add_colorbar = getattr(config.plot, "add_colorbar", False)
    cmap = plt.get_cmap("viridis") if use_color else None

    vmin = np.inf
    vmax = -np.inf
    for arr in data:
        field = arr[..., channel].T
        vmin = min(vmin, np.min(field))
        vmax = max(vmax, np.max(field))

    frames = []
    for arr in data:
        field = arr[..., channel].T
        if use_color:
            if add_colorbar:
                fig, ax = plt.subplots(figsize=(4,4))
                im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                frame = buf[..., :3]
                plt.close()
            else:
                rgba = cmap(field)
                frame = (rgba[..., :3] * 255).astype(np.uint8)
        else:
            frame = (field * 255).astype(np.uint8)
        frames.append(frame)

    # drop initial frames if desired
    frames = np.stack(frames[50:])
    suffix = "_gray" if not use_color else ""
    if gif: 
        imageio.mimsave(f"{output_prefix}{suffix}.gif", frames, duration=config.plot.duration)
    return frames

def plot_data(data_path, config, output_path, seed: str | None = None, gif=False):
    with h5py.File(data_path, "r") as h5:
        seeds = list(h5.keys())
        seed = seed or seeds[0]

        latent_sin = np.array(h5[f"{seed}/latent_sin"])
        t_latent = np.array(h5[f"{seed}/grid/t"])

    # subsample the latent series
    t_latent = t_latent[:-1][500:]
    latent_sin = latent_sin[:-1][500:]

    if gif: 
        fig_path = output_path / f"{config.name}_{seed}_latent_sin.png"
        plt.figure()
        plt.plot(t_latent, latent_sin)
        plt.xlabel("time")
        plt.ylabel("latent sin value")
        plt.savefig(fig_path)
        plt.close()

    with h5py.File(data_path, "r") as h5:
        data = np.array(h5[f"{seed}/data"])
        t = np.array(h5[f"{seed}/grid/t"])
    k = config.plot.subsample
    data = data[:-1:k]
    t = t[:-1:k]

    prefix = output_path / f"{config.name}_{seed}"
    frames = plot_data_gif(data, t, config.plot.dim, config.plot.channel_idx, config, prefix, gif)
    return frames, t_latent, latent_sin

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", "-n", type=int, default=None, help="Number of seeds to process per file (None = all)")
    p.add_argument("--gif", "-gif", type=bool, default=False)
    args = p.parse_args()

    # load config
    config_path = Path("/home/kendra/Code/libs/PDEBench/pdebench/data_gen/configs/constant_water_inflow.yaml")
    config: DictConfig = OmegaConf.load(config_path)

    # directory containing all shallow water HDF5 files
    data_dir = Path("/home/kendra/Code/Learning-Dynamics-Latent-Force/data/PDEs/Shallow_Water")
    h5_paths = sorted(data_dir.glob("*.h5"))

    # ensure output directory exists
    output_dir = data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # containers for results
    frames_list = []
    latent_times_list = []
    latent_sins_list = []
    file_seed_pairs = []

    # process each file and its seeds
    for h5_path in h5_paths:
        print(f"Processing file: {h5_path.name}")
        with h5py.File(h5_path, "r") as h5:
            seeds = list(h5.keys())
        selected_seeds = seeds if args.num is None else seeds[:args.num]
        for seed in selected_seeds:
            print(f" - seed={seed}")
            f, t, ls = plot_data(h5_path, config, output_dir, seed, args.gif)
            frames_list.append(f)
            latent_times_list.append(t)
            latent_sins_list.append(ls)
            file_seed_pairs.append((h5_path.name, seed))

    # stack results
    frames = np.array(frames_list) 
    frames = frames / np.max(frames)
    latent_times = np.array(latent_times_list)
    latent_sins = np.array(latent_sins_list)

    # compute dt correctly
    dt = ((config.sim.T_end - 5) / float(config.sim.n_time_steps - 500)) * config.plot.subsample

    # save combined pickle
    with open(data_dir / "shallow_water_data.pkl", "wb") as f:
        pickle.dump({
            "file_seed_pairs": file_seed_pairs,
            "frames": frames,
            "latent_times": latent_times,
            "latent_sins": latent_sins,
            "time": config.sim.T_end,
            "dt": dt,
        }, f)

if __name__ == "__main__":
    main()
