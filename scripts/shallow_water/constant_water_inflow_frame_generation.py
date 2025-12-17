from __future__ import annotations
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import argparse, h5py, numpy as np, imageio, pickle
import matplotlib.pyplot as plt

def plot_data_gif(data, t_latent, dim, channel, cfg, out_prefix, gif=False):
    cmap = plt.get_cmap("viridis") if cfg.plot.use_color else None
    vmin, vmax  = data[..., channel].min(), data[..., channel].max()

    frames = []
    for arr in data:
        field = arr[..., channel].T
        if cfg.plot.use_color:
            rgba  = cmap(field, vmin=vmin, vmax=vmax)
            frame = (rgba[..., :3] * 255).astype(np.uint8)
        else:
            frame = (field * 255).astype(np.uint8)
        frames.append(frame)

    frames = np.stack(frames)
    if gif:
        suffix = "_gray" if not cfg.plot.use_color else ""
        imageio.mimsave(f"{out_prefix}{suffix}.gif", frames, duration=cfg.plot.duration)
    return frames

def plot_data(data_path: Path, cfg: DictConfig, out_dir: Path, k_sub: int, drop_sec: int, seed: str, gif: bool = False):
    with h5py.File(data_path, "r") as h5:
        latent_full = np.array(h5[f"{seed}/latent_sin"])
        t_full  = np.array(h5[f"{seed}/grid/t"])
        data_full = np.array(h5[f"{seed}/data"])       

    raw_dt  = t_full[1] - t_full[0]
    start_idx = int(np.round(drop_sec / raw_dt))

    t_full = t_full[start_idx:]
    latent_full = latent_full[start_idx:]
    data_full = data_full[start_idx:]
    
    t_sub = t_full[::k_sub][:-1]       
    latent_sub = latent_full[::k_sub][:-1]
    data_sub_unsqueeze = data_full[::k_sub][:-1]
    
    data_sub = np.squeeze(data_sub_unsqueeze, axis=-1)
    data_sub = data_sub.transpose(0, 2, 1)    

    baseline = data_sub[0]  
    # data_sub = data_sub - baseline

    prefix = out_dir / f"{cfg.name}_{seed}"

    if gif:
        prefix = out_dir / f"{cfg.name}_{seed}"
        _ = plot_data_gif(data_sub_unsqueeze, t_sub, cfg.plot.dim,cfg.plot.channel_idx, cfg, prefix, gif=True)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(t_sub, latent_sub, '-o', markersize=3)
        ax.set_title(f"Latent sine (seed={seed})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Latent sin")
        fig.tight_layout()
        fig.savefig(f"{prefix}_latent.png", dpi=150)
        plt.close(fig)

    return data_sub.astype(np.float32), t_sub, latent_sub

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=None)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    cfg_path = Path("/home/kendra/Code/libs/PDEBench/pdebench/data_gen/configs/constant_water_inflow.yaml")
    cfg: DictConfig = OmegaConf.load(cfg_path)

    raw_dt  = cfg.sim.T_end / cfg.sim.n_time_steps     
    k_sub = cfg.plot.subsample                       
    dt_used = raw_dt * k_sub                           
    drop_sec = 0                  

    data_dir = Path("/home/kendra/Code/Learning-Dynamics-Latent-Force/data/PDEs/Shallow_Water/latent_sin_ls_1")
    # data_dir = Path("/home/kendra/Code/Learning-Dynamics-Latent-Force/data/PDEs/Shallow_Water/sin")

    out_dir = data_dir
    out_dir.mkdir(exist_ok=True)
    
    frames_all, t_all, latent_all, pairs = [], [], [], []

    for h5_path in sorted(data_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as h5: seeds = list(h5.keys())
        for seed in (seeds if args.num is None else seeds[:args.num]):
            print(f"Processing {h5_path.name}  seed={seed}")
            frames_32, t, ls = plot_data(h5_path, cfg, out_dir, k_sub=k_sub, drop_sec=drop_sec, seed=seed, gif=args.gif)
            frames_all.append(frames_32)
            t_all.append(t)
            latent_all.append(ls)
            pairs.append((h5_path.name, seed))

    pickle_path = data_dir / "shallow_water_data.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump({
            "file_seed_pairs": pairs,
            "frames": np.array(frames_all),
            "latent_times": np.array(t_all),
            "latent_sins": np.array(latent_all),
            "time": cfg.sim.T_end - drop_sec,
            "dt": dt_used,         
        }, f)
    print("Saved ->", pickle_path)

if __name__ == "__main__":
    main()
