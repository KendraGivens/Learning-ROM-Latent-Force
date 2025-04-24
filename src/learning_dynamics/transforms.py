class PendulumTruncateTransform():
    def __init__(self, length_ratio=0.6):
        self.length_ratio = length_ratio
        
    def __call__(self, data):
        frames, trajectories = data
        # train with 60/40 split
        train_time = int(self.length_ratio * len(trajectories))
        return frames[:train_time], trajectories[:train_time]

class ShallowWaterTruncateTransform():
    def __init__(self, length_ratio=0.6, norm_constant=None):
        self.length_ratio = length_ratio
        self.norm_constant = norm_constant
        
    def __call__(self, data):
        frames, latents = data
        # train with 60/40 split
        train_time_frames = int(self.length_ratio * len(frames))
        train_time_latents = int(self.length_ratio * len(latents ))
        if self.norm_constant is not None:
            frames = frames / self.norm_constant
        return frames[:train_time_frames], latents[:train_time_latents]