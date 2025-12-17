class PendulumTruncateTransform():
    def __init__(self, length_ratio=0.6):
        self.length_ratio = length_ratio
        
    def __call__(self, data):
        frames, trajectories = data
        train_time = int(self.length_ratio * len(trajectories))
        return frames[:train_time], trajectories[:train_time]

class FingerTruncateTransform():
    def __init__(self, length_ratio=0.6):
        self.length_ratio = length_ratio
        
    def __call__(self, data):
        frames, latents = data
        train_time = int(self.length_ratio * len(frames))
        train_time_latents = int(self.length_ratio * len(latents))
        return frames[:train_time], latents[:train_time_latents]

class ShallowWaterTruncateTransform():
    def __init__(self, length_ratio=0.6, norm_constant=None, mean=None):
        self.length_ratio = length_ratio
        self.norm_constant = norm_constant
        self.mean = mean
        
    def __call__(self, data):
        frames, latents = data
        train_time_frames = int(self.length_ratio * len(frames))
        train_time_latents = int(self.length_ratio * len(latents))
        
        if self.norm_constant is not None:
            frames = (frames - self.mean) / self.norm_constant

        return frames[...,:train_time_frames,:,:], latents[...,:train_time_latents]
