class PendulumTruncateTransform():
    def __init__(self, length_ratio=0.6):
        self.length_ratio = length_ratio
        
    def __call__(self, data):
        frames, trajectories = data
        # train with 60/40 split
        train_time = int(self.length_ratio * len(trajectories))
        return frames[:train_time], trajectories[:train_time]

class ShallowWaterTruncateTransform():
    def __init__(self, length_ratio=0.6):
        self.length_ratio = length_ratio
        
    def __call__(self, data):
        # train with 60/40 split
        train_time = int(self.length_ratio * len(frames))
        return frames[:train_time]