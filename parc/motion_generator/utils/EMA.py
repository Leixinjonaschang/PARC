class EMA():
    def __init__(self, decay):
        self.decay = decay
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new
    
    def update_model_average(self, ema_model, current_model):
        decay = self.decay
        for ema_params, current_params in zip(ema_model.parameters(), current_model.parameters()):
            ema_params.mul_(decay).add_(current_params, alpha=1.0 - decay)