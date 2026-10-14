import torch


class Optimizer():
    CHECK_SYNC_STEPS = 1000

    def __init__(self, config, param_list):
        self._param_list = param_list
        self._optimizer = self._build_optimizer(config, param_list)
        self._steps = 0

        self.sync()
        return
    
    def step(self, loss, **kwargs):
        self._optimizer.zero_grad()
        loss.backward()
        
        if "model" in kwargs:
            max_norm = kwargs["max_norm"]
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm(kwargs["model"].parameters(), max_norm, 2)
            if grad_norm.item() > max_norm:
                print("clipped grad norm:", grad_norm.item())

        self._optimizer.step()

        self._steps += 1
        return

    def get_steps(self):
        return self._steps

    def sync(self):
        return

    def _build_optimizer(self, config, param_list):
        lr = float(config["learning_rate"])
        weight_decay = float(config.get("weight_decay", 0.0))

        optimizer_type = config["type"]
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(param_list, lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_type == "Adam":
            optimizer = torch.optim.AdamW(param_list, lr, weight_decay=weight_decay)
        else:
            assert(False), "Unsupported optimizer type: " + optimizer_type
        return optimizer