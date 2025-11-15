import torch
import numpy as np


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            score = out.squeeze()
            if score.dim() > 0:
                score = score.mean()
        else:
            score = out[:, class_idx].sum()
        score.backward(retain_graph=True)

        grads = self.gradients
        activations = self.activations
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        gcam = torch.relu(torch.sum(weights * activations, dim=1, keepdim=True))
        gcam = torch.nn.functional.interpolate(gcam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        gcam = gcam.squeeze().cpu().numpy()
        gcam = (gcam - gcam.min()) / (gcam.max() - gcam.min() + 1e-8)
        return gcam
