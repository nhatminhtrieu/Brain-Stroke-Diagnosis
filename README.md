# Brain-Stroke-Diagnosis

# Notes for Pytorch implementation

- optimizer.zero_grad(set_to_none=True) is faster than optimizer.zero_grad() for large models
- torch.inference_mode() is better than torch.no_grad() for inference
- torch.autograd.set_detect_anomaly(True) is useful for debugging