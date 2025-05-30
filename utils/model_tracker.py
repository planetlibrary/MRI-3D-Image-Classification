import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
import os

class ModelChangeTracker:
    def __init__(
        self,
        kl_threshold: float = 1e-3,
        track_kl: bool = True,
        track_norms: bool = True,
        track_grads: bool = True,
        config = None
    ):
        """
        kl_threshold: emit warning if KL < threshold
        track_kl:      compute KL-divergence of softmaxed params
        track_norms:   compute L2/L1 norm of delta params
        track_grads:   compute L2/L1/max norms of gradients
        """
        self.kl_threshold = kl_threshold
        self.track_kl    = track_kl
        self.track_norms = track_norms
        self.track_grads = track_grads

        # storage for logs
        self.prev_params: Optional[torch.Tensor] = None
        self.kl_divs:       List[Tuple[int, float]]           = []
        self.param_norms:   List[Tuple[int, float, float]]    = []
        self.grad_norms:    List[Tuple[int, float, float, float]] = []
        
        os.makedirs(config.logs_dir, exist_ok = True)

    def _flatten_params(self, model: torch.nn.Module) -> torch.Tensor:
        # concatenates all trainable params into one vector, on same device
        return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])

    def _flatten_grads(self, model: torch.nn.Module) -> torch.Tensor:
        # concatenates all gradients into one vector (zeros if no grad)
        grads = []
        for p in model.parameters():
            if p.requires_grad:
                g = p.grad
                if g is None:
                    # no gradient this step → treat as zero
                    grads.append(torch.zeros_like(p).flatten())
                else:
                    grads.append(g.detach().flatten())
        return torch.cat(grads)

    def update(self, model: torch.nn.Module, epoch: int):
        """
        Call this *after* optimizer.step() (so params are updated)
        and *after* backward() (so grads are populated).
        """
        device = next(model.parameters()).device
        curr = self._flatten_params(model).to(device)

        if self.prev_params is None:
            print(f"[Tracker] Epoch {epoch}: baseline captured ({curr.numel()} params).")
            with open(os.path.join(f'{config.logs_dir}', 'terminal.log'), 'a') as f:
                f.write(f"\n\t[Tracker] Epoch {epoch}: baseline captured ({curr.numel()} params).\n")
        else:
            # ----- 1) KL divergence -----
            if self.track_kl:
                prev_probs = F.softmax(self.prev_params, dim=0)
                curr_logp  = F.log_softmax(curr,     dim=0)
                kl = F.kl_div(curr_logp, prev_probs, reduction='batchmean').item()
                self.kl_divs.append((epoch, kl))
                print(f"[Tracker] Epoch {epoch}: KL divergence = {kl:.6g}")
                with open(os.path.join(f'{config.logs_dir}', 'terminal.log'), 'a') as f:
                    f.write(f"\n\t[Tracker] Epoch {epoch}: KL divergence = {kl:.6g}\n")

                if kl < self.kl_threshold:
                    print(f"  → Warning: KL divergence ({kl:.2e}) below threshold ({self.kl_threshold}).")

            # ----- 2) Parameter-norm changes -----
            if self.track_norms:
                delta = curr - self.prev_params
                l2 = delta.norm().item()
                l1 = delta.abs().sum().item()
                self.param_norms.append((epoch, l2, l1))
                print(f"[Tracker] Epoch {epoch}: ‖Δparams‖₂ = {l2:.4g}, ‖Δparams‖₁ = {l1:.4g}")
                with open(os.path.join(f'{config.logs_dir}', 'terminal.log'), 'a') as f:
                    f.write(f"\n\t[Tracker] Epoch {epoch}: ‖Δparams‖₂ = {l2:.4g}, ‖Δparams‖₁ = {l1:.4g}\n")

            # ----- 3) Gradient norms -----
            if self.track_grads:
                grads = self._flatten_grads(model).to(device)
                g_l2  = grads.norm().item()
                g_l1  = grads.abs().sum().item()
                g_max = grads.abs().max().item()
                self.grad_norms.append((epoch, g_l2, g_l1, g_max))
                print(f"[Tracker] Epoch {epoch}: ‖grads‖₂ = {g_l2:.4g}, ‖grads‖₁ = {g_l1:.4g}, max|grad| = {g_max:.4g}")
                with open(os.path.join(f'{config.logs_dir}', 'terminal.log'), 'a') as f:
                    f.write(f"\n\t[Tracker] Epoch {epoch}: ‖grads‖₂ = {g_l2:.4g}, ‖grads‖₁ = {g_l1:.4g}, max|grad| = {g_max:.4g}\n")

        # swap in the new baseline
        self.prev_params = curr.clone()

    # convenience getters
    def get_kl_log(self) -> List[Tuple[int, float]]:
        return self.kl_divs

    def get_param_norm_log(self) -> List[Tuple[int, float, float]]:
        return self.param_norms

    def get_grad_norm_log(self) -> List[Tuple[int, float, float, float]]:
        return self.grad_norms
