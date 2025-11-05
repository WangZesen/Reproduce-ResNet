# The code is from https://github.com/jjsrf/SSAM-NEURIPS2024
# Permalink: https://github.com/jjsrf/SSAM-NEURIPS2024/blob/e7d16325f04fba9db738c657052473c45389dfc9/sam.py

import torch
from typing import List


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, v2=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.v2 = v2

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if self.v2 and "prev_grad" in self.state[p]:
                    grad = self.state[p]["prev_grad"]
                    prev_p = self.state[p]["prev_grad"]
                else:
                    grad = p.grad
                    prev_p = p
                if p.grad is None:
                    continue
                e_w = (torch.pow(prev_p, 2) if group["adaptive"] else 1.0) * grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if self.v2:
                    self.state[p]["prev_grad"] = torch.clone(p.grad)
                    self.state[p]["prev_u"] = torch.clone(p)
                if p.grad is None:
                    continue
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore
        assert closure is None
        self.second_step(zero_grad=False)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism

        grad_list = [
            (
                (
                    torch.abs((self.state[p]["prev_u"] if (self.v2 and "prev_u" in self.state[p]) else p))
                    if group["adaptive"]
                    else 1.0
                )
                * (self.state[p]["prev_grad"] if (self.v2 and "prev_grad" in self.state[p]) else p.grad)
            )
            .norm(p=2)
            .to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        if len(grad_list) == 0:
            return 0
        stack = torch.stack(grad_list)
        norm = torch.norm(stack, p=2)

        return norm


class S2SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=1.0, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(S2SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        last_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if "last_grad" not in self.state[p]:
                    self.state[p]["last_grad"] = torch.empty_like(p)

        grad_norm = self._grad_norm(last_grads)
        scale = self.param_groups[0]["rho"] / (grad_norm + 1e-12)

        cached_params = []
        params = []

        for group in self.param_groups:
            for p in group["params"]:
                if "cached_p" not in self.state[p]:
                    self.state[p]["cached_p"] = torch.empty_like(p)
                cached_params.append(self.state[p]["cached_p"])
                params.append(p)

        torch._foreach_copy_(cached_params, params)

        if len(last_grads) > 0:
            torch._foreach_add_(params, last_grads, alpha=scale)

    @torch.no_grad()
    def second_step(self):
        params = []
        cached_params = []
        grads = []
        last_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                params.append(p)
                cached_params.append(self.state[p]["cached_p"])
                grads.append(p.grad)
                last_grads.append(self.state[p]["last_grad"])
        # restore the parameters to the original values
        torch._foreach_copy_(params, cached_params)

        # save the lastest gradients
        torch._foreach_copy_(last_grads, grads)

        # do the actual "sharpness-aware" update
        self.base_optimizer.step()

    def _grad_norm(self, grads: List[torch.Tensor]) -> torch.Tensor | float:
        grad_norm_list = [g.norm(p=2) for g in grads]

        if len(grad_norm_list) == 0:
            return 0
        stack = torch.stack(grad_norm_list)
        norm = torch.norm(stack, p=2)
        return norm


class SAMV1(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAMV1, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        scale = self.param_groups[0]["rho"] / (grad_norm + 1e-12)

        params = []
        grads = []
        cached_params = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad)
                if "cached_p" not in self.state[p]:
                    self.state[p]["cached_p"] = torch.empty_like(p)
                cached_params.append(self.state[p]["cached_p"])

        torch._foreach_copy_(cached_params, params)
        torch._foreach_add_(params, grads, alpha=scale)

    @torch.no_grad()
    def second_step(self):
        params = []
        cached_params = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                params.append(p)
                cached_params.append(self.state[p]["cached_p"])

        torch._foreach_copy_(params, cached_params)
        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism

        grad_list = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        if len(grad_list) == 0:
            return 0
        stack = torch.stack(grad_list)
        norm = torch.norm(stack, p=2)

        return norm
