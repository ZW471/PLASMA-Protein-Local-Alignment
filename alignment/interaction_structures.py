import torch
from torch import nn


class Sinkhorn(nn.Module):
    def __init__(self, n_iters: int = 20, eps: float = 1e-8, temperature: float = 1.0):
        super().__init__()
        self.n_iters = n_iters
        self.eps = eps
        self.temperature = temperature

    def forward(self, eta_result: torch.Tensor) -> torch.Tensor:
        # scores: (n_q, n_c)
        K = (eta_result[0] * self.temperature).exp_()               # in-place exp on scores if you don't need it later
        if len(eta_result) > 1:
            K = K * eta_result[1].float()
        n_q, n_c = K.shape

        # two scaling vectors
        u = torch.full((n_q,), 1.0 / n_q, device=K.device, dtype=K.dtype)
        v = torch.full((n_c,), 1.0 / n_c, device=K.device, dtype=K.dtype)

        # Sinkhorn–Knopp via matrix-vector muls
        for _ in range(self.n_iters):
            u = 1.0 / (K.matmul(v) + self.eps)
            v = 1.0 / (K.t().matmul(u) + self.eps)

        # final doubly-stochastic matrix: diag(u) @ K @ diag(v)
        return (u.unsqueeze(1) * K) * v.unsqueeze(0)


if __name__ == "__main__":
    from topotein_la.models.alignment.interaction_nonlinearity import hinge_non_linearity
    H_q = torch.randn([8, 5])
    H_c = torch.randn([10, 5])
    batch_q = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]).long()
    batch_c = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)

    result = hinge_non_linearity(H_q, H_c, batch_q, batch_c)

    omega = Sinkhorn()
    out = omega(result)
    print(out)