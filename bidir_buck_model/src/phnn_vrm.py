"""VRMBuckPHNN -- Port-Hamiltonian model for the 12V->1V synchronous buck VRM.

Direct port of BidirBuckPHNN (phnn_bidir.py) to the VRM operating point
(COMPEL 2025 SCB VRM Table II parameters, single-phase conventional buck).

The three architectural differences from BidirBuckPHNN are in the
normalisation buffers only -- the network widths and structure are identical:

  I_typical : 5.0 A  ->  10.0 A   (VRM i_L DC range ~2-14 A)
  V_typical : 12.0 V ->   1.0 V   (VRM V_out ~ 1 V, not 12 V)
  L default : 100 uH ->  440 nH
  C default : 100 uF ->  200 uF
  initial_R_L: 0.10  ->  3e-3 Ohm (ESR_L from COMPEL 2025 Table II)
  initial_R_C: 0.05  ->  0.0  Ohm (ESR_C = 0 for VRM model)

Using V_typical=12V with V_out~1V would normalise the capacitor co-state to
q/(C*V_typ) = V_C/V_typ ~ 1/12 ~ 0.08, giving near-zero H_NN inputs and
very slow / poor convergence.  V_typical=1V gives q/(C*1) = V_C ~ 1.0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class VRMBuckPHNN(nn.Module):
    """Port-Hamiltonian VRM buck model for unidirectional and bidirectional operation.

    Identical architecture to BidirBuckPHNN; only the normalisation constants
    and physical defaults differ (see module docstring).

    Fixed L, C from design (440 nH, 200 uF).  Learns:
      - H_NN(x, u_eff): neural Hamiltonian correction  (64-unit hidden layers)
      - R_theta(x): state-dependent 2x2 PSD dissipation via Cholesky MLP
      - delta_G(x): diagonal state-dependent input coupling correction

    Dynamics:
        xdot = (J - R_theta(x)) grad_H_total(x, u_eff) + (G_physics + delta_G(x)) u_eff

    Inputs to forward():
        x_phys : [batch, 2]  physical states  [i_L [A], V_C [V]]
        u_eff  : [batch, 2]  effective inputs  [alpha*V_in [V], i_o [A]]

    Output:
        [batch, 2]  physical derivatives  [di_L/dt, dV_C/dt]
    """

    def __init__(
        self,
        L: float = 440e-9,
        C: float = 200e-6,
        initial_R_L: float = 3e-3,
        initial_R_C: float = 0.0,
        r_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        self.register_buffer("L", torch.tensor(L, dtype=torch.float32))
        self.register_buffer("C", torch.tensor(C, dtype=torch.float32))
        self.r_epsilon = float(r_epsilon)

        self.register_buffer(
            "J",
            torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32),
        )
        self.register_buffer(
            "G_physics",
            torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32),
        )

        # Normalisation scales for H_NN inputs.
        # I_typical=10A: VRM i_L DC range 2-14A, peaks to ~30A during transients.
        # V_typical=1V:  VRM V_out ~ 1V.  Using 12V (old value) gives inputs of
        #                ~0.08, making the NN effectively blind to voltage variations.
        self.register_buffer("I_typical", torch.tensor(10.0, dtype=torch.float32))
        self.register_buffer("V_typical", torch.tensor(1.0, dtype=torch.float32))

        # H_NN: 4 -> 64 -> 64 -> 64 -> 1
        self.H_NN = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.H_NN[-1].weight)
        nn.init.zeros_(self.H_NN[-1].bias)

        # State-dependent Cholesky factor for R_theta(x): 2 -> 64 -> 64 -> 3
        # Output [l00, l10, l11] -> lower-triangular L_R -> R = L_R @ L_R.T + eps*I
        # Bias: L_R[0,0] = sqrt(R_L), L_R[1,0] = 0, L_R[1,1] = sqrt(R_C)
        self.R_chol_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.R_chol_net[-1].weight)
        with torch.no_grad():
            self.R_chol_net[-1].bias.data = torch.tensor(
                [float(np.sqrt(initial_R_L)), 0.0, float(np.sqrt(max(initial_R_C, 0.0)))],
                dtype=torch.float32,
            )

        # Diagonal delta_G: 2 -> 128 -> 128 -> 128 -> 2  (zero-initialised)
        self.delta_G_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.delta_G_net[-1].weight)
        nn.init.zeros_(self.delta_G_net[-1].bias)

    def _normalise_state(self, x_ph: torch.Tensor) -> torch.Tensor:
        L = self.L.to(x_ph.device, x_ph.dtype)
        C = self.C.to(x_ph.device, x_ph.dtype)
        I_typ = self.I_typical.to(x_ph.device, x_ph.dtype)
        V_typ = self.V_typical.to(x_ph.device, x_ph.dtype)
        phi_norm = x_ph[..., 0:1] / (L * I_typ)
        q_norm   = x_ph[..., 1:2] / (C * V_typ)
        return torch.cat([phi_norm, q_norm], dim=-1)

    def _h_nn(self, x_ph: torch.Tensor, u_eff: torch.Tensor) -> torch.Tensor:
        x_norm = self._normalise_state(x_ph)
        return self.H_NN(torch.cat([x_norm, u_eff], dim=-1))

    def _r_theta(self, x_ph: torch.Tensor) -> torch.Tensor:
        x_norm = self._normalise_state(x_ph)
        elems  = self.R_chol_net(x_norm)        # [..., 3]
        l00 = elems[..., 0:1]
        l10 = elems[..., 1:2]
        l11 = elems[..., 2:3]
        zeros = torch.zeros_like(l00)
        L_R = torch.stack(
            [torch.cat([l00, zeros], dim=-1),
             torch.cat([l10, l11],   dim=-1)],
            dim=-2,
        )
        eps = self.r_epsilon
        return L_R @ L_R.transpose(-1, -2) + eps * torch.eye(
            2, device=x_ph.device, dtype=x_ph.dtype
        )

    def _g_theta(self, x_ph: torch.Tensor) -> torch.Tensor:
        x_norm = self._normalise_state(x_ph)
        dg = self.delta_G_net(x_norm)
        g_phys = self.G_physics.to(x_ph.device, x_ph.dtype)
        return g_phys + torch.diag_embed(dg)

    def forward(
        self,
        x_phys: torch.Tensor,
        u_eff: torch.Tensor,
        create_graph: bool = True,
    ) -> torch.Tensor:
        L = self.L.to(x_phys.device, x_phys.dtype)
        C = self.C.to(x_phys.device, x_phys.dtype)

        i_L = x_phys[..., 0]
        V_C = x_phys[..., 1]
        x_ph = torch.stack([L * i_L, C * V_C], dim=-1)
        u = u_eff.to(x_phys.device, x_phys.dtype)

        I_typ = self.I_typical.to(x_phys.device, x_phys.dtype)
        # H_NN energy scale: ensures the neural Hamiltonian correction and its
        # gradient are comparable in magnitude to the physical Hamiltonian.
        #
        # Without scaling, grad(H_NN)/grad(Phi) ~ 1/(L * I_typical).
        # With L=440nH: ~227,000 A/Wb vs. physical grad i_L ~ 5 A -- 45,000x too large.
        # Any nonzero H_NN output instantly dominates the ODE and diverges the rollout.
        #
        # Scaling H_NN by L * I_typical**2 brings the gradient to O(I_typical) ~ 10 A,
        # matching the physical scale. (Bidir model at L=100uH had ~400x mismatch and
        # survived by luck; VRM at 440nH is 113x worse and always diverges without this.)
        h_energy_scale = (L * I_typ * I_typ).detach()  # ~4.4e-5 J for VRM

        with torch.enable_grad():
            x_req = (
                x_ph if x_ph.requires_grad
                else x_ph.detach().clone().requires_grad_(True)
            )
            H_phys = (
                0.5 / L * x_req[..., 0:1].square()
                + 0.5 / C * x_req[..., 1:2].square()
            )
            H_total = H_phys + h_energy_scale * self._h_nn(x_req, u)
            grad_H = torch.autograd.grad(
                H_total.sum(), x_req,
                create_graph=create_graph, retain_graph=create_graph,
            )[0]

            J   = self.J.to(x_req.device, x_req.dtype)
            R   = self._r_theta(x_req)
            G   = self._g_theta(x_req)
            xdot_ph = (
                ((J - R) @ grad_H.unsqueeze(-1)).squeeze(-1)
                + (G @ u.unsqueeze(-1)).squeeze(-1)
            )

        return torch.stack([xdot_ph[..., 0] / L, xdot_ph[..., 1] / C], dim=-1)

    def get_params(self) -> dict:
        with torch.no_grad():
            dev = self.L.device
            L = float(self.L)
            C = float(self.C)
            # Representative VRM operating point: i_L=5A, V_C=1V, u=[1V, 5A]
            x_rep = torch.tensor([[L * 5.0, C * 1.0]], dtype=torch.float32, device=dev)
            u_rep = torch.tensor([[1.0, 5.0]],         dtype=torch.float32, device=dev)
            R_rep  = self._r_theta(x_rep)[0].cpu()
            dg_rep = self.delta_G_net(self._normalise_state(x_rep))[0].cpu()
            H_val  = float(self._h_nn(x_rep, u_rep).abs())
        return {
            "L": L, "C": C,
            "R_11": float(R_rep[0, 0]), "R_12": float(R_rep[0, 1]),
            "R_22": float(R_rep[1, 1]),
            "delta_G_norm": float(dg_rep.norm()),
            "H_NN_norm": H_val,
        }

    def get_regime_diagnostics(
        self,
        x_unidir: torch.Tensor, u_unidir: torch.Tensor,
        x_bidir:  torch.Tensor, u_bidir:  torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            L = self.L.to(x_unidir.device, x_unidir.dtype)
            C = self.C.to(x_unidir.device, x_unidir.dtype)

            def _ph(xp: torch.Tensor) -> torch.Tensor:
                return torch.stack([L * xp[:, 0], C * xp[:, 1]], dim=-1)

            xph_u = _ph(x_unidir); xph_b = _ph(x_bidir)
            h_u = float(self._h_nn(xph_u, u_unidir).abs().mean())
            h_b = float(self._h_nn(xph_b, u_bidir).abs().mean())
            dg_u = float(self.delta_G_net(self._normalise_state(xph_u)).norm(dim=-1).mean())
            dg_b = float(self.delta_G_net(self._normalise_state(xph_b)).norm(dim=-1).mean())
        return {
            "H_NN_norm_unidir": h_u, "H_NN_norm_bidir": h_b,
            "dG_norm_unidir": dg_u,  "dG_norm_bidir": dg_b,
        }
