"""BidirBuckPHNN — extended Port-Hamiltonian model for unidirectional + bidirectional operation.

Regime convention for this synchronous buck (no freewheeling diode):
  - Unidirectional (CCM): i_L ≥ 0 throughout — forward conduction only.
  - Bidirectional:        i_L goes negative — the synchronous MOSFET allows
                          reverse current that a freewheeling diode would block.
                          In a converter WITH a diode the same operating point
                          produces classical DCM (current clamped to zero).
                          Here, current actually flows backwards; the regime is
                          identified by i_L < 0 rather than i_L == 0.

Extends SimulationBuckPHNN with three changes that give the model capacity
to represent regime-dependent (unidirectional vs bidirectional) behaviour:

1. State-dependent dissipation R_theta(x): parameterised as a full 2×2 PSD
   matrix via a state-dependent Cholesky factor L_R(x). This lets the model
   represent the fact that reverse-current losses are fundamentally different
   from forward-current losses — the synchronous MOSFET conducts differently
   when i_L < 0.

2. Richer H_NN: 4 → 64 → 64 → 64 → 1 (vs 4 → 32 → 32 → 1 in the unidirectional
   model). The Hamiltonian correction needs more capacity to capture the
   non-quadratic energy storage that arises when the inductor current reverses.

3. Richer delta_G_net: 2 → 128 → 128 → 128 → 2 (vs none in the unidirectional
   SimBuckPHNN). The input coupling correction handles state-dependent gain
   changes across regimes. Output is a diagonal correction to G_physics so the
   structure constraint G = diag(g1, g2) is preserved.

All neural output layers are zero-initialised so training starts from the
pure-physics solution (same initialisation strategy as SimulationBuckPHNN).

The R_theta(x) Cholesky MLP is initialised so that at zero output the
resulting R equals diag(0.1, 0.05) — matching the unidirectional working
initial guess.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BidirBuckPHNN(nn.Module):
    """Port-Hamiltonian buck converter model for unidirectional and bidirectional operation.

    This synchronous buck has no freewheeling diode, so inductor current can go
    negative under light load. Unidirectional = i_L ≥ 0; the bidirectional
    regime = i_L < 0 (what would be classical DCM with a diode, but here current
    actually flows in reverse rather than clamping to zero).

    Fixed L, C from design. Learns:
      - H_NN(x, u_eff): rich neural Hamiltonian correction (64-unit hidden layers)
      - R_theta(x): state-dependent 2x2 PSD dissipation matrix via Cholesky MLP
      - delta_G(x): diagonal state-dependent input coupling correction

    The total dynamics are:
        ẋ = (J - R_theta(x)) ∇H_total(x, u_eff) + (G_physics + delta_G(x)) u_eff

    where H_total = H_physics + H_NN, R_theta(x) = L_R(x) @ L_R(x).T + eps*I,
    and G_physics = diag(1, -1) is the fixed physics input matrix.

    Inputs to forward():
        x_phys : [batch, 2]  physical states  [i_L [A], V_C [V]]
        u_eff  : [batch, 2]  effective inputs  [alpha*V_in [V], i_o [A]]

    Output:
        [batch, 2]  physical derivatives  [di_L/dt, dV_C/dt]
    """

    def __init__(
        self,
        L: float = 100e-6,
        C: float = 100e-6,
        initial_R_L: float = 0.10,
        initial_R_C: float = 0.05,
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

        # Typical operating scales for normalising H_NN inputs
        self.register_buffer("I_typical", torch.tensor(5.0, dtype=torch.float32))
        self.register_buffer("V_typical", torch.tensor(12.0, dtype=torch.float32))

        # ---- Larger H_NN: 4 → 64 → 64 → 64 → 1 ----
        # Input: [Phi_norm, q_norm, alpha*Vin, i_o]
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

        # ---- State-dependent Cholesky factor for R_theta(x) ----
        # Input: [i_L_norm, V_C_norm] (2D normalised state)
        # Output: 3 elements → lower-triangular L_R: [l00, l10, l11]
        # R_theta = L_R @ L_R.T + eps*I
        #
        # Initialise bias so that at zero network output:
        #   L_R = [[sqrt(R_L), 0], [0, sqrt(R_C)]]
        # which gives R_theta ≈ diag(R_L, R_C) + eps*I at the origin.
        self.R_chol_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.R_chol_net[-1].weight)
        # Bias initialised so L_R[0,0]=sqrt(R_L), L_R[1,0]=0, L_R[1,1]=sqrt(R_C)
        with torch.no_grad():
            self.R_chol_net[-1].bias.data = torch.tensor(
                [float(np.sqrt(initial_R_L)), 0.0, float(np.sqrt(initial_R_C))],
                dtype=torch.float32,
            )

        # ---- Diagonal delta_G: 2 → 128 → 128 → 128 → 2 ----
        # Output: [dg1, dg2] additive corrections to G_physics diagonal
        # Zero-initialised so G starts as pure physics
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_state(self, x_ph: torch.Tensor) -> torch.Tensor:
        """Normalise PH state (Phi, q) to order-unity for MLP inputs."""
        L = self.L.to(x_ph.device, x_ph.dtype)
        C = self.C.to(x_ph.device, x_ph.dtype)
        I_typ = self.I_typical.to(x_ph.device, x_ph.dtype)
        V_typ = self.V_typical.to(x_ph.device, x_ph.dtype)
        phi_norm = x_ph[..., 0:1] / (L * I_typ)
        q_norm = x_ph[..., 1:2] / (C * V_typ)
        return torch.cat([phi_norm, q_norm], dim=-1)

    def _h_nn(self, x_ph: torch.Tensor, u_eff: torch.Tensor) -> torch.Tensor:
        x_norm = self._normalise_state(x_ph)
        inp = torch.cat([x_norm, u_eff], dim=-1)
        return self.H_NN(inp)

    def _r_theta(self, x_ph: torch.Tensor) -> torch.Tensor:
        """Return state-dependent 2×2 PSD dissipation matrix R_theta(x).

        R_theta(x) = L_R(x) @ L_R(x).T + eps*I

        where L_R is lower-triangular:
            L_R = [[l00, 0  ],
                   [l10, l11]]
        """
        x_norm = self._normalise_state(x_ph)  # [..., 2]
        elems = self.R_chol_net(x_norm)       # [..., 3]: [l00, l10, l11]

        l00 = elems[..., 0:1]   # [..., 1]
        l10 = elems[..., 1:2]
        l11 = elems[..., 2:3]

        # Build lower-triangular L_R: [..., 2, 2]
        zeros = torch.zeros_like(l00)
        row0 = torch.cat([l00, zeros], dim=-1)    # [..., 2]
        row1 = torch.cat([l10, l11], dim=-1)       # [..., 2]
        L_R = torch.stack([row0, row1], dim=-2)    # [..., 2, 2]

        R = L_R @ L_R.transpose(-1, -2)           # [..., 2, 2]
        eps = self.r_epsilon
        eye = torch.eye(2, device=x_ph.device, dtype=x_ph.dtype)
        return R + eps * eye

    def _g_theta(self, x_ph: torch.Tensor) -> torch.Tensor:
        """Return state-dependent input matrix G_theta(x) = G_physics + diag(delta_G(x))."""
        x_norm = self._normalise_state(x_ph)     # [..., 2]
        dg = self.delta_G_net(x_norm)             # [..., 2]
        g_phys = self.G_physics.to(x_ph.device, x_ph.dtype)
        # Add diagonal correction to G_physics
        dg_mat = torch.diag_embed(dg)             # [..., 2, 2]
        return g_phys + dg_mat                    # [..., 2, 2]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x_phys: torch.Tensor,
        u_eff: torch.Tensor,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Return [di_L/dt, dV_C/dt] from physical states and effective inputs."""
        L = self.L.to(x_phys.device, x_phys.dtype)
        C = self.C.to(x_phys.device, x_phys.dtype)

        i_L = x_phys[..., 0]
        V_C = x_phys[..., 1]
        Phi = L * i_L
        q = C * V_C
        x_ph = torch.stack([Phi, q], dim=-1)
        u = u_eff.to(x_phys.device, x_phys.dtype)

        with torch.enable_grad():
            x_req = (
                x_ph if x_ph.requires_grad
                else x_ph.detach().clone().requires_grad_(True)
            )

            H_phys = (
                0.5 / L * x_req[..., 0:1].square()
                + 0.5 / C * x_req[..., 1:2].square()
            )
            H_total = H_phys + self._h_nn(x_req, u)

            grad_H = torch.autograd.grad(
                H_total.sum(),
                x_req,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]  # [..., 2]

            J = self.J.to(x_req.device, x_req.dtype)
            R = self._r_theta(x_req)               # [..., 2, 2]
            JmR = J - R                            # [..., 2, 2]

            # auto_part = (J - R) ∇H  — handles both batched and unbatched R
            auto_part = (JmR @ grad_H.unsqueeze(-1)).squeeze(-1)  # [..., 2]

            G = self._g_theta(x_req)               # [..., 2, 2]
            input_part = (G @ u.unsqueeze(-1)).squeeze(-1)  # [..., 2]

            xdot_ph = auto_part + input_part

        di_L_dt = xdot_ph[..., 0] / L
        dV_C_dt = xdot_ph[..., 1] / C
        return torch.stack([di_L_dt, dV_C_dt], dim=-1)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return diagnostics: fixed L/C, R_theta at origin, delta_G norm, H_NN norm."""
        with torch.no_grad():
            dev = self.L.device
            L = float(self.L)
            C = float(self.C)

            # Evaluate at a representative operating point
            x_rep = torch.tensor([[L * 1.0, C * 6.0]], dtype=torch.float32, device=dev)
            u_rep = torch.tensor([[6.0, 0.5]], dtype=torch.float32, device=dev)

            R_rep = self._r_theta(x_rep)[0].cpu()      # [2, 2]
            dg_rep = self.delta_G_net(
                self._normalise_state(x_rep)
            )[0].cpu()                                  # [2]
            H_NN_val = float(self._h_nn(x_rep, u_rep).abs())

        return {
            "L": L,
            "C": C,
            "R_11": float(R_rep[0, 0]),
            "R_12": float(R_rep[0, 1]),
            "R_22": float(R_rep[1, 1]),
            "delta_G_norm": float(dg_rep.norm()),
            "H_NN_norm": H_NN_val,
        }

    def get_regime_diagnostics(
        self,
        x_unidir: torch.Tensor,
        u_unidir: torch.Tensor,
        x_bidir: torch.Tensor,
        u_bidir: torch.Tensor,
    ) -> dict:
        """Evaluate H_NN and delta_G norms separately for unidirectional and bidirectional samples.

        Args:
            x_unidir: [batch, 2] physical states from unidirectional windows (i_L ≥ 0)
            u_unidir: [batch, 2] effective inputs from unidirectional windows
            x_bidir:  [batch, 2] physical states from bidirectional windows (i_L < 0)
            u_bidir:  [batch, 2] effective inputs from bidirectional windows

        Returns:
            Dict with 'H_NN_norm_unidir', 'H_NN_norm_bidir', 'dG_norm_unidir', 'dG_norm_bidir'.
        """
        with torch.no_grad():
            L = self.L.to(x_unidir.device, x_unidir.dtype)
            C = self.C.to(x_unidir.device, x_unidir.dtype)

            def _ph(xp: torch.Tensor) -> torch.Tensor:
                return torch.stack([L * xp[:, 0], C * xp[:, 1]], dim=-1)

            xph_unidir = _ph(x_unidir)
            xph_bidir = _ph(x_bidir)

            h_unidir = float(self._h_nn(xph_unidir, u_unidir).abs().mean())
            h_bidir = float(self._h_nn(xph_bidir, u_bidir).abs().mean())

            n_unidir = self._normalise_state(xph_unidir)
            n_bidir = self._normalise_state(xph_bidir)
            dg_unidir = float(self.delta_G_net(n_unidir).norm(dim=-1).mean())
            dg_bidir = float(self.delta_G_net(n_bidir).norm(dim=-1).mean())

        return {
            "H_NN_norm_unidir": h_unidir,
            "H_NN_norm_bidir": h_bidir,
            "dG_norm_unidir": dg_unidir,
            "dG_norm_bidir": dg_bidir,
        }
