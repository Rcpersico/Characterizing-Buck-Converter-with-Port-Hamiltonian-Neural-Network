"""Port-Hamiltonian neural networks for LC and damped RLC buck experiments."""

from __future__ import annotations

import torch
import numpy as np
from torch import nn


class LCTankPHNN(nn.Module):
    """Learn a scalar Hamiltonian and derive LC-tank dynamics from its gradient.

    The public input and output coordinates are the physical energy coordinates
    ``x = (Phi, q)``.  Internally, ``x`` is divided by ``state_scale`` before it
    enters the MLP so both coordinates are order-one during optimization.
    Autograd still differentiates with respect to the unscaled physical state,
    so the returned vector field is ``dx/dt = J grad_x H_theta(x)`` in physical
    units.
    """

    def __init__(self, state_scale: torch.Tensor | None = None) -> None:
        """Initialize the Hamiltonian MLP.

        Args:
            state_scale: Optional shape-(2,) tensor used to normalize
                ``(Phi, q)`` before the MLP.  Defaults to ones.
        """
        super().__init__()
        if state_scale is None:
            state_scale = torch.ones(2, dtype=torch.float32)

        self.register_buffer("state_scale", state_scale.detach().clone().float())
        self.register_buffer(
            "J",
            torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32),
        )

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Use small final-layer weights to keep initial vector fields tame."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.uniform_(final.weight, -1e-4, 1e-4)
            nn.init.zeros_(final.bias)

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """Return the learned scalar Hamiltonian ``H_theta(x)``.

        Args:
            x: State tensor with shape ``[batch, 2]`` in ``(Phi, q)`` units.

        Returns:
            Tensor with shape ``[batch, 1]``.
        """
        x_normalized = x / self.state_scale.to(device=x.device, dtype=x.dtype)
        return self.net(x_normalized)

    def forward(self, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """Return ``dx/dt = J grad_x H_theta(x)``.

        Args:
            x: State tensor with shape ``[batch, 2]`` in ``(Phi, q)`` units.
            create_graph: Whether to retain the derivative graph for training.

        Returns:
            Predicted derivative tensor with shape ``[batch, 2]``.
        """
        with torch.enable_grad():
            if x.requires_grad:
                x_req = x
            else:
                x_req = x.detach().clone().requires_grad_(True)

            h = self.hamiltonian(x_req)
            grad_h = torch.autograd.grad(
                h.sum(),
                x_req,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
            j = self.J.to(device=x_req.device, dtype=x_req.dtype)
            return grad_h @ j.T


class RLCBuckPHNN(nn.Module):
    """Learn no-input damped buck dynamics with PH dissipation.

    The no-input buck/RLC dynamics are

        dx/dt = (J - R_theta) grad H_theta(x)

    with ``x = (Phi, q)`` and canonical ``J``.  For the Stage 2 identifier,
    ``H_theta`` is a learned positive diagonal quadratic Hamiltonian:

        H_theta = 0.5 * inv_L * Phi^2 + 0.5 * inv_C * q^2

    This keeps the physical scale of ``grad H`` identifiable, so the learned
    dissipation entry ``R_theta[0, 0]`` can be compared directly with the true
    inductor series resistance.
    """

    def __init__(
        self,
        initial_l: float = 100e-6,
        initial_c: float = 100e-6,
        initial_r: float = 0.05,
        r_epsilon: float = 1e-6,
        r_factor_mask: torch.Tensor | None = None,
    ) -> None:
        """Initialize the learned Hamiltonian and dissipative structure.

        Args:
            initial_l: Initial inductance value used for ``H_theta`` [H].
            initial_c: Initial capacitance value used for ``H_theta`` [F].
            initial_r: Initial inductor resistance for ``R_theta[0, 0]`` [Ohm].
            r_epsilon: Small positive diagonal added to ``R_theta``.
            r_factor_mask: Optional shape-(2, 2) mask applied after taking the
                lower-triangular part of the learnable ``L_R``.  The default
                constrains dissipation to the inductor branch, giving
                ``R_theta[1, 1] ~= r_epsilon``.
        """
        super().__init__()
        self.log_inv_l = nn.Parameter(torch.tensor(float(1.0 / initial_l)).log())
        self.log_inv_c = nn.Parameter(torch.tensor(float(1.0 / initial_c)).log())
        self.r_epsilon = float(r_epsilon)

        if r_factor_mask is None:
            r_factor_mask = torch.tensor(
                [[1.0, 0.0], [0.0, 0.0]],
                dtype=torch.float32,
            )

        self.register_buffer("r_factor_mask", r_factor_mask.detach().clone().float())
        self.register_buffer(
            "J",
            torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32),
        )

        r_factor = torch.zeros(2, 2, dtype=torch.float32)
        r_factor[0, 0] = float(max(initial_r, r_epsilon) ** 0.5)
        self.r_factor_raw = nn.Parameter(r_factor)

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """Return the learned scalar Hamiltonian ``H_theta(x)``."""
        inv_l = torch.exp(self.log_inv_l).to(device=x.device, dtype=x.dtype)
        inv_c = torch.exp(self.log_inv_c).to(device=x.device, dtype=x.dtype)
        phi = x[:, 0:1]
        q = x[:, 1:2]
        return 0.5 * inv_l * phi.square() + 0.5 * inv_c * q.square()

    def r_factor(self) -> torch.Tensor:
        """Return the masked lower-triangular dissipation factor ``L_R``."""
        mask = self.r_factor_mask.to(
            device=self.r_factor_raw.device,
            dtype=self.r_factor_raw.dtype,
        )
        return torch.tril(self.r_factor_raw) * mask

    def dissipation_matrix(self) -> torch.Tensor:
        """Return ``R_theta = L_R L_R^T + epsilon I``."""
        lr = self.r_factor()
        eye = torch.eye(2, device=lr.device, dtype=lr.dtype)
        return lr @ lr.T + self.r_epsilon * eye

    def learned_physical_parameters(self) -> dict[str, float]:
        """Return learned ``L``, ``C``, and entries of ``R_theta`` as floats."""
        with torch.no_grad():
            r = self.dissipation_matrix().detach().cpu()
            inv_l = float(torch.exp(self.log_inv_l).detach().cpu())
            inv_c = float(torch.exp(self.log_inv_c).detach().cpu())
        return {
            "L": 1.0 / inv_l,
            "C": 1.0 / inv_c,
            "R_11": float(r[0, 0]),
            "R_12": float(r[0, 1]),
            "R_22": float(r[1, 1]),
        }

    def forward(self, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """Return ``dx/dt = (J - R_theta) grad_x H_theta(x)``."""
        with torch.enable_grad():
            if x.requires_grad:
                x_req = x
            else:
                x_req = x.detach().clone().requires_grad_(True)

            h = self.hamiltonian(x_req)
            grad_h = torch.autograd.grad(
                h.sum(),
                x_req,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
            j = self.J.to(device=x_req.device, dtype=x_req.dtype)
            r = self.dissipation_matrix().to(device=x_req.device, dtype=x_req.dtype)
            return grad_h @ (j - r).T


class BuckPHNN(nn.Module):
    """Port-Hamiltonian buck converter model with external inputs.

    Implements ``dx/dt = (J - R_theta) grad H_theta(x) + G_theta(x) u_eff``
    where ``u_eff = [alpha * V_in, i_o]`` is the effective 2-vector input.

    The input matrix is ``G_theta(x) = G_physics + delta_G_theta(x)`` where
    ``G_physics = [[1, 0], [0, -1]]`` is fixed from physics and
    ``delta_G_theta`` is a state-dependent correction learned by a small MLP.
    The output layer of ``delta_G_theta`` is zero-initialised so the correction
    starts at zero and only grows if the data demands it.

    ``H_theta`` and ``R_theta`` share the same parameterisation as
    ``RLCBuckPHNN`` and are warm-started from that checkpoint in training.
    """

    def __init__(
        self,
        initial_l: float = 100e-6,
        initial_c: float = 100e-6,
        initial_r: float = 0.05,
        r_epsilon: float = 1e-6,
        r_factor_mask: torch.Tensor | None = None,
    ) -> None:
        """Initialise learnable PH structure and input-coupling correction.

        Args:
            initial_l: Initial inductance guess [H].
            initial_c: Initial capacitance guess [F].
            initial_r: Initial inductor series resistance guess [Ohm].
            r_epsilon: Small positive diagonal added to ``R_theta`` for the
                PSD guarantee.
            r_factor_mask: Optional 2×2 mask for the Cholesky factor of
                ``R_theta``.  Defaults to inductor-branch-only dissipation.
        """
        super().__init__()
        self.log_inv_l = nn.Parameter(torch.tensor(float(1.0 / initial_l)).log())
        self.log_inv_c = nn.Parameter(torch.tensor(float(1.0 / initial_c)).log())
        self.r_epsilon = float(r_epsilon)

        if r_factor_mask is None:
            r_factor_mask = torch.tensor(
                [[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32
            )

        self.register_buffer("r_factor_mask", r_factor_mask.detach().clone().float())
        self.register_buffer(
            "J",
            torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32),
        )
        # Fixed physical input matrix G_physics = [[1, 0], [0, -1]].
        # Matches dPhi/dt = alpha*V_in - ..., dq/dt = ... - i_o exactly.
        self.register_buffer(
            "G_physics",
            torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32),
        )

        r_factor = torch.zeros(2, 2, dtype=torch.float32)
        r_factor[0, 0] = float(max(initial_r, r_epsilon) ** 0.5)
        self.r_factor_raw = nn.Parameter(r_factor)

        # State-dependent correction delta_G_theta: 2 -> 32 -> 4, tanh.
        # Only the output layer is zeroed so gradients are alive from epoch 1.
        self.delta_G_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 4),
        )
        nn.init.xavier_uniform_(self.delta_G_net[0].weight)
        nn.init.zeros_(self.delta_G_net[0].bias)
        nn.init.zeros_(self.delta_G_net[2].weight)
        nn.init.zeros_(self.delta_G_net[2].bias)

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """Return learned scalar Hamiltonian ``H_theta(x)``.

        Args:
            x: State tensor shape ``[batch, 2]`` in ``(Phi, q)`` coordinates.

        Returns:
            Tensor shape ``[batch, 1]``.
        """
        inv_l = torch.exp(self.log_inv_l).to(device=x.device, dtype=x.dtype)
        inv_c = torch.exp(self.log_inv_c).to(device=x.device, dtype=x.dtype)
        phi = x[:, 0:1]
        q = x[:, 1:2]
        return 0.5 * inv_l * phi.square() + 0.5 * inv_c * q.square()

    def r_factor(self) -> torch.Tensor:
        """Return masked lower-triangular Cholesky factor ``L_R``."""
        mask = self.r_factor_mask.to(
            device=self.r_factor_raw.device, dtype=self.r_factor_raw.dtype
        )
        return torch.tril(self.r_factor_raw) * mask

    def dissipation_matrix(self) -> torch.Tensor:
        """Return ``R_theta = L_R L_R^T + epsilon I`` (PSD by construction)."""
        lr = self.r_factor()
        eye = torch.eye(2, device=lr.device, dtype=lr.dtype)
        return lr @ lr.T + self.r_epsilon * eye

    def forward(
        self,
        x: torch.Tensor,
        u_eff: torch.Tensor,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Return ``dx/dt = (J - R_theta) grad H_theta(x) + G_theta(x) u_eff``.

        Args:
            x: State tensor shape ``[batch, 2]`` in ``(Phi, q)`` coordinates.
            u_eff: Input tensor shape ``[batch, 2]`` containing
                ``[alpha * V_in, i_o]`` per sample.
            create_graph: Retain autograd graph for backprop through
                ``autograd.grad``.

        Returns:
            Predicted state derivative, shape ``[batch, 2]``.
        """
        with torch.enable_grad():
            x_req = (
                x if x.requires_grad else x.detach().clone().requires_grad_(True)
            )

            h = self.hamiltonian(x_req)
            grad_h = torch.autograd.grad(
                h.sum(),
                x_req,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]

            j = self.J.to(device=x_req.device, dtype=x_req.dtype)
            r = self.dissipation_matrix().to(device=x_req.device, dtype=x_req.dtype)
            auto_part = grad_h @ (j - r).T  # [batch, 2]

            # State-dependent input matrix correction
            delta_G_flat = self.delta_G_net(x_req)          # [batch, 4]
            delta_G = delta_G_flat.view(-1, 2, 2)            # [batch, 2, 2]
            g_phys = self.G_physics.to(device=x_req.device, dtype=x_req.dtype)
            G = g_phys.unsqueeze(0) + delta_G               # [batch, 2, 2]

            u = u_eff.to(device=x_req.device, dtype=x_req.dtype).unsqueeze(-1)
            input_part = (G @ u).squeeze(-1)                 # [batch, 2]

        return auto_part + input_part

    def get_params(self) -> dict:
        """Return learned physical parameters and mean effective input matrix.

        Returns:
            Dict with keys ``'L'`` [H], ``'C'`` [F], ``'R_eff'`` [Ohm], and
            ``'G_eff'`` (2×2 tensor averaged over a representative state grid).
        """
        with torch.no_grad():
            inv_l = torch.exp(self.log_inv_l)
            inv_c = torch.exp(self.log_inv_c)
            r = self.dissipation_matrix()
            L = float(1.0 / inv_l)
            C = float(1.0 / inv_c)

            # Evaluate mean delta_G over a 5×5 grid spanning typical (Phi, q)
            phi_vals = torch.linspace(-1.0 * L, 3.0 * L, 5, dtype=torch.float32)
            q_vals = torch.linspace(0.0, 10.0 * C, 5, dtype=torch.float32)
            phi_g, q_g = torch.meshgrid(phi_vals, q_vals, indexing="ij")
            pts = torch.stack([phi_g.flatten(), q_g.flatten()], dim=1).to(
                device=self.G_physics.device, dtype=self.G_physics.dtype
            )
            delta_G_mean = self.delta_G_net(pts).mean(dim=0).view(2, 2)
            G_eff = self.G_physics + delta_G_mean

        return {
            "L": L,
            "C": C,
            "R_eff": float(r[0, 0]),
            "G_eff": G_eff.cpu(),
        }


class PhysicalBuckModel(nn.Module):
    """
    Parameterized averaged buck converter model in physical coordinates (i_L, V_C).

    Learns the averaged ODE:
        di_L/dt = inv_L * (alpha * V_in - R * i_L - V_C)
        dV_C/dt = inv_C * (i_L - i_o)

    where inv_L = 1/L, inv_C = 1/C, and R are learnable scalars.
    This model can be trained directly on measured (i_L, V_C) trajectories
    without prior knowledge of L or C.

    Args:
        initial_L: Initial inductance guess [H].
        initial_C: Initial capacitance guess [F].
        initial_R: Initial series resistance guess [Ohm].

    Inputs:
        x: Tensor [..., 2] of [i_L [A], V_C [V]]
        u: Tensor [..., 2] of [alpha * V_in [V], i_o [A]]

    Output:
        Tensor [..., 2] of [di_L/dt [A/s], dV_C/dt [V/s]]
    """

    def __init__(
        self,
        initial_L: float = 100e-6,
        initial_C: float = 100e-6,
        initial_R: float = 0.1,
    ) -> None:
        super().__init__()
        self.log_inv_L = nn.Parameter(
            torch.tensor(float(np.log(1.0 / initial_L)), dtype=torch.float32)
        )
        self.log_inv_C = nn.Parameter(
            torch.tensor(float(np.log(1.0 / initial_C)), dtype=torch.float32)
        )
        self.log_R = nn.Parameter(
            torch.tensor(float(np.log(initial_R)), dtype=torch.float32)
        )

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute [di_L/dt, dV_C/dt] from physical states and inputs."""
        i_L = x[..., 0]
        V_C = x[..., 1]
        aVin = u[..., 0]
        i_o = u[..., 1]

        inv_L = torch.exp(self.log_inv_L).to(device=x.device, dtype=x.dtype)
        inv_C = torch.exp(self.log_inv_C).to(device=x.device, dtype=x.dtype)
        R = torch.exp(self.log_R).to(device=x.device, dtype=x.dtype)

        diL_dt = inv_L * (aVin - R * i_L - V_C)
        dVC_dt = inv_C * (i_L - i_o)

        return torch.stack([diL_dt, dVC_dt], dim=-1)

    def get_params(self) -> dict:
        """Return identified physical parameters."""
        inv_L = float(torch.exp(self.log_inv_L).detach().cpu())
        inv_C = float(torch.exp(self.log_inv_C).detach().cpu())
        R = float(torch.exp(self.log_R).detach().cpu())
        return {
            "L": 1.0 / inv_L,
            "C": 1.0 / inv_C,
            "R": R,
            "inv_L": inv_L,
            "inv_C": inv_C,
        }


class SelfIdentifyingBuckPHNN(nn.Module):
    """Full Port-Hamiltonian buck converter in physical measurements.

    Trains directly on (i_L, V_C) measurements without prior knowledge of L or
    C. The model learns L and C as parameters of the coordinate transform from
    physical coordinates to PH energy coordinates (Phi, q), applies the PHNN
    dynamics in that space, then maps the derivatives back to physical units.
    It also identifies separate inductor-branch and capacitor ESR terms:
    R_L in the inductor branch, R_C through the capacitor terminal voltage, and
    a duty-scaled switching-cell voltage loss V_offset.

    The state-dependent delta_G_net is the neural component. It starts at zero
    so the initial model is pure physics, then learns diagonal input-coupling
    corrections only where the data demands them. The physical R_C term supplies
    the off-diagonal i_o -> dPhi/dt coupling implied by capacitor ESR.
    """

    def __init__(
        self,
        initial_L: float = 100e-6,
        initial_C: float = 100e-6,
        initial_R: float = 0.1,
        initial_R_C: float = 0.05,
        initial_V_offset: float = 0.5,
        r_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        self.log_inv_L = nn.Parameter(
            torch.tensor(float(np.log(1.0 / initial_L)), dtype=torch.float32)
        )
        self.log_inv_C = nn.Parameter(
            torch.tensor(float(np.log(1.0 / initial_C)), dtype=torch.float32)
        )
        self.log_R = nn.Parameter(
            torch.tensor(
                float(np.log(max(initial_R, r_epsilon))),
                dtype=torch.float32,
            )
        )
        self.log_R_C = nn.Parameter(
            torch.tensor(
                float(np.log(max(initial_R_C, 1e-4))),
                dtype=torch.float32,
            )
        )
        self.V_offset = nn.Parameter(
            torch.tensor(float(initial_V_offset), dtype=torch.float32)
        )

        self.register_buffer(
            "J",
            torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32),
        )
        self.register_buffer(
            "G_physics",
            torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32),
        )

        # Diagonal neural corrections only. The explicit R_C physics term
        # handles the off-diagonal i_o -> dPhi/dt ESR coupling.
        self.delta_G_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )
        nn.init.xavier_uniform_(self.delta_G_net[0].weight)
        nn.init.zeros_(self.delta_G_net[0].bias)
        nn.init.zeros_(self.delta_G_net[2].weight)
        nn.init.zeros_(self.delta_G_net[2].bias)

    def forward(
        self,
        x_phys: torch.Tensor,
        u_eff: torch.Tensor,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Return [di_L/dt, dV_C/dt] from physical states and inputs."""
        inv_L = torch.exp(self.log_inv_L).to(
            device=x_phys.device, dtype=x_phys.dtype
        )
        inv_C = torch.exp(self.log_inv_C).to(
            device=x_phys.device, dtype=x_phys.dtype
        )
        R_L = torch.exp(self.log_R).to(device=x_phys.device, dtype=x_phys.dtype)
        R_C = torch.exp(self.log_R_C).to(device=x_phys.device, dtype=x_phys.dtype)
        V_off = torch.clamp(self.V_offset, -2.0, 3.0).to(
            device=x_phys.device, dtype=x_phys.dtype
        )

        i_L = x_phys[..., 0]
        V_C = x_phys[..., 1]
        Phi = i_L / inv_L
        q = V_C / inv_C
        x_ph = torch.stack([Phi, q], dim=-1)
        u = u_eff.to(device=x_phys.device, dtype=x_phys.dtype)

        with torch.enable_grad():
            x_req = (
                x_ph
                if x_ph.requires_grad
                else x_ph.detach().clone().requires_grad_(True)
            )

            H = (
                0.5 * inv_L * x_req[..., 0:1].square()
                + 0.5 * inv_C * x_req[..., 1:2].square()
            )
            grad_H = torch.autograd.grad(
                H.sum(),
                x_req,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]

            i_L_h = grad_H[..., 0]
            V_C_h = grad_H[..., 1]
            aVin = u[..., 0]
            alpha = u[..., 1]
            i_o = u[..., 2]
            i_cap = i_L_h - i_o

            # Capacitor ESR appears as an extra terminal voltage drop in KVL:
            # V_terminal = V_C + R_C * i_cap.
            dPhi_dt = aVin - V_off * alpha - R_L * i_L_h - V_C_h - R_C * i_cap
            dq_dt = i_cap
            xdot_ph = torch.stack([dPhi_dt, dq_dt], dim=-1)

            delta_G_diag = self.delta_G_net(x_req)
            zeros = torch.zeros_like(delta_G_diag[..., 0])
            delta_G = torch.stack(
                [
                    torch.stack([delta_G_diag[..., 0], zeros], dim=-1),
                    torch.stack([zeros, delta_G_diag[..., 1]], dim=-1),
                ],
                dim=-2,
            )

            u_delta = torch.stack([aVin, i_o], dim=-1)
            delta_input = (delta_G @ u_delta.unsqueeze(-1)).squeeze(-1)
            xdot_ph = xdot_ph + delta_input

        diL_dt = inv_L * xdot_ph[..., 0]
        dVC_dt = inv_C * xdot_ph[..., 1]
        return torch.stack([diL_dt, dVC_dt], dim=-1)

    def get_params(self) -> dict:
        """Return identified physical parameters and effective input matrix."""
        with torch.no_grad():
            inv_L_t = torch.exp(self.log_inv_L)
            inv_C_t = torch.exp(self.log_inv_C)
            R_L_t = torch.exp(self.log_R)
            R_C_t = torch.exp(self.log_R_C)
            V_offset_t = torch.clamp(self.V_offset, -2.0, 3.0)
            inv_L = float(inv_L_t.detach().cpu())
            inv_C = float(inv_C_t.detach().cpu())
            R_L = float(R_L_t.detach().cpu())
            R_C = float(R_C_t.detach().cpu())
            V_offset = float(V_offset_t.detach().cpu())

            L_est = 1.0 / inv_L
            C_est = 1.0 / inv_C
            phi_vals = torch.linspace(-1.0 * L_est, 3.0 * L_est, 5)
            q_vals = torch.linspace(0.0, 10.0 * C_est, 5)
            phi_g, q_g = torch.meshgrid(phi_vals, q_vals, indexing="ij")
            pts = torch.stack([phi_g.flatten(), q_g.flatten()], dim=1).to(
                device=self.G_physics.device,
                dtype=self.G_physics.dtype,
            )
            delta_G_mean = self.delta_G_net(pts).mean(dim=0)
            G_eff = self.G_physics.cpu().clone()
            G_eff[0, 0] += delta_G_mean[0].item()
            G_eff[0, 1] += R_C
            G_eff[1, 1] += delta_G_mean[1].item()

        return {
            "L": 1.0 / inv_L,
            "C": 1.0 / inv_C,
            "R_L": R_L,
            "R_C": R_C,
            "R_eff": R_L + R_C,
            "V_offset": V_offset,
            "G_eff": G_eff,
            "delta_G_norm": float(delta_G_mean.norm()),
        }


class SimulationBuckPHNN(nn.Module):
    """Port-Hamiltonian buck converter for simulation accuracy, not parameter ID.

    L and C are fixed (known from design). The model learns effective resistances
    R_L and R_C plus a neural Hamiltonian correction H_NN(x, u_eff) that absorbs
    energy-storage non-idealities such as diode drop, switching losses, dead-time,
    and ESR variations directly in the Hamiltonian — the structurally correct place
    for energy-storage corrections.

    The total Hamiltonian is ``H = H_physics(x) + H_NN(x, u_eff)``, where
    ``H_physics`` is the fixed quadratic form and ``H_NN`` is a small MLP whose
    output layer is zero-initialized so training starts at the pure-physics solution.

    Inputs to forward():
        x_phys: [batch, 2]  physical states  [i_L [A], V_C [V]]
        u_eff:  [batch, 2]  effective inputs  [alpha*V_in [V], i_o [A]]

    Output:
        [batch, 2]  physical derivatives  [di_L/dt, dV_C/dt]
    """

    def __init__(
        self,
        L: float = 100e-6,
        C: float = 100e-6,
        initial_R_L: float = 0.15,
        initial_R_C: float = 0.05,
        r_epsilon: float = 1e-6,
    ) -> None:
        """Initialise fixed LC plant, learnable resistances, and H_NN correction.

        Args:
            L: Fixed inductance [H].
            C: Fixed capacitance [F].
            initial_R_L: Initial inductor series resistance guess [Ohm].
            initial_R_C: Initial capacitor ESR guess [Ohm].
            r_epsilon: Small positive floor added to R_L and R_C.
        """
        super().__init__()

        self.register_buffer("L", torch.tensor(L, dtype=torch.float32))
        self.register_buffer("C", torch.tensor(C, dtype=torch.float32))
        self.r_epsilon = float(r_epsilon)

        self.log_R_L = nn.Parameter(
            torch.tensor(float(np.log(max(initial_R_L, r_epsilon))), dtype=torch.float32)
        )
        self.log_R_C = nn.Parameter(
            torch.tensor(float(np.log(max(initial_R_C, r_epsilon))), dtype=torch.float32)
        )

        self.register_buffer(
            "J",
            torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32),
        )
        self.register_buffer(
            "G_physics",
            torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32),
        )

        # Typical operating scales used to normalise H_NN inputs.
        self.register_buffer("I_typical", torch.tensor(5.0, dtype=torch.float32))
        self.register_buffer("V_typical", torch.tensor(12.0, dtype=torch.float32))

        # Learnable Hamiltonian correction H_NN(Phi_norm, q_norm, alpha*V_in, i_o).
        # Final layer is zero-initialized so the model starts as pure physics.
        self.H_NN = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.H_NN[-1].weight)
        nn.init.zeros_(self.H_NN[-1].bias)

    def _h_nn(self, x_req: torch.Tensor, u_eff: torch.Tensor) -> torch.Tensor:
        """Evaluate the neural Hamiltonian correction H_NN(x, u_eff).

        Args:
            x_req: PH state tensor shape ``[..., 2]`` in ``(Phi, q)`` coordinates.
            u_eff: Input tensor shape ``[..., 2]`` containing
                ``[alpha * V_in, i_o]`` per sample.

        Returns:
            Scalar energy correction, shape ``[..., 1]``.
        """
        L = self.L.to(device=x_req.device, dtype=x_req.dtype)
        C = self.C.to(device=x_req.device, dtype=x_req.dtype)
        I_typ = self.I_typical.to(device=x_req.device, dtype=x_req.dtype)
        V_typ = self.V_typical.to(device=x_req.device, dtype=x_req.dtype)
        phi_norm = x_req[..., 0:1] / (L * I_typ)
        q_norm = x_req[..., 1:2] / (C * V_typ)
        inp = torch.cat([phi_norm, q_norm, u_eff], dim=-1)  # [..., 4]
        return self.H_NN(inp)  # [..., 1]

    def forward(
        self,
        x_phys: torch.Tensor,
        u_eff: torch.Tensor,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Return ``[di_L/dt, dV_C/dt]`` from physical states and effective inputs.

        Args:
            x_phys: State tensor shape ``[batch, 2]`` in ``[i_L, V_C]`` units.
            u_eff: Input tensor shape ``[batch, 2]`` containing
                ``[alpha * V_in, i_o]`` per sample.
            create_graph: Retain autograd graph for backprop through
                ``autograd.grad``.

        Returns:
            Predicted state derivative, shape ``[batch, 2]``.
        """
        L = self.L.to(device=x_phys.device, dtype=x_phys.dtype)
        C = self.C.to(device=x_phys.device, dtype=x_phys.dtype)
        R_L = (
            torch.exp(self.log_R_L).to(device=x_phys.device, dtype=x_phys.dtype)
            + self.r_epsilon
        )
        R_C = (
            torch.exp(self.log_R_C).to(device=x_phys.device, dtype=x_phys.dtype)
            + self.r_epsilon
        )

        i_L = x_phys[..., 0]
        V_C = x_phys[..., 1]
        Phi = L * i_L
        q = C * V_C
        x_ph = torch.stack([Phi, q], dim=-1)
        u = u_eff.to(device=x_phys.device, dtype=x_phys.dtype)

        with torch.enable_grad():
            x_req = (
                x_ph if x_ph.requires_grad else x_ph.detach().clone().requires_grad_(True)
            )

            # Total Hamiltonian: fixed quadratic physics + neural correction.
            H_phys = (
                0.5 / L * x_req[..., 0:1].square()
                + 0.5 / C * x_req[..., 1:2].square()
            )
            H_total = H_phys + self._h_nn(x_req, u)  # [batch, 1]

            grad_H = torch.autograd.grad(
                H_total.sum(),
                x_req,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]  # [batch, 2]

            R_mat = torch.diag(torch.stack([R_L, R_C]))
            J = self.J.to(device=x_req.device, dtype=x_req.dtype)
            auto_part = grad_H @ (J - R_mat).T  # [batch, 2]

            g_phys = self.G_physics.to(device=x_req.device, dtype=x_req.dtype)
            input_part = u @ g_phys.T  # [batch, 2]
            xdot_ph = auto_part + input_part

        di_L_dt = xdot_ph[..., 0] / L
        dV_C_dt = xdot_ph[..., 1] / C
        return torch.stack([di_L_dt, dV_C_dt], dim=-1)

    def get_params(self) -> dict:
        """Return fixed L, C, learned R_L, R_C, delta_G norm, and H_NN norm.

        Returns:
            Dict with keys ``'L'``, ``'C'``, ``'R_L'``, ``'R_C'``,
            ``'delta_G_norm'`` (always 0.0), and ``'H_NN_norm'``.
        """
        with torch.no_grad():
            L = float(self.L)
            C = float(self.C)
            x_rep = torch.tensor(
                [[L * 1.0, C * 6.0]], dtype=torch.float32, device=self.L.device
            )
            u_rep = torch.tensor(
                [[6.0, 0.5]], dtype=torch.float32, device=self.L.device
            )
            H_NN_norm = float(self._h_nn(x_rep, u_rep).abs())
            return {
                "L": L,
                "C": C,
                "R_L": float(torch.exp(self.log_R_L)),
                "R_C": float(torch.exp(self.log_R_C)),
                "delta_G_norm": 0.0,
                "H_NN_norm": H_NN_norm,
            }
