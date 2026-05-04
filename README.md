# Characterizing-Buck-Converter-with-Port-Hamiltonian-Neural-Network
Port-Hamiltonian Neural Network for data-driven system identification and control of DC-DC buck converters.
Buck Converter PHNN
A Port-Hamiltonian Neural Network for system identification and control design of DC-DC buck converters.
Overview
Traditional system identification for power converters relies on either accurate first-principles models (which require knowing component values precisely) or black-box neural networks (which lack physical interpretability and need large datasets). This project takes a middle path: a neural network constrained to the Port-Hamiltonian form
dx/dt = (J - R_θ(x)) ∇H_θ(x) + G_θ(x) u
where the network learns the Hamiltonian H_θ (stored energy), the dissipation matrix R_θ (resistive losses), and a small correction to the input map G_θ, while the skew-symmetric interconnection matrix J is fixed by physics.
Training on trajectory data, the model recovers physical parameters — L, C, R, and input coupling gains — that can be used to linearize the converter around an operating point, build a transfer function, and design a closed-loop controller.
Pipeline

Generate buck converter trajectories from an averaged ODE simulator
Train the PHNN on derivative-matching loss
Extract physical parameters from the trained model
Linearize around an operating point to get a state-space model (A, B, C, D)
Design a closed-loop controller against the identified plant

Status
Work in progress.
