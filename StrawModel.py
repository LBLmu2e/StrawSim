import numpy as np
import matplotlib.pyplot as plt

class StrawModel:
    def __init__(self, length_L=1.0, winding_angle_deg=37.0):
        """
        Initializes the straw tube detector with its physical dimensions and material constants.
        The dynamic, frequency-dependent skin depth corrections are calculated during execution.
        """
        self.length_L = length_L
        self.theta = np.radians(winding_angle_deg)

        # Material Resistivities (Ohm * m)
        self.rho_W = 5.60e-8    # Tungsten core
        self.rho_Au = 2.44e-8   # Gold cladding
        self.rho_Al = 2.65e-8   # Aluminum shield

        # Material Permeabilities (H / m)
        self.mu_0 = 4.0 * np.pi * 1e-7

        # Geometric Specifications (meters)
        self.r_W_core = 12.25e-6                         # Tungsten core radius (25um total wire diameter minus plating)
        self.r_inner_wire = 12.50e-6                     # Outer wire radius (including 0.25um gold plating)
        self.t_Au = 0.25e-6                              # Gold plating thickness
        self.D_out = 2.50e-3                             # Inner diameter of the tube
        self.t_Al = 0.10e-6                              # Aluminum coating thickness

        # Core Ideal Transmission Parameters (Static geometric capacitance and helical inductance)
        self.L_line = 976e-9                             # Inductance per unit length (H/m)
        self.C_line = 12.09e-12                          # Capacitance per unit length (F/m)
        self.G_line = 1e-9                               # Negligible shunt dielectric conductance (S/m)

    def _get_frequency_dependent_impedance(self, f):
        """
        Calculates the dynamic complex impedance parameters (Z0, gamma) for a given frequency,
        incorporating skin-depth corrections for composite inner and thin-film outer conductors.
        """
        if f == 0:
            # Fallback to pure DC limit calculations
            R_wire_dc = 1.0 / ((np.pi * self.r_W_core**2 / self.rho_W) +
                               (np.pi * (self.r_inner_wire**2 - self.r_W_core**2) / self.rho_Au))
            G_width_shield_dc = self.t_Al / self.rho_Al
            R_shield_dc = 1.0 / (G_width_shield_dc * np.pi * self.D_out * (np.cos(self.theta)**2))
            R_total = R_wire_dc + R_shield_dc

            Z0 = np.sqrt(R_total / (self.G_line if self.G_line > 0 else 1e-12))
            gamma = np.sqrt(R_total * self.G_line)
            return Z0, gamma

        w = 2.0 * np.pi * f

        # --- 1. Inner Conductor (Gold-clad Tungsten) skin effect ---
        delta_W = np.sqrt(self.rho_W / (np.pi * f * self.mu_0))
        delta_Au = np.sqrt(self.rho_Au / (np.pi * f * self.mu_0))

        # Effective conducting area of the gold cladding layer
        # If the skin depth is smaller than the cladding layer, current is restricted entirely to the gold shell
        if delta_Au < self.t_Au:
            area_Au_eff = np.pi * (self.r_inner_wire**2 - (self.r_inner_wire - delta_Au)**2)
            R_wire = self.rho_Au / area_Au_eff
        else:
            # Current penetrates completely through the gold shell and into the tungsten core
            area_Au_eff = np.pi * (self.r_inner_wire**2 - self.r_W_core**2)
            R_Au_part = self.rho_Au / area_Au_eff

            # Remaining skin depth depth assigned into the tungsten core
            delta_W_eff = delta_Au - self.t_Au
            if delta_W_eff > self.r_W_core:
                delta_W_eff = self.r_W_core
            area_W_eff = np.pi * (self.r_W_core**2 - (self.r_W_core - delta_W_eff)**2)
            R_W_part = self.rho_W / max(area_W_eff, 1e-20)

            # Parallel resistance combination
            R_wire = 1.0 / ((1.0 / R_Au_part) + (1.0 / R_W_part))

        # --- 2. Outer Conductor (Thin-Film Spiral Aluminum) skin effect ---
        delta_Al = np.sqrt(self.rho_Al / (np.pi * f * self.mu_0))

        # Bound effective conduction thickness by the physical film limits (0.1 um)
        t_Al_eff = min(delta_Al, self.t_Al)
        G_width = t_Al_eff / self.rho_Al
        # Incorporate the geometric 1/cos^2 escalation for the helical pitch path
        R_shield = 1.0 / (G_width * np.pi * self.D_out * (np.cos(self.theta)**2))

        # Combined dynamic loop resistance
        R_dynamic = R_wire + R_shield

        # High-frequency skin inductance contribution (L_internal ≈ R_internal / \omega)
        L_dynamic = self.L_line + (R_dynamic / w)

        # Heaviside transmission line solution
        Z0 = np.sqrt((R_dynamic + 1j * w * L_dynamic) / (self.G_line + 1j * w * self.C_line))
        gamma = np.sqrt((R_dynamic + 1j * w * L_dynamic) * (self.G_line + 1j * w * self.C_line))

        return Z0, gamma

    def compute_response(
            self,
            inject_l,
            freq_max_mhz=15000.0,
            num_points=16384,
            input_time=None,
            input_current=None,
            Z_termination_A=50.0,
            Z_termination_B=50.0
            ):
        """
        Computes the real-time voltage waveforms using the skin-effect corrected transmission matrix.
        """
        f_max = freq_max_mhz * 1e6
        dt = 1.0 / (2.0 * f_max)
        time_axis = np.arange(num_points) * dt

        freqs = np.fft.rfftfreq(num_points, d=dt)

        if input_current is None or input_time is None:
            # Default ultra-narrow 30 ps RMS Gaussian current pulse
            t0 = 2e-9
            sigma = 30e-12
            i_t = 100e-6 * np.exp(-0.5 * ((time_axis - t0) / sigma) ** 2)
        else:
            i_t = np.interp(time_axis, input_time, input_current, left=0.0, right=0.0)

        I_omega = np.fft.rfft(i_t)
        V_A_omega = np.zeros_like(I_omega, dtype=complex)
        V_B_omega = np.zeros_like(I_omega, dtype=complex)

        for idx, f in enumerate(freqs):
            # Fetch dynamic parameters for the current frequency index
            Z0, gamma = self._get_frequency_dependent_impedance(f)

            Gamma_A = (Z_termination_A - Z0) / (Z_termination_A + Z0)
            Gamma_B = (Z_termination_B - Z0) / (Z_termination_B + Z0)

            x_A = inject_l
            x_B = self.length_L - inject_l

            V_fwd_initial = (I_omega[idx] * Z0) / 2.0
            V_bwd_initial = (I_omega[idx] * Z0) / 2.0

            V_A_direct = V_bwd_initial * np.exp(-gamma * x_A)
            V_B_direct = V_fwd_initial * np.exp(-gamma * x_B)

            loop_factor = 1.0 - Gamma_A * Gamma_B * np.exp(-2.0 * gamma * self.length_L)

            V_A_omega[idx] = (V_A_direct * (1.0 + Gamma_A) + V_B_direct * Gamma_B * np.exp(-gamma * self.length_L) * (1.0 + Gamma_A)) / loop_factor
            V_B_omega[idx] = (V_B_direct * (1.0 + Gamma_B) + V_A_direct * Gamma_A * np.exp(-gamma * self.length_L) * (1.0 + Gamma_B)) / loop_factor

        v_A = np.fft.irfft(V_A_omega, n=num_points)
        v_B = np.fft.irfft(V_B_omega, n=num_points)

        return time_axis, v_A, v_B

    def plot_comparison(self, inject_l):
        """Plots standard 50 Ohm electronic matching versus an updated dynamic matched load."""
        t, v_A_50, v_B_50 = self.compute_response(
                inject_l=inject_l, Z_termination_A=50.0, Z_termination_B=50.0
                )

        # High frequency asymptotic impedance approaches roughly 284.5 Ohms
        Z_match = 284.50
        _, v_A_match, v_B_match = self.compute_response(
                inject_l=inject_l, Z_termination_A=Z_match, Z_termination_B=Z_match
                )

        t_ns = t * 1e9
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        # Mismatched Plot
        ax1.plot(t_ns, v_A_50 * 1e3, label=f'End A (x = 0m, Dist = {inject_l}m)', color='#1f77b4', linewidth=1.8)
        ax1.plot(t_ns, v_B_50 * 1e3, label=f'End B (x = {self.length_L}m, Dist = {self.length_L-inject_l:.2f}m)', color='#ff7f0e', linewidth=1.8)
        ax1.set_title('Mismatched Termination (Standard 50 Ω Electronic Readout Layout)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Voltage (mV)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        # Matched Plot
        ax2.plot(t_ns, v_A_match * 1e3, label='End A (Matched)', color='#2ca02c', linewidth=1.8)
        ax2.plot(t_ns, v_B_match * 1e3, label='End B (Matched)', color='#d62728', linewidth=1.8)
        ax2.set_title(f'Perfectly Matched High-Frequency Termination (Z_load = {Z_match} Ω)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time (ns)', fontsize=11)
        ax2.set_ylabel('Voltage (mV)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')

        plt.suptitle(f'Straw Tube Pulse Response WITH Dynamic Skin Effect Corrections\n(Injection Site: {inject_l}m | Total Length: {self.length_L}m)', fontsize=13, fontweight='bold', y=0.96)
        plt.xlim(0, 15)
        plt.tight_layout()
        plt.show()

