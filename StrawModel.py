import numpy as np
import matplotlib.pyplot as plt
import os

class StrawModel:
    def __init__(self, length_L=1.0, winding_angle_deg=37.0, Z_termination=50.0):
        """
        Initializes the straw tube detector with its physical properties, material dimensions,
        and a symmetric termination impedance applied uniformly to both ends.
        """
        self.length_L = length_L
        self.theta = np.radians(winding_angle_deg)
        self.Z_termination = Z_termination

        # Material Resistivities (Ohm * m)
        self.rho_W = 5.60e-8
        self.rho_Au = 2.44e-8
        self.rho_Al = 2.65e-8

        # Material Permeabilities (H / m)
        self.mu_0 = 4.0 * np.pi * 1e-7

        # Geometric Specifications (meters)
        self.r_W_core = 12.25e-6
        self.r_inner_wire = 12.50e-6
        self.t_Au = 0.25e-6
        self.D_out = 2.50e-3

        # Outer Shield: 2 layers, cross-wound (0.05 um per layer)
        self.t_Al_total = 0.10e-6
        self.t_Al_layer = self.t_Al_total / 2.0

        # Core Transmission Line Parameters (Helical Inductance Cancelled)
        self.L_line = 921e-9
        self.C_line = 12.09e-12
        self.G_line = 1e-9

    def _get_frequency_dependent_impedance(self, f):
        """Calculates dynamic Z0 and gamma incorporating skin-depth and layout parameters."""
        if f <= 1e-3:
            R_wire_dc = 1.0 / ((np.pi * self.r_W_core**2 / self.rho_W) +
                               (np.pi * (self.r_inner_wire**2 - self.r_W_core**2) / self.rho_Au))
            G_width_shield_dc = (self.t_Al_layer / self.rho_Al) * 2.0
            R_shield_dc = 1.0 / (G_width_shield_dc * np.pi * self.D_out * (np.cos(self.theta)**2))
            R_total = R_wire_dc + R_shield_dc

            Z0 = np.sqrt(R_total / (self.G_line if self.G_line > 0 else 1e-12))
            gamma = np.sqrt(R_total * self.G_line)
            return Z0, gamma

        w = 2.0 * np.pi * f
        delta_Au = np.sqrt(self.rho_Au / (np.pi * f * self.mu_0))

        if delta_Au < self.t_Au:
            area_Au_eff = np.pi * (self.r_inner_wire**2 - (self.r_inner_wire - delta_Au)**2)
            R_wire = self.rho_Au / area_Au_eff
        else:
            area_Au_eff = np.pi * (self.r_inner_wire**2 - self.r_W_core**2)
            R_Au_part = self.rho_Au / area_Au_eff
            delta_W_eff = delta_Au - self.t_Au
            if delta_W_eff > self.r_W_core:
                delta_W_eff = self.r_W_core
            area_W_eff = np.pi * (self.r_W_core**2 - (self.r_W_core - delta_W_eff)**2)
            R_W_part = self.rho_W / max(area_W_eff, 1e-20)
            R_wire = 1.0 / ((1.0 / R_Au_part) + (1.0 / R_W_part))

        delta_Al = np.sqrt(self.rho_Al / (np.pi * f * self.mu_0))
        t_Al_eff = min(delta_Al, self.t_Al_total)
        G_width = t_Al_eff / self.rho_Al
        R_shield = 1.0 / (G_width * np.pi * self.D_out * (np.cos(self.theta)**2))

        R_dynamic = R_wire + R_shield

        L_internal_max = 0.05 * self.L_line
        L_internal = min(R_dynamic / w, L_internal_max)
        L_dynamic = self.L_line + L_internal

        Z0 = np.sqrt((R_dynamic + 1j * w * L_dynamic) / (self.G_line + 1j * w * self.C_line))
        gamma = np.sqrt((R_dynamic + 1j * w * L_dynamic) * (self.G_line + 1j * w * self.C_line))
        return Z0, gamma

    def _compute_single_end_response(self, x_near, x_far, freqs, I_omega):
        """
        Low-level primitive that solves the frequency-domain boundary equation for a
        single readout end located x_near meters away from the source injection point.
        """
        V_end_omega = np.zeros_like(I_omega, dtype=complex)

        for idx, f in enumerate(freqs):
            Z0, gamma = self._get_frequency_dependent_impedance(f)
            Gamma = (self.Z_termination - Z0) / (self.Z_termination + Z0)

            # Initial voltage wave injection magnitude
            V_initial = (I_omega[idx] * Z0) / 2.0

            # Infinite boundary round-trip loop lattice denominator
            loop_factor = 1.0 - (Gamma ** 2) * np.exp(-2.0 * gamma * self.length_L)

            # Wave components arriving at this specific near end:
            # 1. The forward/backward wave propagating directly over x_near
            # 2. The opposite wave traveling x_far, bouncing off the far end, and returning over the full length L
            V_direct = V_initial * np.exp(-gamma * x_near)
            V_reflected_far = V_initial * np.exp(-gamma * x_far) * Gamma * np.exp(-gamma * self.length_L)

            # Total voltage including near-end reflection scaling: V_total = V_incident * (1 + Gamma)
            V_end_omega[idx] = ((V_direct + V_reflected_far) * (1.0 + Gamma)) / loop_factor

        return V_end_omega

    def compute_response(
        self,
        inject_l,
        freq_max_mhz=15000.0,
        num_points=16384,
        input_time=None,
        input_current=None,
        default_sigma=30e-12
    ):
        """
        Computes symmetric dual-ended voltage waveforms by strictly mirroring the
        spatial argument constraints passed to the single-end processing primitive.
        """
        f_max = freq_max_mhz * 1e6
        dt = 1.0 / (2.0 * f_max)
        time_axis = np.arange(num_points) * dt
        freqs = np.fft.rfftfreq(num_points, d=dt)

        if input_current is None or input_time is None:
            t0 = 2e-9
            i_t = 100e-6 * np.exp(-0.5 * ((time_axis - t0) / default_sigma) ** 2)
        else:
            i_t = np.interp(time_axis, input_time, input_current, left=0.0, right=0.0)

        I_omega = np.fft.rfft(i_t)

        # Spatial constraints relative to End A and End B
        x_A = inject_l
        x_B = self.length_L - inject_l

        # Symmetric by construction: swapping arguments guarantees completely mirrored behavior
        V_A_omega = self._compute_single_end_response(x_near=x_A, x_far=x_B, freqs=freqs, I_omega=I_omega)
        V_B_omega = self._compute_single_end_response(x_near=x_B, x_far=x_A, freqs=freqs, I_omega=I_omega)

        v_A = np.fft.irfft(V_A_omega, n=num_points)
        v_B = np.fft.irfft(V_B_omega, n=num_points)
        return time_axis, v_A, v_B

    def plot_response(self, inject_l, sigma=30e-12, time_window_ns=15.0):
        """Plots the response of the straw tube to an analytical Gaussian current pulse."""
        f_max_mhz = max(15000.0, 0.5 / (sigma * 1e6) if sigma > 0 else 15000.0)
        t, v_A, v_B = self.compute_response(inject_l=inject_l, default_sigma=sigma, freq_max_mhz=f_max_mhz)
        t_ns = t * 1e9

        plt.figure(figsize=(10, 5))
        plt.plot(t_ns, v_A * 1e3, label=f'End A (x = 0m)', color='#1f77b4', linewidth=1.8)
        plt.plot(t_ns, v_B * 1e3, label=f'End B (x = {self.length_L}m)', color='#ff7f0e', linewidth=1.8, linestyle='--')

        plt.title(f'Symmetric Straw Tube Response to Gaussian Pulse (σ = {sigma*1e12:.1f} ps)\nInjection Site: {inject_l}m from End A', fontsize=11, fontweight='bold')
        plt.xlabel('Time (ns)', fontsize=11)
        plt.ylabel('Recorded Voltage (mV)', fontsize=11)
        plt.xlim(0, time_window_ns)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def compute_response_from_pwle(self, filepath, inject_l, **kwargs):
        """Loads a piece-wise linear current profile file and runs the signal engine."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The target StrawSim PWLE file could not be found at: {filepath}")
        try:
            data = np.loadtxt(filepath, comments=['#', '//', 'A', 't', 'T'])
        except Exception:
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    clean = line.strip().replace(',', ' ')
                    if clean and not clean.isalpha():
                        parts = clean.split()
                        if len(parts) >= 2:
                            data.append([float(parts), float(parts)])
            data = np.array(data)

        if data.ndim != 2 or data.shape < 2:
            raise ValueError("Malformed file format. File must contain at least 2 numeric columns.")

        pwle_time = data[:, 0]
        pwle_current = data[:, 1]

        t, v_A, v_B = self.compute_response(
            inject_l=inject_l, input_time=pwle_time, input_current=pwle_current, **kwargs
        )
        return t, v_A, v_B, pwle_time, pwle_current

    def plot_pwle_response(self, filepath, inject_l, time_window_ns=45.0, freq_max_mhz=15000.0, num_points=16384):
        """Parses a target PWLE current file, simulates transmissions, and plots stacked axes."""
        t, v_A, v_B, source_t, source_i = self.compute_response_from_pwle(
            filepath=filepath, inject_l=inject_l, freq_max_mhz=freq_max_mhz, num_points=num_points
        )
        t_ns = t * 1e9
        source_t_ns = source_t * 1e9
        source_i_ua = source_i * 1e6
        v_A_mv = v_A * 1e3
        v_B_mv = v_B * 1e3

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        filename = os.path.basename(filepath)

        ax1.plot(source_t_ns, source_i_ua, color='#7f7f7f', linestyle='-', linewidth=2.0, marker='.', markersize=4, label='Injected Cluster Pulse')
        ax1.set_title(f'Source Input: Injected Current Profile ({filename})', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Injected Current (μA)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        ax2.plot(t_ns, v_A_mv, label=f'End A (x = 0m, Dist = {inject_l}m)', color='#1f77b4', linewidth=1.8)
        ax2.plot(t_ns, v_B_mv, label=f'End B (x = {self.length_L}m, Dist = {self.length_L-inject_l:.2f}m)', color='#ff7f0e', linewidth=1.8)
        ax2.set_title(f'Output Response (Termination Load: Z_term = {self.Z_termination} Ω)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time (ns)', fontsize=11)

        plt.xlim(0, time_window_ns)

