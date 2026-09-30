import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk


def calculate_sensor_spread(theta_deg, valid_mask, focal_length_mm):
    sensor_positions_mm = np.full(theta_deg.shape, np.nan)
    valid_angles = theta_deg[valid_mask]

    if focal_length_mm <= 0:
        raise ValueError("Tube lens focal length must be positive")
    if valid_angles.size == 0:
        return sensor_positions_mm, 0.0

    reference_angle_deg = (np.min(valid_angles) + np.max(valid_angles)) / 2
    angle_difference_rad = np.radians(valid_angles - reference_angle_deg)
    sensor_positions_mm[valid_mask] = focal_length_mm * np.tan(angle_difference_rad)
    spread_mm = np.ptp(sensor_positions_mm[valid_mask])
    return sensor_positions_mm, spread_mm


class GratingSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Grating Diffraction & Efficiency Simulator")
        self.root.geometry("1180x860")
        self.root.minsize(960, 700)

        self.colors = {
            "background": "#F2F5F2",
            "surface": "#FFFFFF",
            "text": "#20302D",
            "muted": "#687873",
            "border": "#D7E0DC",
            "teal": "#147D78",
            "coral": "#D76B53",
            "blue": "#4779A8",
            "grid": "#E5EBE8",
        }
        self.spread_text = tk.StringVar(value="-- mm")
        self.configure_style()

        # Default parameters
        self.lines_per_mm = 600.0
        self.blaze_wavelength = 500.0  # nm
        self.incidence_angle_deg = 30.0
        self.order = 1
        self.tube_lens_focal_length_mm = 200.0
        self.wavelengths = np.linspace(380, 750, 100)  # Visible spectrum

        # Create UI
        self.create_widgets()
        
        # Initial plot
        self.update_plot()

    def configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=self.colors["background"])
        style.configure("TLabel", background=self.colors["background"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"), foreground=self.colors["text"])
        style.configure("Eyebrow.TLabel", font=("Segoe UI", 9, "bold"), foreground=self.colors["teal"])
        style.configure("Metric.TLabel", font=("Segoe UI", 19, "bold"), foreground=self.colors["teal"])
        style.configure("MetricCaption.TLabel", font=("Segoe UI", 9, "bold"), foreground=self.colors["muted"])
        style.configure("TLabelframe", background=self.colors["background"], bordercolor=self.colors["border"])
        style.configure("TLabelframe.Label", background=self.colors["background"], foreground=self.colors["teal"], font=("Segoe UI", 9, "bold"))
        style.configure("TEntry", fieldbackground=self.colors["surface"], foreground=self.colors["text"], padding=(8, 7))
        style.configure("TCombobox", fieldbackground=self.colors["surface"], foreground=self.colors["text"], padding=(8, 6))

    def create_widgets(self):
        app_frame = ttk.Frame(self.root, padding=(22, 18))
        app_frame.grid(row=0, column=0, sticky="nsew")
        app_frame.columnconfigure(1, weight=1)
        app_frame.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        header = ttk.Frame(app_frame)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="OPTICAL MODEL / 01", style="Eyebrow.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(header, text="Grating simulator", style="Title.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        metric = ttk.Frame(header)
        metric.grid(row=0, column=2, rowspan=2, sticky="e", padx=(24, 0))
        ttk.Label(metric, text="CAMERA SPREAD", style="MetricCaption.TLabel").pack(anchor="e")
        ttk.Label(metric, textvariable=self.spread_text, style="Metric.TLabel").pack(anchor="e", pady=(1, 0))

        control_frame = ttk.Frame(app_frame, padding=(0, 0, 18, 0))
        control_frame.grid(row=1, column=0, sticky="new")
        control_frame.columnconfigure(0, weight=1)
        ttk.Label(control_frame, text="MODEL INPUTS", style="Eyebrow.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))

        grating_frame = ttk.LabelFrame(control_frame, text="GRATING", padding=(13, 10))
        grating_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        grating_frame.columnconfigure(1, weight=1)

        ttk.Label(grating_frame, text="Lines per mm").grid(row=0, column=0, sticky="w", pady=5, padx=(0, 12))
        self.entry_lines = ttk.Entry(grating_frame, width=12, justify="right")
        self.entry_lines.insert(0, str(self.lines_per_mm))
        self.entry_lines.grid(row=0, column=1, sticky="ew", pady=5)
        self.entry_lines.bind('<KeyRelease>', lambda e: self.update_plot())

        ttk.Label(grating_frame, text="Blaze (nm)").grid(row=1, column=0, sticky="w", pady=5, padx=(0, 12))
        self.entry_blaze = ttk.Entry(grating_frame, width=12, justify="right")
        self.entry_blaze.insert(0, str(self.blaze_wavelength))
        self.entry_blaze.grid(row=1, column=1, sticky="ew", pady=5)
        self.entry_blaze.bind('<KeyRelease>', lambda e: self.update_plot())

        ttk.Label(grating_frame, text="Incidence (°)").grid(row=2, column=0, sticky="w", pady=5, padx=(0, 12))
        self.entry_angle = ttk.Entry(grating_frame, width=12, justify="right")
        self.entry_angle.insert(0, str(self.incidence_angle_deg))
        self.entry_angle.grid(row=2, column=1, sticky="ew", pady=5)
        self.entry_angle.bind('<KeyRelease>', lambda e: self.update_plot())

        ttk.Label(grating_frame, text="Order (m)").grid(row=3, column=0, sticky="w", pady=5, padx=(0, 12))
        self.combo_order = ttk.Combobox(grating_frame, values=[-2, -1, 0, 1, 2], state='readonly', justify="right")
        self.combo_order.set(str(self.order))
        self.combo_order.grid(row=3, column=1, sticky="ew", pady=5)
        self.combo_order.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        lens_frame = ttk.LabelFrame(control_frame, text="IMAGING", padding=(13, 10))
        lens_frame.grid(row=2, column=0, sticky="ew")
        lens_frame.columnconfigure(1, weight=1)
        ttk.Label(lens_frame, text="Tube lens (mm)").grid(row=0, column=0, sticky="w", pady=5, padx=(0, 12))
        self.entry_focal_length = ttk.Entry(lens_frame, width=12, justify="right")
        self.entry_focal_length.insert(0, str(self.tube_lens_focal_length_mm))
        self.entry_focal_length.grid(row=0, column=1, sticky="ew", pady=5)
        self.entry_focal_length.bind('<KeyRelease>', lambda e: self.update_plot())

        plot_frame = ttk.Frame(app_frame)
        plot_frame.grid(row=1, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig = plt.figure(figsize=(10, 7), dpi=100, facecolor=self.colors["background"])
        grid = self.fig.add_gridspec(2, 2, hspace=0.42, wspace=0.3, height_ratios=(1, 1))
        self.ax1 = self.fig.add_subplot(grid[0, 0])
        self.ax2 = self.fig.add_subplot(grid[0, 1])
        self.ax3 = self.fig.add_subplot(grid[1, :])
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.09)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

    def style_axis(self, axis):
        axis.set_facecolor(self.colors["surface"])
        axis.grid(True, color=self.colors["grid"], linewidth=0.8)
        axis.tick_params(colors=self.colors["muted"], labelsize=9)
        axis.title.set_color(self.colors["text"])
        axis.title.set_fontsize(10)
        axis.title.set_fontweight("bold")
        axis.xaxis.label.set_color(self.colors["muted"])
        axis.yaxis.label.set_color(self.colors["muted"])
        for spine in axis.spines.values():
            spine.set_color(self.colors["border"])

    def update_plot(self):
        try:
            # Get inputs
            self.lines_per_mm = float(self.entry_lines.get())
            self.blaze_wavelength = float(self.entry_blaze.get())
            self.incidence_angle_deg = float(self.entry_angle.get())
            self.order = int(self.combo_order.get())
            self.tube_lens_focal_length_mm = float(self.entry_focal_length.get())
            
            # Constants
            c = 299792458  # Speed of light (m/s)
            lambda_m = self.wavelengths * 1e-9  # Convert nm to meters
            d = 1.0 / (self.lines_per_mm * 1000)  # Grating spacing in meters (1/mm -> 1/m)
            
            # 1. Calculate Diffraction Angle
            sin_theta_i = np.sin(np.radians(self.incidence_angle_deg))
            sin_theta_m = (self.order * lambda_m / d) - sin_theta_i
            
            # Check for valid angles (sin must be between -1 and 1)
            valid_mask = (sin_theta_m >= -1) & (sin_theta_m <= 1)
            theta_m_rad = np.arcsin(sin_theta_m)
            theta_m_deg = np.degrees(theta_m_rad)
            sensor_positions_mm, sensor_spread_mm = calculate_sensor_spread(
                theta_m_deg, valid_mask, self.tube_lens_focal_length_mm
            )
            self.spread_text.set(f"{sensor_spread_mm:.3f} mm")
            
            # Clear first axis
            self.ax1.clear()
            self.ax1.plot(
                self.wavelengths[valid_mask], theta_m_deg[valid_mask],
                color=self.colors["blue"], linewidth=2.2, label=f'Order m={self.order}'
            )
            self.ax1.set_title('Diffraction angle')
            self.ax1.set_xlabel('Wavelength (nm)')
            self.ax1.set_ylabel('Angle (°)')
            self.ax1.axvspan(380, 750, color=self.colors["teal"], alpha=0.035, zorder=0)
            self.ax1.legend(frameon=False, loc="best", fontsize=8)

            # 2. Calculate Efficiency (Simplified Blaze Model)
            # This is a Gaussian approximation centered at the blaze wavelength
            # Real efficiency requires RCWA simulation, but this gives a visual idea
            sigma = 50.0  # Width of the blaze curve
            efficiency = np.exp(-0.5 * ((self.wavelengths - self.blaze_wavelength) / sigma) ** 2)
            
            # Normalize to 100%
            efficiency = efficiency / np.max(efficiency) * 100
            
            # Clear second axis
            self.ax2.clear()
            self.ax2.plot(self.wavelengths, efficiency, color=self.colors["coral"], linewidth=2.2)
            self.ax2.axvline(self.blaze_wavelength, color=self.colors["coral"], linestyle="--", linewidth=1, alpha=0.65)
            self.ax2.set_title('Blaze efficiency')
            self.ax2.set_xlabel('Wavelength (nm)')
            self.ax2.set_ylabel('Efficiency (%)')
            self.ax2.set_ylim(0, 105)

            # 3. Map angular differences to positions at the tube lens focal plane
            self.ax3.clear()
            self.ax3.plot(
                self.wavelengths[valid_mask], sensor_positions_mm[valid_mask],
                color=self.colors["teal"], linewidth=2.2, marker="o", markevery=8, markersize=3
            )
            self.ax3.axhline(0, color=self.colors["muted"], linewidth=0.9, linestyle="--", alpha=0.65)
            self.ax3.set_title('Camera sensor position')
            self.ax3.set_xlabel('Wavelength (nm)')
            self.ax3.set_ylabel('Position on sensor (mm)')

            for axis in (self.ax1, self.ax2, self.ax3):
                self.style_axis(axis)

            # Redraw
            self.canvas.draw()
            
        except ValueError:
            pass  # Ignore invalid input while typing

if __name__ == "__main__":
    root = tk.Tk()
    app = GratingSimulator(root)
    root.mainloop()