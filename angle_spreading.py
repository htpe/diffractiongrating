import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


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


class GratingSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grating Diffraction & Efficiency Simulator")
        self.resize(1180, 860)
        self.setMinimumSize(960, 700)

        self.colors = {
            "background": "#F2F5F2",
            "surface": "#FFFFFF",
            "text": "#20302D",
            "muted": "#4B5552",
            "border": "#D7E0DC",
            "teal": "#147D78",
            "coral": "#D76B53",
            "blue": "#4779A8",
            "grid": "#E5EBE8",
        }
        self.spread_text = "-- mm"

        # Default parameters
        self.lines_per_mm = 600.0
        self.blaze_wavelength = 500.0  # nm
        self.incidence_angle_deg = 30.0
        self.order = 1
        self.tube_lens_focal_length_mm = 200.0
        self.wavelengths = np.linspace(380, 750, 100)  # Visible spectrum
        self.sensor_annotation = None

        # Create UI
        self.create_widgets()
        
        # Initial plot
        self.update_plot()

        self.setStyleSheet("""
            QMainWindow, QWidget#appRoot {
                background: #F2F5F2;
                color: #20302D;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QLabel {
                color: #4B5552;
            }
            QLabel#eyebrow {
                color: #4B5552;
                font-size: 9pt;
                font-weight: 700;
            }
            QLabel#pageTitle {
                color: #20302D;
                font-size: 22pt;
                font-weight: 700;
            }
            QLabel#metricCaption {
                color: #4B5552;
                font-size: 9pt;
                font-weight: 700;
            }
            QLabel#metricValue {
                color: #4B5552;
                font-size: 19pt;
                font-weight: 700;
            }
            QLabel#metricHint {
                color: #4B5552;
                font-size: 9pt;
            }
            QGroupBox {
                border: 1px solid #D7E0DC;
                border-radius: 7px;
                margin-top: 12px;
                padding: 14px 12px 12px;
                color: #4B5552;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
            QLineEdit, QComboBox {
                background: #FFFFFF;
                border: 1px solid #D7E0DC;
                border-radius: 5px;
                padding: 7px 8px;
                min-height: 20px;
                color: #20302D;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #147D78;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #4B5552;
                selection-background-color: #D7E0DC;
                selection-color: #20302D;
                outline: 0;
            }
            QStatusBar {
                background: #F2F5F2;
                color: #4B5552;
            }
        """)

    def create_widgets(self):
        app_frame = QWidget()
        app_frame.setObjectName("appRoot")
        app_layout = QVBoxLayout(app_frame)
        app_layout.setContentsMargins(22, 18, 22, 16)
        app_layout.setSpacing(18)
        self.setCentralWidget(app_frame)

        header = QHBoxLayout()
        title_stack = QVBoxLayout()
        eyebrow = QLabel("OPTICAL MODEL / 01")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Grating simulator")
        title.setObjectName("pageTitle")
        title_stack.addWidget(eyebrow)
        title_stack.addWidget(title)
        header.addLayout(title_stack)
        header.addStretch(1)

        metric_stack = QVBoxLayout()
        metric_stack.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        metric_caption = QLabel("CAMERA SPREAD")
        metric_caption.setObjectName("metricCaption")
        self.spread_value = QLabel(self.spread_text)
        self.spread_value.setObjectName("metricValue")
        self.derivative_value = QLabel("Click sensor curve to inspect slope")
        self.derivative_value.setObjectName("metricHint")
        metric_stack.addWidget(metric_caption, alignment=Qt.AlignmentFlag.AlignRight)
        metric_stack.addWidget(self.spread_value, alignment=Qt.AlignmentFlag.AlignRight)
        metric_stack.addWidget(self.derivative_value, alignment=Qt.AlignmentFlag.AlignRight)
        header.addLayout(metric_stack)
        app_layout.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(16)
        app_layout.addLayout(content, stretch=1)

        control_frame = QWidget()
        control_frame.setFixedWidth(275)
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(10)
        inputs_heading = QLabel("MODEL INPUTS")
        inputs_heading.setObjectName("eyebrow")
        control_layout.addWidget(inputs_heading)

        grating_frame = QGroupBox("GRATING")
        grating_form = QFormLayout(grating_frame)
        grating_form.setContentsMargins(12, 18, 12, 12)
        grating_form.setSpacing(10)
        self.entry_lines = self.create_numeric_input(self.lines_per_mm, 0.001, 100000)
        self.entry_blaze = self.create_numeric_input(self.blaze_wavelength, 1, 1000000)
        self.entry_angle = self.create_numeric_input(self.incidence_angle_deg, -89.9, 89.9)
        grating_form.addRow("Lines per mm", self.entry_lines)
        grating_form.addRow("Blaze (nm)", self.entry_blaze)
        grating_form.addRow("Incidence (°)", self.entry_angle)

        self.combo_order = QComboBox()
        self.combo_order.addItems(["-2", "-1", "0", "1", "2"])
        self.combo_order.setCurrentText(str(self.order))
        self.combo_order.currentTextChanged.connect(lambda _value: self.update_plot())
        grating_form.addRow("Order (m)", self.combo_order)
        control_layout.addWidget(grating_frame)

        lens_frame = QGroupBox("IMAGING")
        lens_form = QFormLayout(lens_frame)
        lens_form.setContentsMargins(12, 18, 12, 12)
        lens_form.setSpacing(10)
        self.entry_focal_length = self.create_numeric_input(self.tube_lens_focal_length_mm, 0.001, 100000)
        lens_form.addRow("Tube lens (mm)", self.entry_focal_length)
        control_layout.addWidget(lens_frame)
        control_layout.addStretch(1)
        content.addWidget(control_frame)

        plot_frame = QWidget()
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)
        content.addWidget(plot_frame, stretch=1)

        self.fig = plt.figure(figsize=(10, 7), dpi=100, facecolor=self.colors["background"])
        grid = self.fig.add_gridspec(2, 2, hspace=0.42, wspace=0.3, height_ratios=(1, 1))
        self.ax1 = self.fig.add_subplot(grid[0, 0])
        self.ax2 = self.fig.add_subplot(grid[0, 1])
        self.ax3 = self.fig.add_subplot(grid[1, :])
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.09)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.canvas, stretch=1)
        plot_layout.addWidget(self.toolbar)
        self.canvas.mpl_connect("pick_event", self.show_sensor_derivative)
        self.canvas.mpl_connect("scroll_event", self.zoom_plot)

    def create_numeric_input(self, value, minimum, maximum):
        field = QLineEdit(str(value))
        field.setAlignment(Qt.AlignmentFlag.AlignRight)
        field.setValidator(QDoubleValidator(minimum, maximum, 4, field))
        field.textChanged.connect(lambda _value: self.update_plot())
        return field

    def style_axis(self, axis):
        axis.set_facecolor(self.colors["surface"])
        axis.grid(True, color=self.colors["grid"], linewidth=0.8)
        axis.tick_params(colors=self.colors["muted"], labelsize=9)
        axis.title.set_color(self.colors["muted"])
        axis.title.set_fontsize(10)
        axis.title.set_fontweight("bold")
        axis.xaxis.label.set_color(self.colors["muted"])
        axis.yaxis.label.set_color(self.colors["muted"])
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_color(self.colors["muted"])
        for spine in axis.spines.values():
            spine.set_color(self.colors["border"])

    def show_sensor_derivative(self, event):
        if event.artist is not self.sensor_line or len(event.ind) == 0:
            return
        if self.sensor_wavelengths_nm.size < 2:
            self.derivative_value.setText("Need at least two valid wavelengths")
            return

        indices = np.asarray(event.ind, dtype=int)
        curve_points = np.column_stack((
            self.sensor_wavelengths_nm[indices],
            self.sensor_positions_mm[indices],
        ))
        pixel_points = self.ax3.transData.transform(curve_points)
        click_point = np.array([event.mouseevent.x, event.mouseevent.y])
        index = indices[np.argmin(np.linalg.norm(pixel_points - click_point, axis=1))]

        derivative = np.gradient(self.sensor_positions_mm, self.sensor_wavelengths_nm)[index]
        wavelength = self.sensor_wavelengths_nm[index]
        position = self.sensor_positions_mm[index]
        self.derivative_value.setText(f"At {wavelength:.1f} nm: {derivative:.4f} mm/nm")

        if self.sensor_annotation is not None:
            self.sensor_annotation.remove()
        self.sensor_annotation = self.ax3.annotate(
            f"{wavelength:.1f} nm\n{derivative:.4f} mm/nm",
            xy=(wavelength, position),
            xytext=(12, 16),
            textcoords="offset points",
            color=self.colors["muted"],
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "fc": self.colors["surface"], "ec": self.colors["border"]},
            arrowprops={"arrowstyle": "->", "color": self.colors["muted"]},
        )
        self.canvas.draw_idle()

    def zoom_plot(self, event):
        if event.inaxes not in (self.ax1, self.ax2, self.ax3):
            return
        if event.xdata is None or event.ydata is None or event.step == 0:
            return

        scale = 1.2 ** -event.step
        x_min, x_max = event.inaxes.get_xlim()
        y_min, y_max = event.inaxes.get_ylim()
        event.inaxes.set_xlim(
            event.xdata - (event.xdata - x_min) * scale,
            event.xdata + (x_max - event.xdata) * scale,
        )
        event.inaxes.set_ylim(
            event.ydata - (event.ydata - y_min) * scale,
            event.ydata + (y_max - event.ydata) * scale,
        )
        self.canvas.draw_idle()

    def update_plot(self):
        try:
            # Get inputs
            self.lines_per_mm = float(self.entry_lines.text())
            self.blaze_wavelength = float(self.entry_blaze.text())
            self.incidence_angle_deg = float(self.entry_angle.text())
            self.order = int(self.combo_order.currentText())
            self.tube_lens_focal_length_mm = float(self.entry_focal_length.text())
            
            # Constants
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
            self.spread_text = f"{sensor_spread_mm:.3f} mm"
            self.spread_value.setText(self.spread_text)
            
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
            self.sensor_wavelengths_nm = self.wavelengths[valid_mask]
            self.sensor_positions_mm = sensor_positions_mm[valid_mask]
            self.sensor_line, = self.ax3.plot(
                self.sensor_wavelengths_nm, self.sensor_positions_mm,
                color=self.colors["teal"], linewidth=2.2, marker="o", markevery=8, markersize=3
            )
            self.sensor_line.set_picker(8)
            self.sensor_annotation = None
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
    application = QApplication(sys.argv)
    application.setApplicationName("Grating Diffraction Simulator")
    window = GratingSimulator()
    window.show()
    sys.exit(application.exec())