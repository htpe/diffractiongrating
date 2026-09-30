import math
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)


def calculate_resolution(
    wavelength_nm: float,
    numerical_aperture: float,
    magnification: float,
    tube_lens_focal_length_mm: float,
    field_of_view_um: float | None = None,
) -> dict[str, float | None]:
    if wavelength_nm <= 0 or numerical_aperture <= 0:
        raise ValueError("Wavelength and numerical aperture must be positive")
    if magnification <= 0 or tube_lens_focal_length_mm <= 0:
        raise ValueError("Magnification and tube lens focal length must be positive")

    resolution_um = 0.61 * wavelength_nm / numerical_aperture / 1000
    resolution_mm = resolution_um / 1000
    objective_focal_length_mm = tube_lens_focal_length_mm / magnification
    angle_span_rad = math.atan2(resolution_mm, objective_focal_length_mm)
    camera_span_mm = tube_lens_focal_length_mm * math.tan(angle_span_rad)
    expected_camera_span_mm = resolution_mm * magnification
    max_field_angle_deg: float | None = None
    if field_of_view_um is not None:
        if field_of_view_um <= 0:
            raise ValueError("Field of view must be positive")
        half_field_mm = field_of_view_um / 2000
        max_field_angle_deg = math.degrees(
            math.atan2(half_field_mm, objective_focal_length_mm)
        )

    return {
        "resolution_um": resolution_um,
        "objective_focal_length_mm": objective_focal_length_mm,
        "angle_span_deg": math.degrees(angle_span_rad),
        "max_field_angle_deg": max_field_angle_deg,
        "camera_span_um": camera_span_mm * 1000,
        "expected_camera_span_um": expected_camera_span_mm * 1000,
    }


class ResolutionCameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Resolution to Camera Span")
        self.resize(940, 650)
        self.setMinimumSize(760, 560)
        self.setStyleSheet("""
            QMainWindow, QWidget#root {
                background: #F2F5F2;
                color: #20302D;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QLabel { color: #4B5552; }
            QLabel#eyebrow {
                color: #147D78;
                font-size: 9pt;
                font-weight: 700;
            }
            QLabel#title {
                color: #20302D;
                font-size: 22pt;
                font-weight: 700;
            }
            QGroupBox {
                border: 1px solid #D7E0DC;
                border-radius: 7px;
                margin-top: 12px;
                padding: 15px 14px 13px;
                color: #4B5552;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
            QLineEdit {
                background: #FFFFFF;
                border: 1px solid #D7E0DC;
                border-radius: 5px;
                color: #20302D;
                padding: 7px 8px;
                min-height: 20px;
            }
            QLineEdit:focus { border-color: #147D78; }
            QFrame#metric {
                background: #FFFFFF;
                border: 1px solid #D7E0DC;
                border-radius: 6px;
            }
            QLabel#metricLabel {
                color: #4B5552;
                font-size: 9pt;
                font-weight: 700;
            }
            QLabel#metricValue {
                color: #20302D;
                font-size: 21pt;
                font-weight: 700;
            }
            QLabel#metricUnit {
                color: #4B5552;
                font-size: 10pt;
            }
            QFrame#formula {
                background: #E7ECE9;
                border: none;
                border-radius: 5px;
            }
            QLabel#formulaText {
                color: #4B5552;
                font-size: 10pt;
            }
        """)

        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(18)
        self.setCentralWidget(root)

        header = QHBoxLayout()
        title_stack = QVBoxLayout()
        eyebrow = QLabel("IMAGING CALCULATOR / 02")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Resolution to sensor span")
        title.setObjectName("title")
        title_stack.addWidget(eyebrow)
        title_stack.addWidget(title)
        header.addLayout(title_stack)
        header.addStretch(1)
        layout.addLayout(header)

        parameters = QGroupBox("OPTICAL PARAMETERS")
        form = QFormLayout(parameters)
        form.setContentsMargins(14, 18, 14, 14)
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(10)
        self.wavelength_input = self.create_input("550", 0.1, 1000000)
        self.na_input = self.create_input("0.75", 0.001, 2.0)
        self.magnification_input = self.create_input("40", 0.001, 10000)
        self.tube_lens_input = self.create_input("200", 0.001, 10000)
        self.fov_input = self.create_input("", 0.001, 10000000)
        self.fov_input.setPlaceholderText("optional")
        form.addRow("Wavelength (nm)", self.wavelength_input)
        form.addRow("Numerical aperture (NA)", self.na_input)
        form.addRow("System magnification (×)", self.magnification_input)
        form.addRow("Tube lens focal length (mm)", self.tube_lens_input)
        form.addRow("Sample field of view (µm, full width)", self.fov_input)
        layout.addWidget(parameters)

        results = QGridLayout()
        results.setHorizontalSpacing(12)
        results.setVerticalSpacing(12)
        results.setColumnStretch(0, 1)
        results.setColumnStretch(1, 1)
        results.setColumnStretch(1, 1)
        self.resolution_value = self.add_metric(results, 0, 0, "OBJECT RESOLUTION", "--", "µm")
        self.angle_value = self.add_metric(results, 0, 1, "DIFFRACTION ANGLE SPAN", "--", "°")
        self.camera_value = self.add_metric(results, 1, 0, "CAMERA-PLANE SPAN", "--", "µm")
        self.field_angle_value = self.add_metric(results, 1, 1, "MAX FIELD ANGLE", "--", "°")
        layout.addLayout(results)

        formula_panel = QFrame()
        formula_panel.setObjectName("formula")
        formula_layout = QVBoxLayout(formula_panel)
        formula_layout.setContentsMargins(15, 12, 15, 12)
        self.objective_focal_label = QLabel("Objective focal length: -- mm")
        self.objective_focal_label.setObjectName("formulaText")
        self.consistency_label = QLabel("Camera span = object resolution × system magnification")
        self.consistency_label.setObjectName("formulaText")
        self.field_angle_note = QLabel("Field angle is measured from the optical axis to the sample field edge.")
        self.field_angle_note.setObjectName("formulaText")
        formula_layout.addWidget(self.objective_focal_label)
        formula_layout.addWidget(self.consistency_label)
        formula_layout.addWidget(self.field_angle_note)
        layout.addWidget(formula_panel)
        layout.addStretch(1)

        for field in (
            self.wavelength_input,
            self.na_input,
            self.magnification_input,
            self.tube_lens_input,
            self.fov_input,
        ):
            field.textChanged.connect(lambda _value: self.update_results())

        self.update_results()

    @staticmethod
    def create_input(value: str, minimum: float, maximum: float) -> QLineEdit:
        field = QLineEdit(value)
        field.setAlignment(Qt.AlignmentFlag.AlignRight)
        field.setValidator(QDoubleValidator(minimum, maximum, 4, field))
        return field

    @staticmethod
    def add_metric(
        layout: QGridLayout,
        row: int,
        column: int,
        title: str,
        value: str,
        unit: str,
    ) -> QLabel:
        panel = QFrame()
        panel.setObjectName("metric")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 14, 14, 14)
        panel_layout.setSpacing(6)

        label = QLabel(title)
        label.setObjectName("metricLabel")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        unit_label = QLabel(unit)
        unit_label.setObjectName("metricUnit")
        value_row = QHBoxLayout()
        value_row.addWidget(value_label)
        value_row.addWidget(unit_label, alignment=Qt.AlignmentFlag.AlignBottom)
        value_row.addStretch(1)

        panel_layout.addWidget(label)
        panel_layout.addLayout(value_row)
        layout.addWidget(panel, row, column)
        return value_label

    def update_results(self):
        try:
            result = calculate_resolution(
                float(self.wavelength_input.text()),
                float(self.na_input.text()),
                float(self.magnification_input.text()),
                float(self.tube_lens_input.text()),
                float(self.fov_input.text()) if self.fov_input.text() else None,
            )
        except ValueError:
            return

        self.resolution_value.setText(f"{result['resolution_um']:.4f}")
        self.angle_value.setText(f"{result['angle_span_deg']:.5f}")
        self.camera_value.setText(f"{result['camera_span_um']:.4f}")
        max_field_angle = result["max_field_angle_deg"]
        self.field_angle_value.setText(
            f"±{max_field_angle:.5f}" if max_field_angle is not None else "--"
        )
        self.objective_focal_label.setText(
            f"Objective focal length: {result['objective_focal_length_mm']:.4f} mm"
        )
        self.consistency_label.setText(
            f"Camera span: {result['camera_span_um']:.4f} µm    |    "
            f"Resolution × magnification: {result['expected_camera_span_um']:.4f} µm"
        )


if __name__ == "__main__":
    application = QApplication(sys.argv)
    application.setApplicationName("Optical Resolution to Camera Span")
    window = ResolutionCameraWindow()
    window.show()
    sys.exit(application.exec())
