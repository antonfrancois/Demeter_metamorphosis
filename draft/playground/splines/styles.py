"""Shared visual language and display metadata for the spline playground."""

DUAL_COLOR = "#ef8354"
PRIMAL_COLOR = "#ffd166"
INK_COLOR = "#20323a"
PANEL_COLOR = "#f7f7f2"
CANVAS_COLOR = "#e7ece9"
TARGET_COLOR = "#4267ac"
TARGET_ACTIVE_COLOR = "#7a3db8"

INPUT_FIELDS = (
    "initial_momentum",
    "initial_acceleration",
    "initial_jerk",
    "control_jerk",
)
INPUT_LABELS = (
    r"Momentum  $p_0$",
    r"Acceleration  $a_0$",
    r"Jerk  $r_0$",
    r"Control jerk  $r(\tau^+)$",
)
CURRENT_FIELDS = (
    None,
    "momentum",
    "force",
    "acceleration",
    "jerk",
    "velocity",
    "vector_momentum",
)
CURRENT_LABELS = (
    "No field overlay",
    r"Momentum  $p$",
    r"Force  $u=A_I^{-1}a$",
    r"Acceleration  $a=A_Iu$",
    r"Jerk  $r$",
    r"Velocity  $v=Km$",
    r"Vector momentum  $m=Lv$",
)
CURRENT_IMAGE_MODES = ("full", "deformation", "photometric")
CURRENT_IMAGE_LABELS = (
    r"Full metamorphosis  $I$",
    r"Deformation only  $I_D$",
    r"Photometric only  $I_{\mathrm{phot}}$",
)
FIELD_CLASS = {
    "momentum": "primal",
    "force": "dual",
    "acceleration": "primal",
    "jerk": "dual",
    "velocity": "primal",
    "vector_momentum": "dual",
}
FIELD_SYMBOL = {
    "momentum": "p",
    "force": "u",
    "acceleration": "a",
    "jerk": "r",
    "velocity": "v",
    "vector_momentum": "m",
}
