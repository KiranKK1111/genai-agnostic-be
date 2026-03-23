"""Visualization clarification and config generation."""
import logging
from app.builders.viz_config import (
    viz_type_clarification,
    axis_mode_clarification,
    axis_specific_clarification,
)

logger = logging.getLogger(__name__)


def build_viz_clarification(columns: list, data_types: dict = None) -> dict:
    """Build viz type clarification payload."""
    return viz_type_clarification(columns)


def build_axis_mode_clarification() -> dict:
    return axis_mode_clarification()


def build_viz_config(viz_type: str, columns: list, x_axis: str = None, y_axes: list = None) -> dict:
    """Build visualization config for the frontend."""
    return {
        "viz_type": viz_type,
        "axis_mode": "on_the_fly" if not x_axis else "specific",
        "x_axis": x_axis,
        "y_axes": y_axes or [],
        "all_columns": columns,
        "config": {
            "legend": {"position": "top-right", "show": True},
            "colors": ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"],
        }
    }
