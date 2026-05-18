#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-20"
# File: scitex_stats/_schema.py

"""
Statistical Result Schema for scitex_stats.

Provides dataclasses for representing statistical results with
positioning and styling metadata for GUI annotation and export.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Optional

# Type aliases
PositionMode = Literal[
    "auto", "absolute", "relative_to_plot", "above_whisker", "below_whisker"
]
UnitType = Literal["mm", "px", "inch", "pt"]
SymbolStyle = Literal["asterisk", "bracket", "line"]


@dataclass
class Position:
    """Coordinate position with unit support."""

    x: float
    y: float
    unit: UnitType = "mm"
    relative_to: Optional[str] = None
    offset: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "unit": self.unit,
            "relative_to": self.relative_to,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create Position from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            unit=data.get("unit", "mm"),
            relative_to=data.get("relative_to"),
            offset=data.get("offset"),
        )

    def to_px(self, dpi: float = 96.0) -> "Position":
        """Convert to pixel units at given DPI."""
        if self.unit == "px":
            return Position(x=self.x, y=self.y, unit="px")
        elif self.unit == "mm":
            factor = dpi / 25.4
            return Position(x=self.x * factor, y=self.y * factor, unit="px")
        elif self.unit == "inch":
            return Position(x=self.x * dpi, y=self.y * dpi, unit="px")
        elif self.unit == "pt":
            factor = dpi / 72.0
            return Position(x=self.x * factor, y=self.y * factor, unit="px")
        return Position(x=self.x, y=self.y, unit="px")

    def to_mm(self, dpi: float = 96.0) -> "Position":
        """Convert to millimeter units."""
        if self.unit == "mm":
            return Position(x=self.x, y=self.y, unit="mm")
        elif self.unit == "px":
            factor = 25.4 / dpi
            return Position(x=self.x * factor, y=self.y * factor, unit="mm")
        elif self.unit == "inch":
            return Position(x=self.x * 25.4, y=self.y * 25.4, unit="mm")
        elif self.unit == "pt":
            factor = 25.4 / 72.0
            return Position(x=self.x * factor, y=self.y * factor, unit="mm")
        return Position(x=self.x, y=self.y, unit="mm")


@dataclass
class StatStyling:
    """Styling configuration for statistical annotations."""

    font_size_pt: float = 7.0
    font_family: str = "Arial"
    color: str = "#000000"
    symbol_style: SymbolStyle = "asterisk"
    line_width_mm: float = 0.25
    theme: str = "auto"

    def get_theme_color(self, is_dark: bool = False) -> str:
        """Get color based on theme setting."""
        if self.theme == "dark":
            return "#ffffff"
        elif self.theme == "light":
            return "#000000"
        else:  # auto
            return "#ffffff" if is_dark else "#000000"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "font_size_pt": self.font_size_pt,
            "font_family": self.font_family,
            "color": self.color,
            "symbol_style": self.symbol_style,
            "line_width_mm": self.line_width_mm,
            "theme": self.theme,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatStyling":
        """Create StatStyling from dictionary."""
        return cls(
            font_size_pt=data.get("font_size_pt", 7.0),
            font_family=data.get("font_family", "Arial"),
            color=data.get("color", "#000000"),
            symbol_style=data.get("symbol_style", "asterisk"),
            line_width_mm=data.get("line_width_mm", 0.25),
            theme=data.get("theme", "auto"),
        )


@dataclass
class StatPositioning:
    """Positioning configuration for statistical annotations."""

    mode: PositionMode = "auto"
    position: Optional[Position] = None
    avoid_overlap: bool = True
    min_distance_mm: float = 2.0
    preferred_corner: Optional[str] = None
    anchor_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode,
            "position": self.position.to_dict() if self.position else None,
            "avoid_overlap": self.avoid_overlap,
            "min_distance_mm": self.min_distance_mm,
            "preferred_corner": self.preferred_corner,
            "anchor_to": self.anchor_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatPositioning":
        """Create StatPositioning from dictionary."""
        position = None
        if data.get("position") is not None:
            position = Position.from_dict(data["position"])
        return cls(
            mode=data.get("mode", "auto"),
            position=position,
            avoid_overlap=data.get("avoid_overlap", True),
            min_distance_mm=data.get("min_distance_mm", 2.0),
            preferred_corner=data.get("preferred_corner"),
            anchor_to=data.get("anchor_to"),
        )


def _to_native(value: Any) -> Any:
    """Recursively convert numpy/special types to native Python types."""
    # `numpy` is a hard dep — plain import (the prior try/except was dead).
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    return value


@dataclass
class StatResult:
    """Complete statistical result with formatting and positioning metadata."""

    test_type: str
    test_category: str
    statistic: Dict[str, Any]
    p_value: float
    stars: str
    created_at: Optional[str] = None
    software_version: Optional[str] = None
    plot_id: Optional[str] = None
    effect_size: Optional[Dict[str, Any]] = None
    correction: Optional[Dict[str, Any]] = None
    samples: Optional[Dict[str, Any]] = None
    assumptions: Optional[Dict[str, Any]] = None
    ci_95: Optional[Any] = None
    positioning: Optional[StatPositioning] = None
    styling: Optional[StatStyling] = None
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Set defaults for optional fields."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.styling is None:
            self.styling = StatStyling()
        if self.positioning is None:
            self.positioning = StatPositioning()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with native Python types."""
        return _to_native(
            {
                "test_type": self.test_type,
                "test_category": self.test_category,
                "statistic": self.statistic,
                "p_value": self.p_value,
                "stars": self.stars,
                "created_at": self.created_at,
                "software_version": self.software_version,
                "plot_id": self.plot_id,
                "effect_size": self.effect_size,
                "correction": self.correction,
                "samples": self.samples,
                "assumptions": self.assumptions,
                "ci_95": self.ci_95,
                "positioning": (
                    self.positioning.to_dict() if self.positioning else None
                ),
                "styling": self.styling.to_dict() if self.styling else None,
                "extra": self.extra,
            }
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatResult":
        """Create StatResult from dictionary."""
        positioning = None
        if data.get("positioning") is not None:
            positioning = StatPositioning.from_dict(data["positioning"])
        styling = None
        if data.get("styling") is not None:
            styling = StatStyling.from_dict(data["styling"])
        return cls(
            test_type=data["test_type"],
            test_category=data["test_category"],
            statistic=data["statistic"],
            p_value=data["p_value"],
            stars=data["stars"],
            created_at=data.get("created_at"),
            software_version=data.get("software_version"),
            plot_id=data.get("plot_id"),
            effect_size=data.get("effect_size"),
            correction=data.get("correction"),
            samples=data.get("samples"),
            assumptions=data.get("assumptions"),
            ci_95=data.get("ci_95"),
            positioning=positioning,
            styling=styling,
            extra=data.get("extra"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "StatResult":
        """Create StatResult from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def format_text(self, style: str = "compact") -> str:
        """Format the statistical result as text.

        Parameters
        ----------
        style : str
            One of: 'compact', 'asterisk', 'text', 'detailed', 'publication'

        Returns
        -------
        str
            Formatted string representation.
        """
        stat_name = self.statistic.get("name", "stat")
        stat_val = self.statistic.get("value", 0.0)

        if style == "compact":
            return f"{stat_name} = {stat_val:.3f}{self.stars}"
        elif style == "asterisk":
            return self.stars
        elif style == "text":
            return f"p = {self.p_value:.3f}"
        elif style == "detailed":
            p_str = f"{self.p_value:.3e}"
            result = f"{stat_name} = {stat_val:.3f}, p = {p_str}"
            if self.effect_size:
                es_name = self.effect_size.get("name", "effect")
                es_val = self.effect_size.get("value", 0.0)
                result += f", {es_name} = {es_val}"
            return result
        elif style == "publication":
            p_str = self._format_p_publication()
            return f"({stat_name} = {stat_val:.2f}, {p_str})"
        return f"{stat_name} = {stat_val}"

    def _format_p_publication(self) -> str:
        """Format p-value in publication style."""
        p = self.p_value
        if p < 0.001:
            return "p < 0.001"
        elif p < 0.01:
            return "p < 0.01"
        elif p < 0.05:
            return "p < 0.05"
        else:
            return f"p = {p:.3f}"

    def get_interpretation(self) -> str:
        """Get human-readable interpretation of the result."""
        is_sig = self.p_value < 0.05
        sig_str = "significant" if is_sig else "non-significant"

        if self.test_category == "correlation":
            stat_val = self.statistic.get("value", 0.0)
            abs_val = abs(stat_val)
            direction = "positive" if stat_val > 0 else "negative"

            if abs_val >= 0.7:
                strength = "Strong"
            elif abs_val >= 0.5:
                strength = "Moderate"
            elif abs_val >= 0.3:
                strength = "Weak"
            else:
                strength = "Very weak"

            r_str = f"r={stat_val}"
            return f"{strength} {direction} correlation ({sig_str}, {r_str})"

        else:
            stat_name = self.statistic.get("name", "stat")
            stat_val = self.statistic.get("value", 0.0)
            return f"Significant difference ({stat_name}={stat_val:.2f}, p={self.p_value:.3f})"

    def to_annotation_dict(self) -> Dict[str, Any]:
        """Convert to annotation dictionary for GUI use."""
        stat_name = self.statistic.get("name", "stat")
        stat_val = self.statistic.get("value", 0.0)

        result = {
            "type": "stat",
            "id": self.plot_id,
            "statResult": {
                "test_name": self.test_type,
                "p_value": self.p_value,
                "stars": self.stars,
                "formatted_output": f"{stat_name} = {stat_val:.3f}{self.stars}",
            },
        }

        # Add group names if samples present
        if self.samples:
            for key, sample_data in self.samples.items():
                if isinstance(sample_data, dict) and "name" in sample_data:
                    result["statResult"][key] = sample_data["name"]

        # Add position if positioning set
        if self.positioning and self.positioning.position is not None:
            result["position"] = self.positioning.position.to_dict()

        return result


# Category mappings for test types
_CATEGORY_MAP = {
    "pearson": "correlation",
    "spearman": "correlation",
    "kendall": "correlation",
    "t-test": "parametric",
    "ttest": "parametric",
    "anova": "parametric",
    "anova_rm": "parametric",
    "anova_2way": "parametric",
    "mannwhitney": "non-parametric",
    "wilcoxon": "non-parametric",
    "kruskal": "non-parametric",
    "friedman": "non-parametric",
    "brunner_munzel": "non-parametric",
}

# Stars thresholds
_STARS_MAP = [
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
    (1.0, "ns"),
]


def _p_to_stars(p_value: float) -> str:
    """Convert p-value to stars string."""
    for threshold, stars in _STARS_MAP:
        if p_value < threshold:
            return stars
    return "ns"


def create_stat_result(
    test_type: str,
    statistic_name: str,
    statistic_value: float,
    p_value: float,
    **kwargs: Any,
) -> StatResult:
    """Convenience function to create a StatResult.

    Parameters
    ----------
    test_type : str
        Name of the statistical test.
    statistic_name : str
        Name of the test statistic (e.g., 'r', 't', 'F').
    statistic_value : float
        Value of the test statistic.
    p_value : float
        P-value from the test.
    **kwargs
        Additional fields passed to StatResult.

    Returns
    -------
    StatResult
    """
    test_category = _CATEGORY_MAP.get(test_type.lower(), "other")
    stars = _p_to_stars(p_value)
    statistic = {"name": statistic_name, "value": statistic_value}

    return StatResult(
        test_type=test_type,
        test_category=test_category,
        statistic=statistic,
        p_value=p_value,
        stars=stars,
        **kwargs,
    )


__all__ = [
    "PositionMode",
    "UnitType",
    "SymbolStyle",
    "Position",
    "StatStyling",
    "StatPositioning",
    "StatResult",
    "create_stat_result",
]

# EOF
