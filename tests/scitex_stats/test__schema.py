#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-29"

"""Tests for scitex.stats._schema module."""

import json

import numpy as np

from scitex_stats._schema import (
    Position,
    StatPositioning,
    StatResult,
    StatStyling,
    create_stat_result,
)


class TestPosition:
    """Test Position class for coordinate handling."""

    def test_position_basic_creation_sets_x(self):
        # Arrange
        # Act
        pos = Position(x=10.0, y=20.0, unit="mm")
        # Assert
        assert pos.x == 10.0

    def test_position_basic_creation_sets_y(self):
        # Arrange
        # Act
        pos = Position(x=10.0, y=20.0, unit="mm")
        # Assert
        assert pos.y == 20.0

    def test_position_basic_creation_sets_unit(self):
        # Arrange
        # Act
        pos = Position(x=10.0, y=20.0, unit="mm")
        # Assert
        assert pos.unit == "mm"

    def test_position_relative_records_relative_to(self):
        # Arrange
        # Act
        pos = Position(
            x=5.0,
            y=10.0,
            unit="mm",
            relative_to="plot_0",
            offset={"dx": 2.0, "dy": -1.0},
        )
        # Assert
        assert pos.relative_to == "plot_0"

    def test_position_relative_records_offset(self):
        # Arrange
        # Act
        pos = Position(
            x=5.0,
            y=10.0,
            unit="mm",
            relative_to="plot_0",
            offset={"dx": 2.0, "dy": -1.0},
        )
        # Assert
        assert pos.offset["dx"] == 2.0

    def test_position_to_dict_includes_x(self):
        # Arrange
        pos = Position(x=10.0, y=20.0, unit="mm")
        # Act
        data = pos.to_dict()
        # Assert
        assert data["x"] == 10.0

    def test_position_to_dict_includes_unit(self):
        # Arrange
        pos = Position(x=10.0, y=20.0, unit="mm")
        # Act
        data = pos.to_dict()
        # Assert
        assert data["unit"] == "mm"

    def test_position_from_dict_reads_x(self):
        # Arrange
        data = {"x": 15.0, "y": 25.0, "unit": "px"}
        # Act
        pos = Position.from_dict(data)
        # Assert
        assert pos.x == 15.0

    def test_position_from_dict_reads_unit(self):
        # Arrange
        data = {"x": 15.0, "y": 25.0, "unit": "px"}
        # Act
        pos = Position.from_dict(data)
        # Assert
        assert pos.unit == "px"

    def test_position_mm_to_px_sets_px_unit(self):
        # Arrange
        pos = Position(x=10.0, y=20.0, unit="mm")
        # Act
        pos_px = pos.to_px(dpi=300.0)
        # Assert
        assert pos_px.unit == "px"

    def test_position_mm_to_px_converts_x(self):
        # Arrange
        pos = Position(x=10.0, y=20.0, unit="mm")
        expected_px = 10.0 * (300.0 / 25.4)  # at 300 DPI: 1mm = 300/25.4 px
        # Act
        pos_px = pos.to_px(dpi=300.0)
        # Assert
        assert abs(pos_px.x - expected_px) < 0.01

    def test_position_px_to_mm_sets_mm_unit(self):
        # Arrange
        pos = Position(x=118.11, y=236.22, unit="px")
        # Act
        pos_mm = pos.to_mm(dpi=300.0)
        # Assert
        assert pos_mm.unit == "mm"

    def test_position_px_to_mm_converts_x(self):
        # Arrange
        pos = Position(x=118.11, y=236.22, unit="px")  # 118.11 px ~= 10mm at 300 DPI
        # Act
        pos_mm = pos.to_mm(dpi=300.0)
        # Assert
        assert abs(pos_mm.x - 10.0) < 0.01

    def test_position_inch_to_mm_converts_x(self):
        # Arrange
        pos = Position(x=1.0, y=2.0, unit="inch")  # 1 inch = 25.4 mm
        # Act
        pos_mm = pos.to_mm()
        # Assert
        assert abs(pos_mm.x - 25.4) < 0.01

    def test_position_inch_to_mm_converts_y(self):
        # Arrange
        pos = Position(x=1.0, y=2.0, unit="inch")  # 2 inch = 50.8 mm
        # Act
        pos_mm = pos.to_mm()
        # Assert
        assert abs(pos_mm.y - 50.8) < 0.01


class TestStatStyling:
    """Test StatStyling class for display styling."""

    def test_default_styling_font_size(self):
        # Arrange
        # Act
        style = StatStyling()
        # Assert
        assert style.font_size_pt == 7.0

    def test_default_styling_font_family(self):
        # Arrange
        # Act
        style = StatStyling()
        # Assert
        assert style.font_family == "Arial"

    def test_default_styling_color(self):
        # Arrange
        # Act
        style = StatStyling()
        # Assert
        assert style.color == "#000000"

    def test_default_styling_symbol_style(self):
        # Arrange
        # Act
        style = StatStyling()
        # Assert
        assert style.symbol_style == "asterisk"

    def test_custom_styling_font_size(self):
        # Arrange
        # Act
        style = StatStyling(
            font_size_pt=8.0,
            font_family="Times",
            color="#FF0000",
            symbol_style="bracket",
            line_width_mm=0.3,
        )
        # Assert
        assert style.font_size_pt == 8.0

    def test_custom_styling_font_family(self):
        # Arrange
        # Act
        style = StatStyling(
            font_size_pt=8.0,
            font_family="Times",
            color="#FF0000",
            symbol_style="bracket",
            line_width_mm=0.3,
        )
        # Assert
        assert style.font_family == "Times"

    def test_custom_styling_symbol_style(self):
        # Arrange
        # Act
        style = StatStyling(
            font_size_pt=8.0,
            font_family="Times",
            color="#FF0000",
            symbol_style="bracket",
            line_width_mm=0.3,
        )
        # Assert
        assert style.symbol_style == "bracket"

    def test_custom_styling_line_width(self):
        # Arrange
        # Act
        style = StatStyling(
            font_size_pt=8.0,
            font_family="Times",
            color="#FF0000",
            symbol_style="bracket",
            line_width_mm=0.3,
        )
        # Assert
        assert style.line_width_mm == 0.3

    def test_theme_color_auto_light_is_black(self):
        # Arrange
        style = StatStyling(theme="auto")
        # Act
        color = style.get_theme_color(is_dark=False)
        # Assert
        assert color == "#000000"

    def test_theme_color_auto_dark_is_white(self):
        # Arrange
        style = StatStyling(theme="auto")
        # Act
        color = style.get_theme_color(is_dark=True)
        # Assert
        assert color == "#ffffff"

    def test_theme_color_explicit_dark_stays_dark(self):
        # Arrange
        style = StatStyling(theme="dark")
        # Act
        color = style.get_theme_color(is_dark=False)  # explicit dark overrides
        # Assert
        assert color == "#ffffff"

    def test_styling_round_trip_preserves_font_size(self):
        # Arrange
        style = StatStyling(font_size_pt=8.0, color="#FF0000")
        # Act
        style2 = StatStyling.from_dict(style.to_dict())
        # Assert
        assert style2.font_size_pt == 8.0

    def test_styling_round_trip_preserves_color(self):
        # Arrange
        style = StatStyling(font_size_pt=8.0, color="#FF0000")
        # Act
        style2 = StatStyling.from_dict(style.to_dict())
        # Assert
        assert style2.color == "#FF0000"


class TestStatPositioning:
    """Test StatPositioning class for GUI-ready positioning."""

    def test_default_positioning_mode(self):
        # Arrange
        # Act
        pos = StatPositioning()
        # Assert
        assert pos.mode == "auto"

    def test_default_positioning_avoid_overlap(self):
        # Arrange
        # Act
        pos = StatPositioning()
        # Assert
        assert pos.avoid_overlap is True

    def test_default_positioning_min_distance(self):
        # Arrange
        # Act
        pos = StatPositioning()
        # Assert
        assert pos.min_distance_mm == 2.0

    def test_positioning_with_position_sets_mode(self):
        # Arrange
        position = Position(x=10.0, y=20.0, unit="mm")
        # Act
        pos = StatPositioning(
            mode="absolute", position=position, preferred_corner="top-right"
        )
        # Assert
        assert pos.mode == "absolute"

    def test_positioning_with_position_keeps_coordinates(self):
        # Arrange
        position = Position(x=10.0, y=20.0, unit="mm")
        # Act
        pos = StatPositioning(
            mode="absolute", position=position, preferred_corner="top-right"
        )
        # Assert
        assert pos.position.x == 10.0

    def test_positioning_with_position_keeps_corner(self):
        # Arrange
        position = Position(x=10.0, y=20.0, unit="mm")
        # Act
        pos = StatPositioning(
            mode="absolute", position=position, preferred_corner="top-right"
        )
        # Assert
        assert pos.preferred_corner == "top-right"

    def test_positioning_to_dict_records_mode(self):
        # Arrange
        position = Position(x=5.0, y=10.0, unit="mm")
        pos = StatPositioning(mode="relative_to_plot", position=position)
        # Act
        data = pos.to_dict()
        # Assert
        assert data["mode"] == "relative_to_plot"

    def test_positioning_to_dict_nests_position_coordinates(self):
        # Arrange
        position = Position(x=5.0, y=10.0, unit="mm")
        pos = StatPositioning(mode="relative_to_plot", position=position)
        # Act
        data = pos.to_dict()
        # Assert
        assert data["position"]["x"] == 5.0

    def test_positioning_from_dict_reads_mode(self):
        # Arrange
        data = {
            "mode": "above_whisker",
            "position": {"x": 15.0, "y": 25.0, "unit": "px"},
            "avoid_overlap": True,
            "min_distance_mm": 2.0,
            "preferred_corner": None,
            "anchor_to": None,
        }
        # Act
        pos = StatPositioning.from_dict(data)
        # Assert
        assert pos.mode == "above_whisker"

    def test_positioning_from_dict_reads_nested_position(self):
        # Arrange
        data = {
            "mode": "above_whisker",
            "position": {"x": 15.0, "y": 25.0, "unit": "px"},
            "avoid_overlap": True,
            "min_distance_mm": 2.0,
            "preferred_corner": None,
            "anchor_to": None,
        }
        # Act
        pos = StatPositioning.from_dict(data)
        # Assert
        assert pos.position.x == 15.0


class TestStatResult:
    """Test StatResult main class."""

    def _pearson_result(self):
        return StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
        )

    def test_result_basic_creation_sets_test_type(self):
        # Arrange
        # Act
        result = self._pearson_result()
        # Assert
        assert result.test_type == "pearson"

    def test_result_basic_creation_sets_statistic_value(self):
        # Arrange
        # Act
        result = self._pearson_result()
        # Assert
        assert result.statistic["value"] == 0.85

    def test_result_basic_creation_sets_p_value(self):
        # Arrange
        # Act
        result = self._pearson_result()
        # Assert
        assert result.p_value == 0.001

    def test_result_post_init_sets_created_at(self):
        # Arrange
        # Act
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.45},
            p_value=0.002,
            stars="**",
        )
        # Assert
        assert result.created_at is not None

    def test_result_post_init_default_styling_font_size(self):
        # Arrange
        # Act
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.45},
            p_value=0.002,
            stars="**",
        )
        # Assert
        assert result.styling.font_size_pt == 7.0

    def test_result_post_init_default_positioning_mode(self):
        # Arrange
        # Act
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.45},
            p_value=0.002,
            stars="**",
        )
        # Assert
        assert result.positioning.mode == "auto"

    def test_result_with_effect_size_records_name(self):
        # Arrange
        # Act
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.45},
            p_value=0.002,
            stars="**",
            effect_size={
                "name": "cohens_d",
                "value": 0.85,
                "interpretation": "large",
                "ci_95": [0.42, 1.28],
            },
        )
        # Assert
        assert result.effect_size["name"] == "cohens_d"

    def test_result_with_effect_size_records_interpretation(self):
        # Arrange
        # Act
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.45},
            p_value=0.002,
            stars="**",
            effect_size={
                "name": "cohens_d",
                "value": 0.85,
                "interpretation": "large",
                "ci_95": [0.42, 1.28],
            },
        )
        # Assert
        assert result.effect_size["interpretation"] == "large"

    def test_result_with_samples_records_group1_n(self):
        # Arrange
        # Act
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.34},
            p_value=0.023,
            stars="*",
            samples={
                "group1": {"name": "Control", "n": 30, "mean": 5.2, "std": 1.1},
                "group2": {"name": "Treatment", "n": 32, "mean": 6.8, "std": 1.3},
            },
        )
        # Assert
        assert result.samples["group1"]["n"] == 30

    def test_result_with_samples_records_group2_mean(self):
        # Arrange
        # Act
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.34},
            p_value=0.023,
            stars="*",
            samples={
                "group1": {"name": "Control", "n": 30, "mean": 5.2, "std": 1.1},
                "group2": {"name": "Treatment", "n": 32, "mean": 6.8, "std": 1.3},
            },
        )
        # Assert
        assert result.samples["group2"]["mean"] == 6.8

    def test_result_to_dict_records_test_type(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.75},
            p_value=0.01,
            stars="**",
        )
        # Act
        data = result.to_dict()
        # Assert
        assert data["test_type"] == "pearson"

    def test_result_to_dict_records_statistic_value(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.75},
            p_value=0.01,
            stars="**",
        )
        # Act
        data = result.to_dict()
        # Assert
        assert data["statistic"]["value"] == 0.75

    def test_result_to_dict_converts_numpy_statistic_to_float(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": np.float64(3.45)},
            p_value=np.float64(0.002),
            stars="**",
            ci_95=[np.float64(0.5), np.float64(1.2)],
        )
        # Act
        data = result.to_dict()
        # Assert
        assert isinstance(data["statistic"]["value"], float)

    def test_result_to_dict_converts_numpy_ci_to_float(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": np.float64(3.45)},
            p_value=np.float64(0.002),
            stars="**",
            ci_95=[np.float64(0.5), np.float64(1.2)],
        )
        # Act
        data = result.to_dict()
        # Assert
        assert isinstance(data["ci_95"][0], float)

    def test_result_to_json_records_test_type(self):
        # Arrange
        result = self._pearson_result()
        # Act
        data = json.loads(result.to_json())
        # Assert
        assert data["test_type"] == "pearson"

    def test_result_to_json_records_stars(self):
        # Arrange
        result = self._pearson_result()
        # Act
        data = json.loads(result.to_json())
        # Assert
        assert data["stars"] == "***"

    def test_result_from_dict_reads_test_type(self):
        # Arrange
        data = {
            "test_type": "spearman",
            "test_category": "correlation",
            "statistic": {"name": "rho", "value": 0.72},
            "p_value": 0.005,
            "stars": "**",
            "created_at": "2025-01-01T00:00:00",
            "software_version": "1.0.0",
            "plot_id": "plot_0",
            "effect_size": None,
            "correction": None,
            "samples": None,
            "assumptions": None,
            "ci_95": None,
            "positioning": None,
            "styling": None,
            "extra": None,
        }
        # Act
        result = StatResult.from_dict(data)
        # Assert
        assert result.test_type == "spearman"

    def test_result_from_dict_reads_statistic_value(self):
        # Arrange
        data = {
            "test_type": "spearman",
            "test_category": "correlation",
            "statistic": {"name": "rho", "value": 0.72},
            "p_value": 0.005,
            "stars": "**",
            "created_at": "2025-01-01T00:00:00",
            "software_version": "1.0.0",
            "plot_id": "plot_0",
            "effect_size": None,
            "correction": None,
            "samples": None,
            "assumptions": None,
            "ci_95": None,
            "positioning": None,
            "styling": None,
            "extra": None,
        }
        # Act
        result = StatResult.from_dict(data)
        # Assert
        assert result.statistic["value"] == 0.72

    def test_result_from_json_reads_test_type(self):
        # Arrange
        json_str = """
        {"test_type": "kendall", "test_category": "correlation",
         "statistic": {"name": "tau", "value": 0.65}, "p_value": 0.01, "stars": "**",
         "created_at": "2025-01-01T00:00:00", "software_version": null, "plot_id": null,
         "effect_size": null, "correction": null, "samples": null, "assumptions": null,
         "ci_95": null, "positioning": null, "styling": null, "extra": null}
        """
        # Act
        result = StatResult.from_json(json_str)
        # Assert
        assert result.test_type == "kendall"

    def test_result_from_json_reads_p_value(self):
        # Arrange
        json_str = """
        {"test_type": "kendall", "test_category": "correlation",
         "statistic": {"name": "tau", "value": 0.65}, "p_value": 0.01, "stars": "**",
         "created_at": "2025-01-01T00:00:00", "software_version": null, "plot_id": null,
         "effect_size": null, "correction": null, "samples": null, "assumptions": null,
         "ci_95": null, "positioning": null, "styling": null, "extra": null}
        """
        # Act
        result = StatResult.from_json(json_str)
        # Assert
        assert result.p_value == 0.01

    def test_format_text_compact_style(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.850},
            p_value=0.001,
            stars="***",
        )
        # Act
        text = result.format_text(style="compact")
        # Assert
        assert text == "r = 0.850***"

    def test_format_text_asterisk_style(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.850},
            p_value=0.001,
            stars="***",
        )
        # Act
        text = result.format_text(style="asterisk")
        # Assert
        assert text == "***"

    def test_format_text_asterisk_non_significant(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.15},
            p_value=0.15,
            stars="ns",
        )
        # Act
        text = result.format_text(style="asterisk")
        # Assert
        assert text == "ns"

    def test_format_text_text_style(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.850},
            p_value=0.023,
            stars="*",
        )
        # Act
        text = result.format_text(style="text")
        # Assert
        assert text == "p = 0.023"

    def test_format_text_detailed_includes_statistic(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.456},
            p_value=0.002,
            stars="**",
            effect_size={"name": "d", "value": 0.85},
        )
        # Act
        text = result.format_text(style="detailed")
        # Assert
        assert "t = 3.456" in text

    def test_format_text_detailed_includes_p_value(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.456},
            p_value=0.002,
            stars="**",
            effect_size={"name": "d", "value": 0.85},
        )
        # Act
        text = result.format_text(style="detailed")
        # Assert
        assert "p = 2.000e-03" in text

    def test_format_text_detailed_includes_effect_size(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.456},
            p_value=0.002,
            stars="**",
            effect_size={"name": "d", "value": 0.85},
        )
        # Act
        text = result.format_text(style="detailed")
        # Assert
        assert "d = 0.85" in text

    def test_format_text_publication_style(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.0005,
            stars="***",
        )
        # Act
        text = result.format_text(style="publication")
        # Assert
        assert text == "(r = 0.85, p < 0.001)"

    def test_format_p_publication_very_small(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 5.0},
            p_value=0.0001,
            stars="***",
        )
        # Act
        p_text = result._format_p_publication()
        # Assert
        assert p_text == "p < 0.001"

    def test_format_p_publication_small(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.0},
            p_value=0.005,
            stars="**",
        )
        # Act
        p_text = result._format_p_publication()
        # Assert
        assert p_text == "p < 0.01"

    def test_format_p_publication_borderline(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 2.0},
            p_value=0.045,
            stars="*",
        )
        # Act
        p_text = result._format_p_publication()
        # Assert
        assert p_text == "p < 0.05"

    def test_format_p_publication_non_significant(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 1.0},
            p_value=0.15,
            stars="ns",
        )
        # Act
        p_text = result._format_p_publication()
        # Assert
        assert p_text == "p = 0.150"

    def test_interpretation_strong_positive_label(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
        )
        # Act
        interp = result.get_interpretation()
        # Assert
        assert "Strong positive correlation" in interp

    def test_interpretation_strong_positive_significance(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
        )
        # Act
        interp = result.get_interpretation()
        # Assert
        assert "significant" in interp

    def test_interpretation_weak_negative_label(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": -0.35},
            p_value=0.045,
            stars="*",
        )
        # Act
        interp = result.get_interpretation()
        # Assert
        assert "Weak negative correlation" in interp

    def test_interpretation_very_weak_label(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.15},
            p_value=0.25,
            stars="ns",
        )
        # Act
        interp = result.get_interpretation()
        # Assert
        assert "Very weak positive correlation" in interp

    def test_interpretation_very_weak_non_significant(self):
        # Arrange
        result = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.15},
            p_value=0.25,
            stars="ns",
        )
        # Act
        interp = result.get_interpretation()
        # Assert
        assert "non-significant" in interp

    def test_interpretation_ttest_significant_difference(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.45},
            p_value=0.002,
            stars="**",
        )
        # Act
        interp = result.get_interpretation()
        # Assert
        assert "Significant difference" in interp

    def test_interpretation_ttest_includes_statistic(self):
        # Arrange
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.45},
            p_value=0.002,
            stars="**",
        )
        # Act
        interp = result.get_interpretation()
        # Assert
        assert "t=3.45" in interp

    def _annotation_result(self):
        return StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
            plot_id="plot_0",
            samples={
                "group1": {"name": "Control", "n": 30},
                "group2": {"name": "Treatment", "n": 32},
            },
        )

    def test_annotation_dict_type_is_stat(self):
        # Arrange
        result = self._annotation_result()
        # Act
        ann = result.to_annotation_dict()
        # Assert
        assert ann["type"] == "stat"

    def test_annotation_dict_uses_plot_id(self):
        # Arrange
        result = self._annotation_result()
        # Act
        ann = result.to_annotation_dict()
        # Assert
        assert ann["id"] == "plot_0"

    def test_annotation_dict_records_test_name(self):
        # Arrange
        result = self._annotation_result()
        # Act
        ann = result.to_annotation_dict()
        # Assert
        assert ann["statResult"]["test_name"] == "pearson"

    def test_annotation_dict_records_formatted_output(self):
        # Arrange
        result = self._annotation_result()
        # Act
        ann = result.to_annotation_dict()
        # Assert
        assert ann["statResult"]["formatted_output"] == "r = 0.850***"

    def test_annotation_dict_records_group_names(self):
        # Arrange
        result = self._annotation_result()
        # Act
        ann = result.to_annotation_dict()
        # Assert
        assert ann["statResult"]["group1"] == "Control"

    def test_annotation_dict_with_position_records_x(self):
        # Arrange
        position = Position(x=10.0, y=20.0, unit="mm")
        positioning = StatPositioning(mode="absolute", position=position)
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.0},
            p_value=0.01,
            stars="**",
            positioning=positioning,
        )
        # Act
        ann = result.to_annotation_dict()
        # Assert
        assert ann["position"]["x"] == 10.0

    def test_annotation_dict_with_position_records_unit(self):
        # Arrange
        position = Position(x=10.0, y=20.0, unit="mm")
        positioning = StatPositioning(mode="absolute", position=position)
        result = StatResult(
            test_type="t-test",
            test_category="parametric",
            statistic={"name": "t", "value": 3.0},
            p_value=0.01,
            stars="**",
            positioning=positioning,
        )
        # Act
        ann = result.to_annotation_dict()
        # Assert
        assert ann["position"]["unit"] == "mm"

    def test_serialization_round_trip_preserves_test_type(self):
        # Arrange
        original = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
            effect_size={"name": "r_squared", "value": 0.7225},
            ci_95=[0.65, 0.95],
        )
        # Act
        restored = StatResult.from_json(original.to_json())
        # Assert
        assert restored.test_type == original.test_type

    def test_serialization_round_trip_preserves_statistic(self):
        # Arrange
        original = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
            effect_size={"name": "r_squared", "value": 0.7225},
            ci_95=[0.65, 0.95],
        )
        # Act
        restored = StatResult.from_json(original.to_json())
        # Assert
        assert restored.statistic["value"] == original.statistic["value"]

    def test_serialization_round_trip_preserves_effect_size(self):
        # Arrange
        original = StatResult(
            test_type="pearson",
            test_category="correlation",
            statistic={"name": "r", "value": 0.85},
            p_value=0.001,
            stars="***",
            effect_size={"name": "r_squared", "value": 0.7225},
            ci_95=[0.65, 0.95],
        )
        # Act
        restored = StatResult.from_json(original.to_json())
        # Assert
        assert restored.effect_size["value"] == original.effect_size["value"]


class TestCreateStatResult:
    """Test create_stat_result convenience function."""

    def test_create_basic_sets_test_type(self):
        # Arrange
        kwargs = dict(
            test_type="pearson",
            statistic_name="r",
            statistic_value=0.85,
            p_value=0.0001,
        )
        # Act
        result = create_stat_result(**kwargs)
        # Assert
        assert result.test_type == "pearson"

    def test_create_basic_maps_category(self):
        # Arrange
        kwargs = dict(
            test_type="pearson",
            statistic_name="r",
            statistic_value=0.85,
            p_value=0.0001,
        )
        # Act
        result = create_stat_result(**kwargs)
        # Assert
        assert result.test_category == "correlation"

    def test_create_basic_sets_statistic_value(self):
        # Arrange
        kwargs = dict(
            test_type="pearson",
            statistic_name="r",
            statistic_value=0.85,
            p_value=0.0001,
        )
        # Act
        result = create_stat_result(**kwargs)
        # Assert
        assert result.statistic["value"] == 0.85

    def test_create_basic_assigns_stars(self):
        # Arrange
        kwargs = dict(
            test_type="pearson",
            statistic_name="r",
            statistic_value=0.85,
            p_value=0.0001,
        )
        # Act
        result = create_stat_result(**kwargs)
        # Assert
        assert result.stars == "***"

    def test_create_with_kwargs_sets_test_type(self):
        # Arrange
        kwargs = dict(
            test_type="t-test",
            statistic_name="t",
            statistic_value=3.45,
            p_value=0.002,
            effect_size={"name": "cohens_d", "value": 0.85},
        )
        # Act
        result = create_stat_result(**kwargs)
        # Assert
        assert result.test_type == "t-test"

    def test_create_with_kwargs_records_effect_size(self):
        # Arrange
        kwargs = dict(
            test_type="t-test",
            statistic_name="t",
            statistic_value=3.45,
            p_value=0.002,
            effect_size={"name": "cohens_d", "value": 0.85},
        )
        # Act
        result = create_stat_result(**kwargs)
        # Assert
        assert result.effect_size["value"] == 0.85

    def test_create_category_mapping_correlation(self):
        # Arrange
        kwargs = dict(
            test_type="spearman",
            statistic_name="rho",
            statistic_value=0.72,
            p_value=0.01,
        )
        # Act
        result = create_stat_result(**kwargs)
        # Assert
        assert result.test_category == "correlation"

    def test_create_category_mapping_parametric(self):
        # Arrange
        kwargs = dict(
            test_type="anova", statistic_name="F", statistic_value=5.23, p_value=0.01
        )
        # Act
        result = create_stat_result(**kwargs)
        # Assert
        assert result.test_category == "parametric"

    def test_create_category_mapping_nonparametric(self):
        # Arrange
        kwargs = dict(
            test_type="mannwhitney",
            statistic_name="U",
            statistic_value=150.0,
            p_value=0.03,
        )
        # Act
        result = create_stat_result(**kwargs)
        # Assert
        assert result.test_category == "non-parametric"

    def test_create_category_mapping_unknown(self):
        # Arrange
        kwargs = dict(
            test_type="custom_test",
            statistic_name="X",
            statistic_value=1.0,
            p_value=0.05,
        )
        # Act
        result = create_stat_result(**kwargs)
        # Assert
        assert result.test_category == "other"


# EOF
