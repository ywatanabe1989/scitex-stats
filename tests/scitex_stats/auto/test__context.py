#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for StatContext dataclass.

Tests cover:
- Basic instantiation with valid inputs
- Property computations (n_total, min_n_per_group, effective_paired)
- Post-init validation and defaults
- Dictionary serialization/deserialization
- Construction from data arrays
- Outcome type inference
- Edge cases and error handling
"""

import numpy as np
import pytest

from scitex_stats.auto._context import StatContext, _infer_outcome_type


class TestBasicInstantiation:
    """Tests for basic StatContext creation."""

    def test_minimal_context_n_groups_ctx(self):
        """Test creation with minimal required fields."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.n_groups == 2

    def test_minimal_context_sample_sizes_ctx(self):
        """Test creation with minimal required fields."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.sample_sizes == [30, 32]

    def test_minimal_context_outcome_type_continuous_ctx(self):
        """Test creation with minimal required fields."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.outcome_type == "continuous"

    def test_minimal_context_design_between_ctx(self):
        """Test creation with minimal required fields."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.design == "between"

    def test_between_design_defaults(self):
        """Test that between design sets paired=False by default."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.paired is False

    def test_within_design_defaults(self):
        """Test that within design sets paired=True by default."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="within",
        )
        # Assert
        assert ctx.paired is True

    def test_mixed_design_paired_none(self):
        """Test that mixed design leaves paired as None."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="mixed",
            paired=None,
        )
        # Assert
        assert ctx.paired is None

    def test_default_group_names(self):
        """Test automatic group name generation."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[10, 15, 20],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.group_names == ["Group_1", "Group_2", "Group_3"]

    def test_custom_group_names(self):
        """Test custom group names."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            group_names=["Control", "Treatment"],
        )
        # Assert
        assert ctx.group_names == ["Control", "Treatment"]

    def test_all_outcome_types(self):
        """Test all valid outcome types."""
        # Arrange
        outcome_types = ["continuous", "ordinal", "binary", "categorical"]
        # Act
        # Assert
        for outcome_type in outcome_types:
            ctx = StatContext(
                n_groups=2,
                sample_sizes=[30, 32],
                outcome_type=outcome_type,
                design="between",
            )
            assert ctx.outcome_type == outcome_type

    def test_all_design_types(self):
        """Test all valid design types."""
        # Arrange
        design_types = ["between", "within", "mixed"]
        # Act
        # Assert
        for design in design_types:
            ctx = StatContext(
                n_groups=2,
                sample_sizes=[30, 32],
                outcome_type="continuous",
                design=design,
            )
            assert ctx.design == design


class TestProperties:
    """Tests for computed properties."""

    def test_n_total_two_groups(self):
        """Test total sample size calculation."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.n_total == 62

    def test_n_total_multiple_groups(self):
        """Test total sample size with multiple groups."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=4,
            sample_sizes=[10, 15, 20, 25],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.n_total == 70

    def test_min_n_per_group(self):
        """Test minimum group size calculation."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[10, 25, 15],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.min_n_per_group == 10

    def test_min_n_per_group_equal_sizes(self):
        """Test minimum with equal group sizes."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.min_n_per_group == 20

    def test_effective_paired_explicit_true(self):
        """Test effective_paired when explicitly set to True."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="between",
            paired=True,
        )
        # Assert
        assert ctx.effective_paired is True

    def test_effective_paired_explicit_false(self):
        """Test effective_paired when explicitly set to False."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="within",
            paired=False,
        )
        # Assert
        assert ctx.effective_paired is False

    def test_effective_paired_from_within_design(self):
        """Test effective_paired inferred from within design."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="within",
        )
        # Assert
        assert ctx.effective_paired is True

    def test_effective_paired_from_between_design(self):
        """Test effective_paired inferred from between design."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.effective_paired is False

    def test_effective_paired_mixed_design_none(self):
        """Test effective_paired is None for mixed design."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="mixed",
        )
        # Assert
        assert ctx.effective_paired is None


class TestValidation:
    """Tests for validation and error handling."""

    def test_sample_sizes_mismatch_raises_error(self):
        """Test that mismatched sample_sizes length raises ValueError."""
        # Arrange
        # Act
        # Assert
        with pytest.raises(
            ValueError, match="sample_sizes length.*must match.*n_groups"
        ):
            StatContext(
                n_groups=3,
                sample_sizes=[30, 32],
                outcome_type="continuous",
                design="between",
            )

    def test_sample_sizes_too_many_raises_error(self):
        """Test that too many sample sizes raises ValueError."""
        # Arrange
        # Act
        # Assert
        with pytest.raises(
            ValueError, match="sample_sizes length.*must match.*n_groups"
        ):
            StatContext(
                n_groups=2,
                sample_sizes=[30, 32, 25],
                outcome_type="continuous",
                design="between",
            )

    def test_empty_sample_sizes_n_total_ctx(self):
        """Test handling of empty sample_sizes."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=0,
            sample_sizes=[],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.n_total == 0

    def test_empty_sample_sizes_min_n_per_group_ctx(self):
        """Test handling of empty sample_sizes."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=0,
            sample_sizes=[],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.min_n_per_group == 0


class TestDictionarySerialization:
    """Tests for dictionary conversion."""

    def test_to_dict_basic_groups(self):
        """Test conversion to dictionary."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        d = ctx.to_dict()
        # Assert
        assert d["n_groups"] == 2

    def test_to_dict_basic_sample_sizes(self):
        """Test conversion to dictionary."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        d = ctx.to_dict()
        # Assert
        assert d["sample_sizes"] == [30, 32]

    def test_to_dict_basic_continuous_outcome_type(self):
        """Test conversion to dictionary."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        d = ctx.to_dict()
        # Assert
        assert d["outcome_type"] == "continuous"

    def test_to_dict_basic_between_design(self):
        """Test conversion to dictionary."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        d = ctx.to_dict()
        # Assert
        assert d["design"] == "between"

    def test_to_dict_basic_paired(self):
        """Test conversion to dictionary."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        d = ctx.to_dict()
        # Assert
        assert d["paired"] is False

    def test_to_dict_all_fields_has_control_group(self):
        """Test dictionary includes all fields."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="within",
            has_control_group=True,
            n_factors=2,
            normality_ok=True,
            variance_homogeneity_ok=False,
            control_group_name="baseline",
        )
        # Act
        d = ctx.to_dict()
        # Assert
        assert d["has_control_group"] is True

    def test_to_dict_all_fields_factors(self):
        """Test dictionary includes all fields."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="within",
            has_control_group=True,
            n_factors=2,
            normality_ok=True,
            variance_homogeneity_ok=False,
            control_group_name="baseline",
        )
        # Act
        d = ctx.to_dict()
        # Assert
        assert d["n_factors"] == 2

    def test_to_dict_all_fields_normality_ok(self):
        """Test dictionary includes all fields."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="within",
            has_control_group=True,
            n_factors=2,
            normality_ok=True,
            variance_homogeneity_ok=False,
            control_group_name="baseline",
        )
        # Act
        d = ctx.to_dict()
        # Assert
        assert d["normality_ok"] is True

    def test_to_dict_all_fields_variance_homogeneity_ok(self):
        """Test dictionary includes all fields."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="within",
            has_control_group=True,
            n_factors=2,
            normality_ok=True,
            variance_homogeneity_ok=False,
            control_group_name="baseline",
        )
        # Act
        d = ctx.to_dict()
        # Assert
        assert d["variance_homogeneity_ok"] is False

    def test_to_dict_all_fields_baseline_control_group_name(self):
        """Test dictionary includes all fields."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="within",
            has_control_group=True,
            n_factors=2,
            normality_ok=True,
            variance_homogeneity_ok=False,
            control_group_name="baseline",
        )
        # Act
        d = ctx.to_dict()
        # Assert
        assert d["control_group_name"] == "baseline"

    def test_from_dict_roundtrip_n_groups_ctx2_ctx1(self):
        """Test dictionary roundtrip conversion."""
        # Arrange
        ctx1 = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
        )
        d = ctx1.to_dict()
        # Act
        ctx2 = StatContext.from_dict(d)
        # Assert
        assert ctx2.n_groups == ctx1.n_groups

    def test_from_dict_roundtrip_sample_sizes_ctx2_ctx1(self):
        """Test dictionary roundtrip conversion."""
        # Arrange
        ctx1 = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
        )
        d = ctx1.to_dict()
        # Act
        ctx2 = StatContext.from_dict(d)
        # Assert
        assert ctx2.sample_sizes == ctx1.sample_sizes

    def test_from_dict_roundtrip_outcome_type_ctx2_ctx1(self):
        """Test dictionary roundtrip conversion."""
        # Arrange
        ctx1 = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
        )
        d = ctx1.to_dict()
        # Act
        ctx2 = StatContext.from_dict(d)
        # Assert
        assert ctx2.outcome_type == ctx1.outcome_type

    def test_from_dict_roundtrip_design_ctx2_ctx1(self):
        """Test dictionary roundtrip conversion."""
        # Arrange
        ctx1 = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
        )
        d = ctx1.to_dict()
        # Act
        ctx2 = StatContext.from_dict(d)
        # Assert
        assert ctx2.design == ctx1.design

    def test_from_dict_roundtrip_normality_ok_ctx2_ctx1(self):
        """Test dictionary roundtrip conversion."""
        # Arrange
        ctx1 = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
        )
        d = ctx1.to_dict()
        # Act
        ctx2 = StatContext.from_dict(d)
        # Assert
        assert ctx2.normality_ok == ctx1.normality_ok


class TestFromData:
    """Tests for construction from data arrays."""

    def test_from_data_two_groups_n_groups_ctx(self):
        """Test creating context from data with two groups."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        # Act
        ctx = StatContext.from_data(y, group)
        # Assert
        assert ctx.n_groups == 2

    def test_from_data_two_groups_sample_sizes_ctx(self):
        """Test creating context from data with two groups."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        # Act
        ctx = StatContext.from_data(y, group)
        # Assert
        assert ctx.sample_sizes == [3, 3]

    def test_from_data_two_groups_group_names_ctx(self):
        """Test creating context from data with two groups."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        # Act
        ctx = StatContext.from_data(y, group)
        # Assert
        assert ctx.group_names == ["A", "B"]

    def test_from_data_unequal_groups_n_groups_ctx(self):
        """Test with unequal group sizes."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        group = np.array(["X", "X", "X", "X", "X", "Y", "Y", "Y"])
        # Act
        ctx = StatContext.from_data(y, group)
        # Assert
        assert ctx.n_groups == 2

    def test_from_data_unequal_groups_sample_sizes_ctx(self):
        """Test with unequal group sizes."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        group = np.array(["X", "X", "X", "X", "X", "Y", "Y", "Y"])
        # Act
        ctx = StatContext.from_data(y, group)
        # Assert
        assert ctx.sample_sizes == [5, 3]

    def test_from_data_three_groups_n_groups_ctx(self):
        """Test with three groups."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        group = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
        # Act
        ctx = StatContext.from_data(y, group)
        # Assert
        assert ctx.n_groups == 3

    def test_from_data_three_groups_sample_sizes_ctx(self):
        """Test with three groups."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        group = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
        # Act
        ctx = StatContext.from_data(y, group)
        # Assert
        assert ctx.sample_sizes == [3, 3, 3]

    def test_from_data_three_groups_group_names_ctx(self):
        """Test with three groups."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        group = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
        # Act
        ctx = StatContext.from_data(y, group)
        # Assert
        assert ctx.group_names == ["A", "B", "C"]

    def test_from_data_custom_design_within_ctx(self):
        """Test specifying design type."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        # Act
        ctx = StatContext.from_data(y, group, design="within")
        # Assert
        assert ctx.design == "within"

    def test_from_data_custom_design_paired_ctx(self):
        """Test specifying design type."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        # Act
        ctx = StatContext.from_data(y, group, design="within")
        # Assert
        assert ctx.paired is True

    def test_from_data_custom_outcome_type(self):
        """Test specifying outcome type."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        # Act
        ctx = StatContext.from_data(y, group, outcome_type="ordinal")
        # Assert
        assert ctx.outcome_type == "ordinal"

    def test_from_data_infers_continuous(self):
        """Test automatic inference of continuous outcome."""
        # Arrange
        y = np.array([1.5, 2.3, 3.7, 4.1, 5.9, 6.2])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        # Act
        ctx = StatContext.from_data(y, group)
        # Assert
        assert ctx.outcome_type == "continuous"

    def test_from_data_additional_kwargs_has_control_group_ctx(self):
        """Test passing additional arguments."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        # Act
        ctx = StatContext.from_data(
            y,
            group,
            has_control_group=True,
            normality_ok=False,
        )
        # Assert
        assert ctx.has_control_group is True

    def test_from_data_additional_kwargs_normality_ok_ctx(self):
        """Test passing additional arguments."""
        # Arrange
        y = np.array([1, 2, 3, 4, 5, 6])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        # Act
        ctx = StatContext.from_data(
            y,
            group,
            has_control_group=True,
            normality_ok=False,
        )
        # Assert
        assert ctx.normality_ok is False


class TestInferOutcomeType:
    """Tests for outcome type inference."""

    def test_infer_binary_01(self):
        """Test binary detection with 0 and 1."""
        # Arrange
        y = np.array([0, 1, 0, 1, 1, 0])
        # Act
        outcome_type = _infer_outcome_type(y)
        # Assert
        assert outcome_type == "binary"

    def test_infer_binary_with_nan(self):
        """Test binary detection ignores NaN."""
        # Arrange
        y = np.array([0, 1, np.nan, 1, 0, 1])
        # Act
        outcome_type = _infer_outcome_type(y)
        # Assert
        assert outcome_type == "binary"

    def test_infer_continuous_floats(self):
        """Test continuous detection with float values."""
        # Arrange
        y = np.array([1.5, 2.3, 3.7, 4.1, 5.9])
        # Act
        outcome_type = _infer_outcome_type(y)
        # Assert
        assert outcome_type == "continuous"

    def test_infer_continuous_many_values(self):
        """Test continuous detection with many unique values."""
        # Arrange
        y = np.arange(100) + 0.1
        # Act
        outcome_type = _infer_outcome_type(y)
        # Assert
        assert outcome_type == "continuous"

    def test_infer_ordinal_few_integers(self):
        """Test ordinal detection with few integer values."""
        # Arrange
        y = np.array([1, 2, 3, 2, 1, 3, 2, 1])
        # Act
        outcome_type = _infer_outcome_type(y)
        # Assert
        assert outcome_type == "ordinal"

    def test_infer_categorical_strings(self):
        """Test categorical detection with string values."""
        # Arrange
        y = np.array(["red", "blue", "green", "red", "blue"], dtype=object)
        # Act
        outcome_type = _infer_outcome_type(y)
        # Assert
        assert outcome_type == "categorical" or outcome_type == "ordinal"

    def test_infer_continuous_boundary(self):
        """Test boundary between ordinal and continuous."""
        # Arrange
        y = np.arange(21)
        # Act
        outcome_type = _infer_outcome_type(y)
        # Assert
        assert outcome_type == "continuous"


class TestAssumptionFlags:
    """Tests for assumption check flags."""

    def test_normality_flags_normality_ok_ctx(self):
        """Test normality flag settings."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
        )
        # Assert
        assert ctx.normality_ok is True

    def test_variance_homogeneity_flags(self):
        """Test variance homogeneity flag settings."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            variance_homogeneity_ok=False,
        )
        # Assert
        assert ctx.variance_homogeneity_ok is False

    def test_both_assumptions_none_by_default_normality_ok_ctx(self):
        """Test that assumptions are None by default."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.normality_ok is None

    def test_both_assumptions_none_by_default_variance_homogeneity_ok_ctx(self):
        """Test that assumptions are None by default."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.variance_homogeneity_ok is None


class TestControlGroup:
    """Tests for control group handling."""

    def test_has_control_group_true_has_control_group_ctx(self):
        """Test context with control group."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            has_control_group=True,
            control_group_name="Control",
        )
        # Assert
        assert ctx.has_control_group is True

    def test_has_control_group_true_control_group_name_ctx(self):
        """Test context with control group."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            has_control_group=True,
            control_group_name="Control",
        )
        # Assert
        assert ctx.control_group_name == "Control"

    def test_no_control_group_by_default_has_control_group_ctx(self):
        """Test that has_control_group is False by default."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.has_control_group is False

    def test_no_control_group_by_default_control_group_name_ctx(self):
        """Test that has_control_group is False by default."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.control_group_name is None


class TestFactorStructure:
    """Tests for factor structure."""

    def test_one_factor_default(self):
        """Test that n_factors defaults to 1."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Assert
        assert ctx.n_factors == 1

    def test_two_factors_n_factors_ctx(self):
        """Test two-factor design."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=4,
            sample_sizes=[20, 20, 20, 20],
            outcome_type="continuous",
            design="between",
            n_factors=2,
        )
        # Assert
        assert ctx.n_factors == 2

    def test_three_factors_n_factors_ctx(self):
        """Test three-factor design."""
        # Arrange
        # Act
        ctx = StatContext(
            n_groups=8,
            sample_sizes=[10] * 8,
            outcome_type="continuous",
            design="mixed",
            n_factors=3,
        )
        # Assert
        assert ctx.n_factors == 3


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_context.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_context.py
#
# """
# Statistical Context - Data structure for automatic test selection.
#
# This module defines StatContext which captures all relevant information
# about data and experimental design needed to determine which statistical
# tests are applicable.
#
# The StatContext is built from figure/axis metadata or directly from data,
# and is used by the test selection engine to filter applicable tests.
# """
#
# from __future__ import annotations
#
# from dataclasses import dataclass, field
# from typing import List, Literal, Optional, Dict, Any
#
# import numpy as np
#
#
# # =============================================================================
# # Type Aliases
# # =============================================================================
#
# OutcomeType = Literal["continuous", "ordinal", "binary", "categorical"]
# DesignType = Literal["between", "within", "mixed"]
#
#
# # =============================================================================
# # StatContext
# # =============================================================================
#
#
# @dataclass
# class StatContext:
#     """
#     Statistical context for determining which tests are applicable.
#
#     This dataclass captures all the information needed to decide which
#     statistical tests can be applied to the current data/figure context.
#     It is used by check_applicable() to filter the test registry.
#
#     Parameters
#     ----------
#     n_groups : int
#         Number of groups/levels to compare (e.g., 2 for A vs B).
#     sample_sizes : list of int
#         Sample sizes per group in the same order as the groups.
#     outcome_type : OutcomeType
#         Type of outcome variable:
#         - "continuous": numeric, interval/ratio scale
#         - "ordinal": ordered categories, ranks
#         - "binary": 0/1 or yes/no
#         - "categorical": nominal with >= 2 categories
#     design : DesignType
#         Overall experimental design:
#         - "between": independent groups
#         - "within": repeated measures / paired
#         - "mixed": mixed design (some within, some between)
#     paired : bool or None
#         Whether the comparison is explicitly paired.
#         If None, inferred from design.
#     has_control_group : bool
#         Whether a control group is identifiable (for Dunnett etc.).
#     n_factors : int
#         Number of factors; 1 for one-way, 2 for two-way, etc.
#     normality_ok : bool or None
#         Result of normality check if available.
#         True = normal, False = non-normal, None = unknown.
#     variance_homogeneity_ok : bool or None
#         Result of homogeneity test (e.g., Levene).
#         True = homogeneous, False = heteroscedastic, None = unknown.
#     missing_allowed : bool
#         Whether missing data is allowed for the chosen method.
#     group_names : list of str, optional
#         Names of the groups for display purposes.
#     control_group_name : str, optional
#         Name of the control group if has_control_group is True.
#
#     Examples
#     --------
#     >>> # Two-group independent comparison
#     >>> ctx = StatContext(
#     ...     n_groups=2,
#     ...     sample_sizes=[30, 32],
#     ...     outcome_type="continuous",
#     ...     design="between",
#     ...     paired=False,
#     ...     has_control_group=False,
#     ...     n_factors=1
#     ... )
#
#     >>> # Three-group repeated measures
#     >>> ctx = StatContext(
#     ...     n_groups=3,
#     ...     sample_sizes=[20, 20, 20],
#     ...     outcome_type="continuous",
#     ...     design="within",
#     ...     paired=True,
#     ...     has_control_group=True,
#     ...     n_factors=1,
#     ...     control_group_name="baseline"
#     ... )
#
#     >>> # Check if data appears normal
#     >>> ctx.normality_ok = True
#     >>> ctx.variance_homogeneity_ok = True
#     """
#
#     # Core group information
#     n_groups: int
#     sample_sizes: List[int]
#
#     # Data characteristics
#     outcome_type: OutcomeType
#
#     # Experimental design
#     design: DesignType
#     paired: Optional[bool] = None
#
#     # Control group (for Dunnett, etc.)
#     has_control_group: bool = False
#
#     # Factor structure (for ANOVA designs)
#     n_factors: int = 1
#
#     # Assumption checks (None = not tested)
#     normality_ok: Optional[bool] = None
#     variance_homogeneity_ok: Optional[bool] = None
#
#     # Missing data handling
#     missing_allowed: bool = False
#
#     # Group metadata
#     group_names: Optional[List[str]] = None
#     control_group_name: Optional[str] = None
#
#     def __post_init__(self):
#         """Validate and set defaults."""
#         # Ensure sample_sizes matches n_groups
#         if len(self.sample_sizes) != self.n_groups:
#             raise ValueError(
#                 f"sample_sizes length ({len(self.sample_sizes)}) must match "
#                 f"n_groups ({self.n_groups})"
#             )
#
#         # Infer paired from design if not specified
#         if self.paired is None:
#             if self.design == "within":
#                 self.paired = True
#             elif self.design == "between":
#                 self.paired = False
#             # mixed design: leave as None, tests must handle both
#
#         # Default group names if not provided
#         if self.group_names is None:
#             self.group_names = [f"Group_{i+1}" for i in range(self.n_groups)]
#
#     @property
#     def n_total(self) -> int:
#         """Total sample size across all groups."""
#         return sum(self.sample_sizes)
#
#     @property
#     def min_n_per_group(self) -> int:
#         """Minimum sample size per group."""
#         return min(self.sample_sizes) if self.sample_sizes else 0
#
#     @property
#     def effective_paired(self) -> Optional[bool]:
#         """
#         Effective paired status, considering design.
#
#         Returns
#         -------
#         bool or None
#             True if paired/within, False if unpaired/between, None if unknown.
#         """
#         if self.paired is not None:
#             return self.paired
#         if self.design == "within":
#             return True
#         if self.design == "between":
#             return False
#         return None  # mixed or unknown
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary for JSON serialization."""
#         return {
#             "n_groups": self.n_groups,
#             "sample_sizes": self.sample_sizes,
#             "outcome_type": self.outcome_type,
#             "design": self.design,
#             "paired": self.paired,
#             "has_control_group": self.has_control_group,
#             "n_factors": self.n_factors,
#             "normality_ok": self.normality_ok,
#             "variance_homogeneity_ok": self.variance_homogeneity_ok,
#             "missing_allowed": self.missing_allowed,
#             "group_names": self.group_names,
#             "control_group_name": self.control_group_name,
#         }
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "StatContext":
#         """Create from dictionary."""
#         return cls(**data)
#
#     @classmethod
#     def from_data(
#         cls,
#         y: np.ndarray,
#         group: np.ndarray,
#         design: DesignType = "between",
#         outcome_type: Optional[OutcomeType] = None,
#         **kwargs,
#     ) -> "StatContext":
#         """
#         Create StatContext from data arrays.
#
#         Parameters
#         ----------
#         y : np.ndarray
#             Outcome values.
#         group : np.ndarray
#             Group labels for each observation.
#         design : DesignType
#             Experimental design.
#         outcome_type : OutcomeType, optional
#             Type of outcome. If None, inferred from data.
#         **kwargs
#             Additional arguments for StatContext.
#
#         Returns
#         -------
#         StatContext
#             Context built from the data.
#
#         Examples
#         --------
#         >>> y = np.array([1, 2, 3, 4, 5, 6])
#         >>> group = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
#         >>> ctx = StatContext.from_data(y, group)
#         >>> ctx.n_groups
#         2
#         """
#         y = np.asarray(y)
#         group = np.asarray(group)
#
#         # Get unique groups and their sizes
#         unique_groups = np.unique(group)
#         n_groups = len(unique_groups)
#         sample_sizes = [int(np.sum(group == g)) for g in unique_groups]
#         group_names = [str(g) for g in unique_groups]
#
#         # Infer outcome type if not provided
#         if outcome_type is None:
#             outcome_type = _infer_outcome_type(y)
#
#         return cls(
#             n_groups=n_groups,
#             sample_sizes=sample_sizes,
#             outcome_type=outcome_type,
#             design=design,
#             group_names=group_names,
#             **kwargs,
#         )
#
#
# def _infer_outcome_type(y: np.ndarray) -> OutcomeType:
#     """
#     Infer outcome type from data.
#
#     Parameters
#     ----------
#     y : np.ndarray
#         Outcome values.
#
#     Returns
#     -------
#     OutcomeType
#         Inferred outcome type.
#     """
#     y = np.asarray(y)
#
#     # Check for categorical first (object dtype like strings)
#     if y.dtype == object:
#         unique = np.unique(y)
#         if len(unique) <= 10:
#             return "categorical"
#         return "categorical"  # Object arrays are always categorical
#
#     # For numeric arrays, filter out NaN
#     try:
#         unique = np.unique(y[~np.isnan(y)])
#     except TypeError:
#         # If isnan fails (e.g., for complex types), use all values
#         unique = np.unique(y)
#
#     # Check for binary
#     if len(unique) == 2:
#         if set(unique).issubset({0, 1}):
#             return "binary"
#
#     # Check for categorical (few unique values)
#     if len(unique) <= 10 and y.dtype == object:
#         return "categorical"
#
#     # Check for ordinal (integer-like with few values)
#     if len(unique) <= 20 and np.allclose(y, y.astype(int), equal_nan=True):
#         return "ordinal"
#
#     # Default to continuous
#     return "continuous"
#
#
# # =============================================================================
# # Public API
# # =============================================================================
#
# __all__ = [
#     "StatContext",
#     "OutcomeType",
#     "DesignType",
# ]
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_context.py
# --------------------------------------------------------------------------------
