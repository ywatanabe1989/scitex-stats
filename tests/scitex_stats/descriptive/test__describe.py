#!/usr/bin/env python3
# Time-stamp: "2025-11-11"

"""Tests for scitex.stats.descriptive._describe module."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scitex_stats.descriptive._describe import describe, verify_non_leakage


class TestDescribe:
    """Test describe function."""

    def test_basic_describe_shape_described(self):
        """Test basic descriptive statistics."""
        # Arrange
        x = torch.randn(10, 100)
        # Act
        described, names = describe(x, dim=-1)
        # Assert
        assert described.shape == (
            10,
            7,
        ), f"Expected shape (10, 7), got {described.shape}"

    def test_basic_describe_names(self):
        """Test basic descriptive statistics."""
        # Arrange
        x = torch.randn(10, 100)
        # Act
        described, names = describe(x, dim=-1)
        # Assert
        assert len(names) == 7, "Should return 7 stat names"

    def test_basic_describe_mean_names(self):
        """Test basic descriptive statistics."""
        # Arrange
        x = torch.randn(10, 100)
        # Act
        described, names = describe(x, dim=-1)
        # Assert
        assert "mean" in names

    def test_basic_describe_std_names(self):
        """Test basic descriptive statistics."""
        # Arrange
        x = torch.randn(10, 100)
        # Act
        described, names = describe(x, dim=-1)
        # Assert
        assert "std" in names

    def test_basic_describe_median_names(self):
        """Test basic descriptive statistics."""
        # Arrange
        x = torch.randn(10, 100)
        # Act
        described, names = describe(x, dim=-1)
        # Assert
        assert "median" in names

    def test_with_nans_all_isnan_described_torch(self):
        """Test with NaN values."""
        # Arrange
        x = torch.randn(5, 50)
        x[0, 10:20] = float("nan")
        # Act
        described, names = describe(x, dim=-1)
        # Assert
        assert not torch.isnan(described).all(), "Should compute valid statistics"

    def test_different_dims_shape_desc1(self):
        """Test with different dimensions."""
        # Arrange
        x = torch.randn(4, 8, 16)
        # Act
        desc1, _ = describe(x, dim=-1)
        # Assert
        assert desc1.shape == (4, 8, 7)
        desc2, _ = describe(x, dim=(1, 2))

    def test_different_dims_shape_desc2(self):
        """Test with different dimensions."""
        # Arrange
        x = torch.randn(4, 8, 16)
        desc1, _ = describe(x, dim=-1)
        # Act
        desc2, _ = describe(x, dim=(1, 2))
        # Assert
        assert desc2.shape == (4, 7)

    def test_keepdims_shape_described(self):
        """Test keepdims option."""
        # Arrange
        x = torch.randn(5, 10, 20)
        # Act
        described, _ = describe(x, dim=-1, keepdims=True)
        # Assert
        assert described.shape == (5, 10, 1, 7), "keepdims should preserve dimensions"

    def test_all_funcs_names(self):
        """Test with funcs='all'."""
        # Arrange
        x = torch.randn(5, 20)
        # Act
        described, names = describe(x, dim=-1, funcs="all")
        # Assert
        assert len(names) > 7, "Should return all available functions"

    def test_all_funcs_max_names(self):
        """Test with funcs='all'."""
        # Arrange
        x = torch.randn(5, 20)
        # Act
        described, names = describe(x, dim=-1, funcs="all")
        # Assert
        assert "max" in names

    def test_all_funcs_min_names(self):
        """Test with funcs='all'."""
        # Arrange
        x = torch.randn(5, 20)
        # Act
        described, names = describe(x, dim=-1, funcs="all")
        # Assert
        assert "min" in names

    def test_all_funcs_count_names(self):
        """Test with funcs='all'."""
        # Arrange
        x = torch.randn(5, 20)
        # Act
        described, names = describe(x, dim=-1, funcs="all")
        # Assert
        assert "count" in names

    def test_custom_funcs_names(self):
        """Test with custom function list."""
        # Arrange
        x = torch.randn(5, 20)
        custom_funcs = ["nanmean", "nanstd", "nanmax", "nanmin"]
        # Act
        described, names = describe(x, dim=-1, funcs=custom_funcs)
        # Assert
        assert len(names) == 4

    def test_custom_funcs_names_custom_funcs(self):
        """Test with custom function list."""
        # Arrange
        x = torch.randn(5, 20)
        custom_funcs = ["nanmean", "nanstd", "nanmax", "nanmin"]
        # Act
        described, names = describe(x, dim=-1, funcs=custom_funcs)
        # Assert
        assert names == custom_funcs

    def test_numpy_input_described_ndarray(self):
        """Test with numpy array input."""
        # Arrange
        x = np.random.randn(5, 20)
        # Act
        described, names = describe(x, dim=-1)
        # Assert
        assert isinstance(described, np.ndarray)

    def test_numpy_input_shape_described(self):
        """Test with numpy array input."""
        # Arrange
        x = np.random.randn(5, 20)
        # Act
        described, names = describe(x, dim=-1)
        # Assert
        assert described.shape == (5, 7)


class TestVerifyNonLeakage:
    """Test verify_non_leakage function."""

    def test_no_leakage_verify_non_leakage(self):
        """Test that no information leaks across batch."""
        # Arrange
        # Act
        x = torch.randn(10, 5, 20)
        # Assert
        assert verify_non_leakage(x, dim=(1, 2))


class TestDescribeEdgeCases:
    """Test edge cases."""

    def test_all_nan_shape_described(self):
        """Test with all NaN values."""
        # Arrange
        x = torch.full((5, 10), float("nan"))
        # Act
        described, names = describe(x, dim=-1)
        # Assert
        assert described.shape == (5, 7)

    def test_single_value_shape_described(self):
        """Test with single value."""
        # Arrange
        x = torch.randn(5, 1)
        # Act
        described, _ = describe(x, dim=-1)
        # Assert
        assert described.shape == (5, 7)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/descriptive/_describe.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-12-27 (refactored)"
# # File: scitex/stats/descriptive/_describe.py
# """
# Comprehensive descriptive statistics.
#
# Uses torch when available (preserves tensor type), falls back to numpy.
# """
#
# from __future__ import annotations
#
# import os
# from typing import List, Optional, Tuple, Union
#
# import numpy as np
#
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
#
# # Optional torch support
# try:
#     import torch
#
#     HAS_TORCH = True
# except ImportError:
#     torch = None
#     HAS_TORCH = False
#
# from ._nan import (
#     nancount,
#     nankurtosis,
#     nanmax,
#     nanmean,
#     nanmin,
#     nanq25,
#     nanq50,
#     nanq75,
#     nanskewness,
#     nanstd,
#     nanvar,
# )
# from ._real import kurtosis, mean, q25, q50, q75, skewness, std
#
#
# def _is_torch_tensor(x):
#     """Check if x is a torch tensor."""
#     return HAS_TORCH and isinstance(x, torch.Tensor)
#
#
# def _normalize_axis(axis, dim):
#     """Normalize axis/dim parameter."""
#     return dim if dim is not None else axis
#
#
# def verify_non_leakage(
#     x,
#     dim: Optional[Union[int, Tuple[int, ...]]] = None,
# ):
#     """Verify that statistics computation doesn't leak information across samples.
#
#     Parameters
#     ----------
#     x : array-like
#         Input data
#     dim : int or tuple, optional
#         Dimension(s) along which to verify
#
#     Returns
#     -------
#     bool
#         True if no leakage detected
#     """
#     described, _ = describe(x, dim=(1, 2))
#     x_first = x[:1] if _is_torch_tensor(x) else np.asarray(x)[:1]
#     described_first, _ = describe(x_first, dim=dim)
#
#     if _is_torch_tensor(x):
#         assert described_first.shape == described[:1].shape, (
#             f"Shape mismatch: {described_first.shape} != {described[:1].shape}"
#         )
#         torch.testing.assert_close(
#             described_first,
#             described[:1],
#             rtol=1e-5,
#             atol=1e-8,
#             msg="Statistics leak information across samples",
#         )
#     else:
#         assert described_first.shape == described[:1].shape, (
#             f"Shape mismatch: {described_first.shape} != {described[:1].shape}"
#         )
#         np.testing.assert_allclose(
#             described_first,
#             described[:1],
#             rtol=1e-5,
#             atol=1e-8,
#             err_msg="Statistics leak information across samples",
#         )
#     return True
#
#
# def describe(
#     x,
#     axis: int = -1,
#     dim: Optional[Union[int, Tuple[int, ...]]] = None,
#     keepdims: bool = False,
#     funcs: Union[List[str], str] = [
#         "nanmean",
#         "nanstd",
#         "nankurtosis",
#         "nanskewness",
#         "nanq25",
#         "nanq50",
#         "nanq75",
#     ],
#     device=None,
#     batch_size: int = -1,
# ) -> Tuple[np.ndarray, List[str]]:
#     """Compute descriptive statistics.
#
#     Parameters
#     ----------
#     x : array-like
#         Input data (numpy array or torch tensor)
#     axis : int, default=-1
#         Deprecated. Use dim instead
#     dim : int or tuple of ints, optional
#         Dimension(s) along which to compute statistics
#     keepdims : bool, default=False
#         Whether to keep reduced dimensions
#     funcs : list of str or "all"
#         Statistical functions to compute
#     device : optional
#         Device for torch tensors (ignored for numpy)
#     batch_size : int, default=-1
#         Batch size for processing (currently unused)
#
#     Returns
#     -------
#     Tuple[ndarray or Tensor, List[str]]
#         Computed statistics stacked along last dimension and their names
#     """
#     dim = _normalize_axis(axis, dim)
#     dim = (dim,) if isinstance(dim, int) else tuple(dim) if dim is not None else None
#
#     func_names = funcs
#     func_candidates = {
#         "mean": mean,
#         "std": std,
#         "kurtosis": kurtosis,
#         "skewness": skewness,
#         "q25": q25,
#         "q50": q50,
#         "q75": q75,
#         "nanmean": nanmean,
#         "nanstd": nanstd,
#         "nanvar": nanvar,
#         "nankurtosis": nankurtosis,
#         "nanskewness": nanskewness,
#         "nanq25": nanq25,
#         "nanq50": nanq50,
#         "nanq75": nanq75,
#         "nanmax": nanmax,
#         "nanmin": nanmin,
#         "nancount": nancount,
#     }
#
#     if funcs == "all":
#         _funcs = list(func_candidates.values())
#         func_names = list(func_candidates.keys())
#     else:
#         _funcs = [func_candidates[ff] for ff in func_names]
#
#     calculated = [ff(x, dim=dim, keepdims=keepdims) for ff in _funcs]
#
#     if _is_torch_tensor(x):
#         return torch.stack(calculated, dim=-1), func_names
#     else:
#         return np.stack(calculated, axis=-1), func_names
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/descriptive/_describe.py
# --------------------------------------------------------------------------------
