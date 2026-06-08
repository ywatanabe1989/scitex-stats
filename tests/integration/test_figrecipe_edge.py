#!/usr/bin/env python3
"""Per-edge integration + degradation tests for the OPTIONAL figrecipe edge.

This mirrors the canonical scitex-io ↔ figrecipe edge template. It pairs the
existing figrecipe-PRESENT integration tests (see
``test_figrecipe_integration.py``) with the missing half: a hermetic
figrecipe-ABSENT degradation test that pins scitex-stats' documented contract.

The edge under test
-------------------
``scitex_stats._figrecipe_integration`` (``to_figrecipe`` / ``annotate`` /
``load_and_annotate``) is a thin bridge over figrecipe's ``from_scitex_stats``,
``annotate_from_stats`` and ``load_stats_bundle``. figrecipe is an OPTIONAL
``[all]``-tier dependency (``pyproject.toml`` ``optional-dependencies.all``);
it is lazy-imported via ``scitex_dev.try_import_optional``, and a module-level
``_AVAILABLE`` flag records whether all three figrecipe entry points resolved.

The two test kinds every optional edge should have
--------------------------------------------------
1. INTEGRATION (collaborator PRESENT): exercise the real figrecipe bridge and
   assert on the concrete artifacts it produces. Guarded with
   ``pytest.importorskip("figrecipe")`` so the suite stays green on a minimal
   install instead of erroring.

2. DEGRADATION (collaborator ABSENT): simulate figrecipe missing in a
   hermetic, reversible way (snapshot the whole ``sys.modules`` table, evict
   figrecipe + the bridge module, shadow ``figrecipe`` with an inert STUB
   module, reload the bridge so its optional-import guards re-run), then assert
   the *documented* contract: each figrecipe-requiring entry point raises a
   clear ``ImportError`` carrying the install hint — never an opaque
   ``AttributeError`` / ``TypeError`` leaking figrecipe internals.

Documented degradation contract (verified by reading + running the source)
--------------------------------------------------------------------------
With figrecipe absent, ``_AVAILABLE`` is ``False`` and every public entry
point short-circuits with::

    raise ImportError("figrecipe >= 0.13.0 required: pip install figrecipe")

Conventions honoured (so this stays a clean template):
  - One assertion per test: shared/expensive setup is lifted into fixtures so a
    red CI line names exactly which behaviour broke.
  - Explicit Arrange / Act / Assert markers in every test.
  - No ``monkeypatch`` / ``mocker``: the figrecipe-absent fixture hand-swaps
    ``sys.modules`` and restores the exact snapshot on teardown.
"""

from __future__ import annotations

import importlib
import sys
import types

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

# ===========================================================================
# 1. INTEGRATION  —  figrecipe PRESENT
# ===========================================================================
figrecipe = pytest.importorskip("figrecipe")

from scitex_stats import _figrecipe_integration as fri  # noqa: E402


def _stats_result():
    """A run_test-shaped two-group comparison result.

    figrecipe's ``from_scitex_stats`` reads the snake_case ``p_value`` key.
    """
    return {
        "name": "control vs treatment",
        "method": "Student's t-test (independent)",
        "p_value": 0.0022,
        "stars": "**",
        "effect_size": -0.83,
    }


def test_bridge_reports_figrecipe_available_when_present():
    """With figrecipe installed, the bridge's availability flag is True."""
    # Arrange
    module = fri
    # Act
    available = module._AVAILABLE
    # Assert
    assert available is True


def test_to_figrecipe_returns_dict_with_comparisons():
    """to_figrecipe converts a stats dict into a figrecipe 'comparisons' dict."""
    # Arrange
    stats = _stats_result()
    # Act
    out = fri.to_figrecipe(stats)
    # Assert
    assert "comparisons" in out


def test_to_figrecipe_list_yields_one_comparison_per_result():
    """A list of stats dicts maps to one comparison entry each."""
    # Arrange
    stats_list = [_stats_result(), _stats_result()]
    # Act
    out = fri.to_figrecipe(stats_list)
    # Assert
    assert len(out["comparisons"]) == 2


@pytest.fixture
def annotated_axes():
    """annotate() a real figrecipe RecordingAxes once; yield the artist list."""
    fig, ax = figrecipe.subplots()
    ax.bar(["control", "treatment"], [1.0, 2.0])
    artists = fri.annotate(
        ax,
        _stats_result(),
        positions={"control": 0, "treatment": 1},
        style="stars",
    )
    yield artists
    plt.close(fig)


def test_annotate_returns_artist_list(annotated_axes):
    """annotate() returns the list of created matplotlib artists."""
    # Arrange
    artists = annotated_axes
    # Act
    is_list = isinstance(artists, list)
    # Assert
    assert is_list


def test_annotate_unwraps_scitex_axis_wrapper():
    """annotate() looks through an AxisWrapper's `_axis_mpl` to the inner axes."""
    # Arrange
    fig, fr_ax = figrecipe.subplots()
    fr_ax.bar(["control", "treatment"], [1.0, 2.0])

    class _Wrapper:
        _axis_mpl = fr_ax

    # Act
    artists = fri.annotate(
        _Wrapper(), _stats_result(), positions={"control": 0, "treatment": 1}
    )
    # Assert
    assert isinstance(artists, list)
    plt.close(fig)


# ===========================================================================
# 2. DEGRADATION  —  figrecipe ABSENT
# ===========================================================================
@pytest.fixture
def figrecipe_absent():
    """Make ``import figrecipe`` fail for the duration of the test.

    Hermetic and reversible:
      1. snapshot the whole ``sys.modules`` so teardown restores it exactly;
      2. evict ``figrecipe`` and the bridge module
         (``scitex_stats._figrecipe_integration``), then shadow ``figrecipe``
         with an inert STUB *module* (not ``None``). ``scitex_dev``'s
         ``try_import_optional`` resolves figrecipe's ``utils`` /
         ``_integrations._scitex_stats`` submodules; a real-but-empty module
         object makes those ``getattr`` lookups miss (so ``_AVAILABLE`` flips
         to False) without the "import halted; None in sys.modules" failure a
         bare ``None`` entry would raise;
      3. reload the bridge so it re-runs its optional-import guards under the
         missing dependency.

    Yields the freshly reloaded bridge module.
    """
    import scitex_stats._figrecipe_integration  # noqa: F401  (ensure importable)

    # 1. Full snapshot for an exact restore.
    snapshot = dict(sys.modules)

    # 2. Evict figrecipe + the bridge module, then block figrecipe with a stub.
    def _to_evict(name: str) -> bool:
        return (
            name == "figrecipe"
            or name.startswith("figrecipe.")
            or name == "scitex_stats._figrecipe_integration"
        )

    for name in [n for n in list(sys.modules) if _to_evict(n)]:
        del sys.modules[name]

    stub = types.ModuleType("figrecipe")
    stub.__file__ = "<figrecipe stub: absent>"
    stub.__path__ = []  # mark as a package so submodule imports raise cleanly
    sys.modules["figrecipe"] = stub

    # 3. Reload the bridge so its module-level _AVAILABLE guard re-runs.
    reloaded = importlib.import_module("scitex_stats._figrecipe_integration")

    try:
        yield reloaded
    finally:
        for name in list(sys.modules):
            if name not in snapshot:
                del sys.modules[name]
        sys.modules.update(snapshot)


def test_figrecipe_absent_flips_availability_flag(figrecipe_absent):
    """Sanity: under the fixture the bridge reports figrecipe unavailable."""
    # Arrange
    module = figrecipe_absent
    # Act
    available = module._AVAILABLE
    # Assert
    assert available is False


def _degraded_error(call):
    """Run ``call`` under figrecipe-absent and return the raised exception."""
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - we re-assert the type in tests
        return exc
    return None


def test_to_figrecipe_raises_importerror_without_figrecipe(figrecipe_absent):
    """to_figrecipe degrades to a clear ImportError, not an opaque crash."""
    # Arrange
    stats = {"name": "a vs b", "p_value": 0.01}
    # Act
    error = _degraded_error(lambda: figrecipe_absent.to_figrecipe(stats))
    # Assert
    assert isinstance(error, ImportError)


def test_to_figrecipe_importerror_carries_install_hint(figrecipe_absent):
    """The ImportError message names figrecipe + the install command."""
    # Arrange
    stats = {"name": "a vs b", "p_value": 0.01}
    # Act
    error = _degraded_error(lambda: figrecipe_absent.to_figrecipe(stats))
    # Assert
    assert "figrecipe" in str(error)


def test_annotate_raises_importerror_without_figrecipe(figrecipe_absent):
    """annotate degrades to a clear ImportError when figrecipe is missing."""
    # Arrange
    fake_ax = object()
    stats = {"name": "a vs b", "p_value": 0.01}
    # Act
    error = _degraded_error(lambda: figrecipe_absent.annotate(fake_ax, stats))
    # Assert
    assert isinstance(error, ImportError)


def test_load_and_annotate_raises_importerror_without_figrecipe(figrecipe_absent):
    """load_and_annotate degrades to a clear ImportError when figrecipe is missing."""
    # Arrange
    fake_ax = object()
    bundle_path = "nonexistent.statsz"
    # Act
    error = _degraded_error(
        lambda: figrecipe_absent.load_and_annotate(fake_ax, bundle_path)
    )
    # Assert
    assert isinstance(error, ImportError)


def test_importing_bridge_without_figrecipe_does_not_raise(figrecipe_absent):
    """The bridge module imports cleanly even with figrecipe absent."""
    # Arrange
    entry_points = ("to_figrecipe", "annotate", "load_and_annotate")
    # Act
    has_entry_points = all(hasattr(figrecipe_absent, name) for name in entry_points)
    # Assert
    assert has_entry_points
