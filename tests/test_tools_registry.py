"""@tool decorator derives schema; dispatch validates bounds → ErrorResult."""
from harness.core.types import ErrorResult, ImageResult, NumericResult
from harness.tools import REGISTRY, dispatch, get_tool_schemas
from tests.conftest import make_volume


def test_registry_populated():
    assert {"zoom", "z_slice", "z_sweep", "prev_frame", "measure"} <= set(REGISTRY)


def test_schema_derivation_enum():
    schema = REGISTRY["measure"].schema
    assert schema["input_schema"]["properties"]["feature"]["enum"] == [
        "fill_fraction",
        "n_segments",
        "aspect_ratio",
    ]


def test_schema_derivation_required():
    schema = REGISTRY["zoom"].schema
    assert set(schema["input_schema"]["required"]) == {"x", "y", "w", "h"}  # view has default


def test_get_tool_schemas_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        get_tool_schemas(["nonexistent"])


def test_dispatch_validates_range():
    vol = make_volume(shape=(10, 32, 32))
    result = dispatch("z_slice", {"index": 99}, volume=vol)
    assert isinstance(result, ErrorResult)
    assert "out of range" in result.message


def test_dispatch_validates_enum():
    vol = make_volume()
    result = dispatch("measure", {"feature": "bogus"}, volume=vol)
    assert isinstance(result, ErrorResult)


def test_dispatch_missing_required():
    vol = make_volume()
    result = dispatch("zoom", {"x": 0, "y": 0}, volume=vol)
    assert isinstance(result, ErrorResult)
    assert "missing" in result.message


def test_dispatch_unknown_tool():
    result = dispatch("nope", {}, volume=make_volume())
    assert isinstance(result, ErrorResult)


def test_dispatch_success_image():
    vol = make_volume()
    result = dispatch("zoom", {"x": 0, "y": 0, "w": 16, "h": 16}, volume=vol)
    assert isinstance(result, ImageResult)
    assert len(result.b64) > 100


def test_dispatch_success_numeric():
    vol = make_volume()
    result = dispatch("measure", {"feature": "fill_fraction"}, volume=vol)
    assert isinstance(result, NumericResult)
    assert 0.0 <= result.value <= 1.0
