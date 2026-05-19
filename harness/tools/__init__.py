"""
Perception tool framework.

A tool is a pure function `(volume, *, **params) -> ToolResult` (or
`(frame, *, **params)` for tools that need frame context like prev_frame).
The `@tool` decorator registers it and derives the API JSON schema from the
function's type hints + docstring (inspect-ai pattern).

Tool contract — every registered tool must satisfy:
  1. Pure: no I/O outside its arguments; no globals; no model calls.
  2. Typed + bounded: numeric params declare bounds via Annotated[int, Range(...)].
  3. Deterministic: same (volume, params) → same result.
  4. Tested: tests/test_tools_{name}.py with synthetic-volume golden.
  5. Cached: dispatch() content-addresses results on (volume_ref, tool, params).

`dispatch()` validates params against the schema before calling; a violation
becomes an ErrorResult that is fed back to the model, not a Python exception.
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

import numpy as np

from harness.core.render import cached_tool_result, load_volume
from harness.core.types import ErrorResult, FrameInput, ImageResult, NumericResult, ToolResult


# --- Param bounds -----------------------------------------------------------


@dataclass(frozen=True)
class Range:
    """Inclusive lower, exclusive upper bound for a numeric parameter.

    Either bound may be a literal int or one of {"Z","Y","X"} to mean the
    corresponding volume dimension at dispatch time.
    """

    lo: int | str = 0
    hi: int | str | None = None

    def resolve(self, volume_shape: tuple[int, ...]) -> tuple[int, int | None]:
        dims = {"Z": volume_shape[0], "Y": volume_shape[1], "X": volume_shape[2]}
        lo = dims.get(self.lo, self.lo) if isinstance(self.lo, str) else self.lo
        hi = dims.get(self.hi, self.hi) if isinstance(self.hi, str) else self.hi
        return int(lo), (int(hi) if hi is not None else None)


# --- ToolSpec & registry ----------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    name: str
    json_type: str
    enum: tuple[str, ...] | None = None
    range: Range | None = None
    default: Any = inspect.Parameter.empty
    required: bool = True


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: tuple[ParamSpec, ...]
    fn: Callable[..., ToolResult]
    returns: Literal["image", "numeric"]
    needs: Literal["volume", "frame"]

    @property
    def schema(self) -> dict[str, Any]:
        """Anthropic API tool schema."""
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in self.params:
            entry: dict[str, Any] = {"type": p.json_type}
            if p.enum:
                entry["enum"] = list(p.enum)
            props[p.name] = entry
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {"type": "object", "properties": props, "required": required},
        }


REGISTRY: dict[str, ToolSpec] = {}


# --- Schema derivation ------------------------------------------------------


_JSON_TYPE = {int: "integer", float: "number", str: "string", bool: "boolean"}


def _param_spec(name: str, hint: Any, default: Any) -> ParamSpec:
    rng: Range | None = None
    enum: tuple[str, ...] | None = None

    if get_origin(hint) is Annotated:
        base, *meta = get_args(hint)
        for m in meta:
            if isinstance(m, Range):
                rng = m
        hint = base

    if get_origin(hint) is Literal:
        enum = tuple(str(a) for a in get_args(hint))
        json_type = "string"
    else:
        json_type = _JSON_TYPE.get(hint, "string")

    return ParamSpec(
        name=name,
        json_type=json_type,
        enum=enum,
        range=rng,
        default=default,
        required=(default is inspect.Parameter.empty),
    )


def _spec_from_fn(fn: Callable, returns: str) -> ToolSpec:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    params: list[ParamSpec] = []
    needs: str = "volume"
    for i, (pname, p) in enumerate(sig.parameters.items()):
        if i == 0:
            needs = "frame" if pname == "frame" else "volume"
            continue
        if p.kind not in {p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD}:
            continue
        params.append(_param_spec(pname, hints.get(pname, str), p.default))
    return ToolSpec(
        name=fn.__name__,
        description=(inspect.getdoc(fn) or "").strip(),
        params=tuple(params),
        fn=fn,
        returns=returns,  # type: ignore[arg-type]
        needs=needs,  # type: ignore[arg-type]
    )


def tool(returns: Literal["image", "numeric"] = "image"):
    """Register a perception tool. Schema derived from signature + docstring."""

    def _wrap(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        spec = _spec_from_fn(fn, returns)
        REGISTRY[spec.name] = spec
        return fn

    return _wrap


def get_tool_schemas(names: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in names:
        if n not in REGISTRY:
            raise KeyError(f"unknown tool {n!r}; registered: {sorted(REGISTRY)}")
        out.append(REGISTRY[n].schema)
    return out


# --- Dispatch ---------------------------------------------------------------


def _validate(spec: ToolSpec, params: dict[str, Any], volume_shape: tuple[int, ...]) -> str | None:
    for p in spec.params:
        if p.name not in params:
            if p.required:
                return f"missing required parameter {p.name!r}"
            continue
        v = params[p.name]
        if p.enum and str(v) not in p.enum:
            return f"{p.name}={v!r} not in {list(p.enum)}"
        if p.range:
            lo, hi = p.range.resolve(volume_shape)
            if not isinstance(v, (int, float)):
                return f"{p.name} must be numeric, got {type(v).__name__}"
            if v < lo or (hi is not None and v >= hi):
                return f"{p.name}={v} out of range [{lo}, {hi})"
    unknown = set(params) - {p.name for p in spec.params}
    if unknown:
        return f"unknown parameters: {sorted(unknown)}"
    return None


def dispatch(
    name: str,
    params: dict[str, Any],
    *,
    volume: np.ndarray | None = None,
    frame: FrameInput | None = None,
) -> ToolResult:
    """Validate params, call the tool, return its result.

    Validation failures return ErrorResult (fed back to the model) rather than
    raising — the model gets a chance to correct itself.
    """
    if name not in REGISTRY:
        return ErrorResult(message=f"unknown tool {name!r}")
    spec = REGISTRY[name]

    if spec.needs == "volume":
        if volume is None:
            assert frame is not None, "dispatch needs either volume or frame"
            volume = load_volume(frame.volume_ref)
        assert volume is not None
        err = _validate(spec, params, volume.shape)
        if err:
            return ErrorResult(message=err)
        kwargs = _fill_defaults(spec, params)
        return _cached(spec, frame, params, lambda: spec.fn(volume, **kwargs))

    assert frame is not None, f"tool {name!r} needs a FrameInput"
    err = _validate(spec, params, (1, 1, 1))
    if err:
        return ErrorResult(message=err)
    kwargs = _fill_defaults(spec, params)
    return _cached(spec, frame, params, lambda: spec.fn(frame, **kwargs))


def _fill_defaults(spec: ToolSpec, params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in spec.params:
        if p.name in params:
            out[p.name] = params[p.name]
        elif p.default is not inspect.Parameter.empty:
            out[p.name] = p.default
    return out


def _cached(spec: ToolSpec, frame: FrameInput | None, params: dict, compute: Callable[[], ToolResult]) -> ToolResult:
    if frame is None or not frame.volume_ref.exists():
        return compute()
    if spec.returns == "image":

        def _img() -> str:
            r = compute()
            assert isinstance(r, ImageResult)
            return r.b64

        b64 = cached_tool_result(frame.volume_ref, spec.name, params, _img, suffix="b64")
        return ImageResult(b64=b64, caption=f"{spec.name}({params})")

    def _num() -> str:
        r = compute()
        assert isinstance(r, NumericResult)
        return json.dumps({"value": r.value, "unit": r.unit, "note": r.note})

    raw = cached_tool_result(frame.volume_ref, spec.name, params, _num, suffix="json")
    d = json.loads(raw)
    return NumericResult(value=d["value"], unit=d.get("unit", ""), note=d.get("note", ""))


# Import tool modules so their @tool decorators run and populate REGISTRY.
# (Kept at the bottom to avoid circular imports — tool modules import `tool` from here.)
from harness.tools import measure as _measure  # noqa: E402,F401
from harness.tools import prev_frame as _prev  # noqa: E402,F401
from harness.tools import z_slice as _z_slice  # noqa: E402,F401
from harness.tools import zoom as _zoom  # noqa: E402,F401
