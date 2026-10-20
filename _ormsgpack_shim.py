"""
Pure-Python ormsgpack compatibility shim.
Replaces the native C extension (blocked by Windows Application Control policy)
with a drop-in backed by the 'msgpack' pure-Python library.
This file is automatically copied into the venv by setup.ps1.
Do not delete or edit manually.
"""
import msgpack as _msgpack

# ── Option flag constants (kept for API compatibility, values ignored) ──────
OPT_NON_STR_KEYS          = 1 << 0
OPT_SERIALIZE_UUID        = 1 << 1
OPT_NAIVE_UTC             = 1 << 2
OPT_PASSTHROUGH_DATETIME  = 1 << 3
OPT_PASSTHROUGH_SUBCLASS  = 1 << 4
OPT_SERIALIZE_NUMPY       = 1 << 5
OPT_PASSTHROUGH_DATACLASS = 1 << 6
OPT_PASSTHROUGH_ENUM      = 1 << 7
OPT_PASSTHROUGH_UUID      = 1 << 8
OPT_REPLACE_SURROGATES    = 1 << 9
OPT_APPEND_NEWLINE        = 1 << 10
OPT_INDENT_2              = 1 << 11
OPT_SORT_KEYS             = 1 << 12
OPT_OMIT_MICROSECONDS     = 1 << 13
OPT_UTC_Z                 = 1 << 14
OPT_STRICT_INTEGER        = 1 << 15


# ── Exception classes ───────────────────────────────────────────────────────
class MsgpackEncodeError(Exception):
    """Raised when serialization to msgpack fails."""


class MsgpackDecodeError(Exception):
    """Raised when deserialization from msgpack fails."""


class Ext:
    """Mimics ormsgpack.Ext — wraps a msgpack ExtType."""

    __slots__ = ("_code", "_data")

    def __init__(self, code: int, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Ext data must be bytes-like")
        self._code = code
        self._data = bytes(data)

    @property
    def code(self) -> int:
        return self._code

    @property
    def data(self) -> bytes:
        return self._data

    def __repr__(self) -> str:
        return f"Ext(code={self._code}, data={self._data!r})"


def _make_default(user_default):
    def _default(obj):
        # Direct Ext passed in as the object to serialize
        if isinstance(obj, Ext):
            return _msgpack.ExtType(obj._code, obj._data)
        if user_default is not None:
            result = user_default(obj)
            # langgraph's _msgpack_default returns Ext — convert it
            if isinstance(result, Ext):
                return _msgpack.ExtType(result._code, result._data)
            return result
        raise TypeError(
            f"Object of type {type(obj).__name__} is not msgpack serializable"
        )
    return _default


def _ext_hook(code: int, data: bytes) -> Ext:
    return Ext(code, data)


def packb(obj, *, option=None, default=None) -> bytes:
    """Serialize obj to msgpack bytes."""
    return _msgpack.packb(
        obj,
        default=_make_default(default),
        use_bin_type=True,
        strict_types=False,
    )


def unpackb(data: bytes, *, option=None, ext_hook=None) -> object:
    """Deserialize msgpack bytes."""
    hook = ext_hook if ext_hook is not None else _ext_hook
    return _msgpack.unpackb(
        data,
        raw=False,
        strict_map_key=False,
        ext_hook=lambda code, buf: hook(code, bytes(buf)),
    )
