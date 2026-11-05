"""Low-overhead ctypes access to the masked OpenCV VR library."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NativeVRParams:
    alpha: float
    delta: float
    gamma: float
    fixed_point_iterations: int
    sor_iterations: int
    omega: float


class _NativeMaskedProfile(ctypes.Structure):
    _fields_ = [
        ("total_ms", ctypes.c_double),
        ("wrapper_split_ms", ctypes.c_double),
        ("wrapper_merge_ms", ctypes.c_double),
        ("prepare_buffers_ms", ctypes.c_double),
        ("prepare_allocations_ms", ctypes.c_double),
        ("prepare_i1_conversion_ms", ctypes.c_double),
        ("prepare_warp_ms", ctypes.c_double),
        ("prepare_average_subtract_ms", ctypes.c_double),
        ("prepare_derivatives_ms", ctypes.c_double),
        ("prepare_masked_buffers_ms", ctypes.c_double),
        ("initialization_ms", ctypes.c_double),
        ("data_gate_ms", ctypes.c_double),
        ("data_term_ms", ctypes.c_double),
        ("smoothness_horizontal_ms", ctypes.c_double),
        ("smoothness_vertical_ms", ctypes.c_double),
        ("sor_ms", ctypes.c_double),
        ("flow_update_ms", ctypes.c_double),
        ("finalize_ms", ctypes.c_double),
        ("fixed_point_iterations", ctypes.c_int),
        ("sor_iterations_per_fixed_point", ctypes.c_int),
        ("used_uv_interface", ctypes.c_int),
    ]


_PROFILE_TIME_FIELDS = tuple(
    name for name, ctype in _NativeMaskedProfile._fields_
    if ctype is ctypes.c_double
)


def _library_patterns(stem: str) -> tuple[str, ...]:
    if os.name == "nt":
        return (f"{stem}*.dll", f"{stem.removeprefix('lib')}*.dll")
    return (f"{stem}.so*", f"{stem}*.dylib")


def _looks_like_lib_dir(path: Path) -> bool:
    return path.is_dir() and any(
        any(path.glob(pattern))
        for pattern in _library_patterns("libopencv_amv_vr_masked")
    )


def _candidate_lib_dirs(install: Path) -> Iterable[Path]:
    yield install
    yield install / "lib"
    yield install / "lib64"
    yield install / "bin"
    if os.name == "nt" and install.is_dir():
        for pattern in _library_patterns("libopencv_amv_vr_masked"):
            for library in sorted(install.rglob(pattern)):
                yield library.parent


def _normalise_install_candidate(path: Path) -> Path | None:
    """Return an installation root from an install root or lib directory."""

    candidate = path.expanduser().resolve()
    for lib_dir in _candidate_lib_dirs(candidate):
        if _looks_like_lib_dir(lib_dir):
            return candidate
    return None


def _repository_roots(start: Path) -> Iterable[Path]:
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir():
            yield candidate


def _find_private_install(
    start: Path,
    explicit_install_dir: str | Path | None = None,
) -> Path:
    attempted: list[Path] = []

    if explicit_install_dir is not None:
        explicit = Path(explicit_install_dir)
        attempted.append(explicit)
        resolved = _normalise_install_candidate(explicit)
        if resolved is None:
            raise FileNotFoundError(
                "native_install_dir does not contain "
                "the masked OpenCV shared library: "
                f"{explicit.expanduser().resolve()}"
            )
        return resolved

    environment_value = os.environ.get(
        "AMV_PRESSURE_CONDITIONED_OPENCV_DIR",
        "",
    ).strip()
    if environment_value:
        environment_path = Path(environment_value)
        attempted.append(environment_path)
        resolved = _normalise_install_candidate(environment_path)
        if resolved is None:
            raise FileNotFoundError(
                "AMV_PRESSURE_CONDITIONED_OPENCV_DIR does not contain "
                "the masked OpenCV shared library: "
                f"{environment_path.expanduser().resolve()}"
            )
        return resolved

    for repository_root in _repository_roots(start):
        candidates = (
            repository_root
            / "_third_party"
            / "opencv_install_pressure_conditioned_flow",
            repository_root
            / "_third_party"
            / "opencv_install_baseline",
            repository_root
            / "test"
            / "DEEPFLOW"
            / "pressure_state_opencv_vr"
            / "_third_party"
            / "opencv_install_baseline",
        )
        for candidate in candidates:
            attempted.append(candidate)
            resolved = _normalise_install_candidate(candidate)
            if resolved is not None:
                return resolved

    # Search parent directories for a native installation.
    for candidate in (start, *start.parents):
        for relative in (
            Path("_third_party/opencv_install_pressure_conditioned_flow"),
            Path("_third_party/opencv_install_baseline"),
        ):
            install = candidate / relative
            attempted.append(install)
            resolved = _normalise_install_candidate(install)
            if resolved is not None:
                return resolved

    attempted_text = "\n".join(
        f"  - {item.expanduser().resolve()}" for item in attempted
    )
    raise FileNotFoundError(
        "Could not find the private masked OpenCV installation.\n"
        "Build it with tools/build_pressure_conditioned_flow_native.py, "
        "set AMV_PRESSURE_CONDITIONED_OPENCV_DIR, or pass "
        "PressureConditionedFlowConfig(native_install_dir=...).\n"
        f"Searched:\n{attempted_text}"
    )


def _find_lib_dir(install: Path) -> Path:
    for candidate in _candidate_lib_dirs(install):
        if _looks_like_lib_dir(candidate):
            return candidate
    raise FileNotFoundError(f"No masked OpenCV shared-library directory under {install}")


def _find_library(lib_dir: Path, stem: str) -> Path:
    for pattern in _library_patterns(stem):
        matches = sorted(lib_dir.glob(pattern))
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(f"Could not find the shared library {stem} in {lib_dir}")


class NativePressureConditionedVR:
    """ctypes interface to masked variational refinement."""

    def __init__(
        self,
        params: NativeVRParams,
        *,
        install_dir: str | Path | None = None,
        search_start: Path | None = None,
        num_threads: int | None = None,
    ) -> None:
        start = (search_start or Path(__file__).resolve().parent).resolve()
        self.install = _find_private_install(start, install_dir)
        self.lib_dir = _find_lib_dir(self.install)

        self._dll_directories = []
        if os.name == "nt":
            self._dll_directories.append(os.add_dll_directory(str(self.lib_dir)))
            compiler = shutil.which("g++")
            if compiler:
                self._dll_directories.append(
                    os.add_dll_directory(str(Path(compiler).resolve().parent))
                )

        global_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        self._dependencies = []
        for stem in (
            "libopencv_core",
            "libopencv_imgproc",
            "libopencv_video",
        ):
            path = _find_library(self.lib_dir, stem)
            self._dependencies.append(
                ctypes.CDLL(str(path), mode=global_mode)
            )

        self.library_path = _find_library(
            self.lib_dir,
            "libopencv_amv_vr_masked",
        )
        self._lib = ctypes.CDLL(
            str(self.library_path),
            mode=global_mode,
        )
        self._configure_signatures()
        self.supports_uv_interface = hasattr(
            self._lib, "amv_vr_masked_refine_uv_f32"
        )
        self.supports_native_profile = hasattr(
            self._lib, "amv_vr_masked_get_last_profile"
        )
        self.supports_sparse_sor_control = (
            hasattr(self._lib, "amv_vr_masked_set_sparse_sor_enabled")
            and hasattr(self._lib, "amv_vr_masked_get_sparse_sor_enabled")
        )

        self._handle = self._lib.amv_vr_masked_create(
            ctypes.c_float(params.alpha),
            ctypes.c_float(params.delta),
            ctypes.c_float(params.gamma),
            ctypes.c_int(params.fixed_point_iterations),
            ctypes.c_int(params.sor_iterations),
            ctypes.c_float(params.omega),
        )
        if not self._handle:
            raise RuntimeError(self.last_error())

        if num_threads is not None:
            if int(num_threads) <= 0:
                raise ValueError("num_threads must be positive or None")
            self._lib.amv_vr_masked_set_num_threads(int(num_threads))

    def _configure_signatures(self) -> None:
        lib = self._lib
        float_ptr = ctypes.POINTER(ctypes.c_float)

        lib.amv_vr_masked_last_error.argtypes = []
        lib.amv_vr_masked_last_error.restype = ctypes.c_char_p

        lib.amv_vr_masked_opencv_version.argtypes = []
        lib.amv_vr_masked_opencv_version.restype = ctypes.c_char_p

        lib.amv_vr_masked_identity.argtypes = []
        lib.amv_vr_masked_identity.restype = ctypes.c_char_p

        lib.amv_vr_masked_set_num_threads.argtypes = [ctypes.c_int]
        lib.amv_vr_masked_set_num_threads.restype = None

        lib.amv_vr_masked_get_num_threads.argtypes = []
        lib.amv_vr_masked_get_num_threads.restype = ctypes.c_int

        if hasattr(lib, "amv_vr_masked_set_sparse_sor_enabled"):
            lib.amv_vr_masked_set_sparse_sor_enabled.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            lib.amv_vr_masked_set_sparse_sor_enabled.restype = ctypes.c_int

        if hasattr(lib, "amv_vr_masked_get_sparse_sor_enabled"):
            lib.amv_vr_masked_get_sparse_sor_enabled.argtypes = [
                ctypes.c_void_p
            ]
            lib.amv_vr_masked_get_sparse_sor_enabled.restype = ctypes.c_int

        lib.amv_vr_masked_create.argtypes = [
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        lib.amv_vr_masked_create.restype = ctypes.c_void_p

        lib.amv_vr_masked_destroy.argtypes = [ctypes.c_void_p]
        lib.amv_vr_masked_destroy.restype = None

        lib.amv_vr_masked_refine_f32.argtypes = [
            ctypes.c_void_p,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            float_ptr,
        ]
        lib.amv_vr_masked_refine_f32.restype = ctypes.c_int

        if hasattr(lib, "amv_vr_masked_refine_uv_f32"):
            lib.amv_vr_masked_refine_uv_f32.argtypes = [
                ctypes.c_void_p,
                float_ptr,
                float_ptr,
                float_ptr,
                float_ptr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                float_ptr,
                float_ptr,
            ]
            lib.amv_vr_masked_refine_uv_f32.restype = ctypes.c_int

        if hasattr(lib, "amv_vr_masked_profile_abi_version"):
            lib.amv_vr_masked_profile_abi_version.argtypes = []
            lib.amv_vr_masked_profile_abi_version.restype = ctypes.c_int

        if hasattr(lib, "amv_vr_masked_get_last_profile"):
            lib.amv_vr_masked_get_last_profile.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_NativeMaskedProfile),
            ]
            lib.amv_vr_masked_get_last_profile.restype = ctypes.c_int

        lib.amv_vr_masked_collect_garbage.argtypes = [ctypes.c_void_p]
        lib.amv_vr_masked_collect_garbage.restype = None

    @property
    def opencv_version(self) -> str:
        return self._lib.amv_vr_masked_opencv_version().decode(
            "utf-8", errors="replace"
        )

    @property
    def identity(self) -> str:
        return self._lib.amv_vr_masked_identity().decode(
            "utf-8", errors="replace"
        )

    def get_num_threads(self) -> int:
        return int(self._lib.amv_vr_masked_get_num_threads())

    def set_sparse_sor_enabled(self, enabled: bool) -> None:
        if not self.supports_sparse_sor_control:
            raise RuntimeError(
                "Native library does not expose sparse SOR control; rebuild first"
            )
        status = self._lib.amv_vr_masked_set_sparse_sor_enabled(
            self._handle, ctypes.c_int(1 if enabled else 0)
        )
        if status != 0:
            raise RuntimeError(self.last_error())

    def get_sparse_sor_enabled(self) -> bool:
        if not self.supports_sparse_sor_control:
            raise RuntimeError(
                "Native library does not expose sparse SOR control; rebuild first"
            )
        value = int(
            self._lib.amv_vr_masked_get_sparse_sor_enabled(self._handle)
        )
        if value < 0:
            raise RuntimeError(self.last_error())
        return bool(value)

    def last_error(self) -> str:
        value = self._lib.amv_vr_masked_last_error()
        if not value:
            return "Unknown masked OpenCV VR error"
        return value.decode("utf-8", errors="replace")

    @staticmethod
    def _as_f32_2d(array: np.ndarray, name: str) -> np.ndarray:
        output = np.asarray(array, dtype=np.float32)
        if output.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array")
        if not output.flags.c_contiguous:
            output = np.ascontiguousarray(output)
        if not np.isfinite(output).all():
            output = output.copy()
            output[~np.isfinite(output)] = 0.0
        return output

    @classmethod
    def _as_membership_f32_2d(
        cls, array: np.ndarray, name: str
    ) -> np.ndarray:
        # The solver clips memberships once before constructing its pyramids.
        # Do not rescan or recopy every full-resolution level here.
        return cls._as_f32_2d(array, name)

    @staticmethod
    def _validate_flow_component(
        flow: np.ndarray, name: str, shape: tuple[int, int]
    ) -> np.ndarray:
        if (
            not isinstance(flow, np.ndarray)
            or flow.dtype != np.float32
            or flow.shape != shape
            or not flow.flags.c_contiguous
            or not flow.flags.writeable
        ):
            raise ValueError(
                f"{name} must be a writable C-contiguous float32 array "
                f"with shape {shape}"
            )
        return flow

    def refine_masked_inplace(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        source_membership: np.ndarray,
        target_membership: np.ndarray,
        flow: np.ndarray,
        *,
        active_threshold: float,
        target_invalid_weight: float,
    ) -> np.ndarray:
        first = self._as_f32_2d(image0, "image0")
        second = self._as_f32_2d(image1, "image1")
        source = self._as_membership_f32_2d(
            source_membership, "source_membership"
        )
        target = self._as_membership_f32_2d(
            target_membership, "target_membership"
        )

        if not (
            first.shape == second.shape == source.shape == target.shape
        ):
            raise ValueError(
                "Images and memberships must have identical HxW shapes"
            )

        height, width = first.shape
        if (
            not isinstance(flow, np.ndarray)
            or flow.dtype != np.float32
            or flow.shape != (height, width, 2)
            or not flow.flags.c_contiguous
            or not flow.flags.writeable
        ):
            raise ValueError(
                "flow must be a writable C-contiguous float32 array with "
                f"shape {(height, width, 2)}"
            )

        if not (0.0 <= active_threshold <= 1.0):
            raise ValueError("active_threshold must be in [0, 1]")
        if not (0.0 <= target_invalid_weight <= 1.0):
            raise ValueError("target_invalid_weight must be in [0, 1]")

        float_ptr = ctypes.POINTER(ctypes.c_float)
        status = self._lib.amv_vr_masked_refine_f32(
            self._handle,
            first.ctypes.data_as(float_ptr),
            second.ctypes.data_as(float_ptr),
            source.ctypes.data_as(float_ptr),
            target.ctypes.data_as(float_ptr),
            ctypes.c_int(height),
            ctypes.c_int(width),
            ctypes.c_float(active_threshold),
            ctypes.c_float(target_invalid_weight),
            flow.ctypes.data_as(float_ptr),
        )
        if status != 0:
            raise RuntimeError(self.last_error())
        if not np.isfinite(flow).all():
            raise FloatingPointError(
                "Masked OpenCV VR returned non-finite flow"
            )
        return flow

    def refine_masked_uv_inplace(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        source_membership: np.ndarray,
        target_membership: np.ndarray,
        flow_u: np.ndarray,
        flow_v: np.ndarray,
        *,
        active_threshold: float,
        target_invalid_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.supports_uv_interface:
            raise RuntimeError(
                "Native library does not expose amv_vr_masked_refine_uv_f32; "
                "rebuild the optimized native module first"
            )

        first = self._as_f32_2d(image0, "image0")
        second = self._as_f32_2d(image1, "image1")
        source = self._as_membership_f32_2d(
            source_membership, "source_membership"
        )
        target = self._as_membership_f32_2d(
            target_membership, "target_membership"
        )
        if not (first.shape == second.shape == source.shape == target.shape):
            raise ValueError(
                "Images and memberships must have identical HxW shapes"
            )
        if not (0.0 <= active_threshold <= 1.0):
            raise ValueError("active_threshold must be in [0, 1]")
        if not (0.0 <= target_invalid_weight <= 1.0):
            raise ValueError("target_invalid_weight must be in [0, 1]")

        height, width = first.shape
        shape = (height, width)
        self._validate_flow_component(flow_u, "flow_u", shape)
        self._validate_flow_component(flow_v, "flow_v", shape)

        float_ptr = ctypes.POINTER(ctypes.c_float)
        status = self._lib.amv_vr_masked_refine_uv_f32(
            self._handle,
            first.ctypes.data_as(float_ptr),
            second.ctypes.data_as(float_ptr),
            source.ctypes.data_as(float_ptr),
            target.ctypes.data_as(float_ptr),
            ctypes.c_int(height),
            ctypes.c_int(width),
            ctypes.c_float(active_threshold),
            ctypes.c_float(target_invalid_weight),
            flow_u.ctypes.data_as(float_ptr),
            flow_v.ctypes.data_as(float_ptr),
        )
        if status != 0:
            raise RuntimeError(self.last_error())
        if not (np.isfinite(flow_u).all() and np.isfinite(flow_v).all()):
            raise FloatingPointError(
                "Masked OpenCV VR returned non-finite u/v flow"
            )
        return flow_u, flow_v

    def last_profile(self) -> dict[str, float | int | bool]:
        if not self.supports_native_profile:
            return {}
        raw = _NativeMaskedProfile()
        status = self._lib.amv_vr_masked_get_last_profile(
            self._handle, ctypes.byref(raw)
        )
        if status != 0:
            raise RuntimeError(self.last_error())

        output: dict[str, float | int | bool] = {}
        for field in _PROFILE_TIME_FIELDS:
            output[field.removesuffix("_ms") + "_seconds"] = (
                float(getattr(raw, field)) / 1000.0
            )
        output["fixed_point_iterations"] = int(
            raw.fixed_point_iterations
        )
        output["sor_iterations_per_fixed_point"] = int(
            raw.sor_iterations_per_fixed_point
        )
        output["used_uv_interface"] = bool(raw.used_uv_interface)
        return output

    def collect_garbage(self) -> None:
        self._lib.amv_vr_masked_collect_garbage(self._handle)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.amv_vr_masked_destroy(handle)
            self._handle = None
        for directory in getattr(self, "_dll_directories", ()):
            directory.close()
        self._dll_directories = []

    def __enter__(self) -> "NativePressureConditionedVR":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
