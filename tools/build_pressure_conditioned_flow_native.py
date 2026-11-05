"""Build the LayerFlow masked variational-refinement module.

Compile src/tracking/pressure_conditioned_flow/csrc/amv_vr_masked against
OpenCV 4.13.0. Supply --opencv-source or use --fetch-opencv to obtain the
source under _third_party/. Build products remain under _third_party/."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


OPENCV_VERSION = (4, 13, 0)
OPENCV_TAG = "4.13.0"
OPENCV_COMMIT = "fe38fc608f6acb8b68953438a62305d8318f4fcd"
OPENCV_REPOSITORY = "https://github.com/opencv/opencv.git"

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
THIRD_PARTY = REPO_ROOT / "_third_party"
EXTRA_MODULES_PATH = (
    REPO_ROOT
    / "src"
    / "tracking"
    / "pressure_conditioned_flow"
    / "csrc"
)
MODULE_SOURCE = EXTRA_MODULES_PATH / "amv_vr_masked"
DEFAULT_SOURCE_DIR = THIRD_PARTY / "opencv_source_4.13.0"
DEFAULT_BUILD_DIR = THIRD_PARTY / "opencv_build_pressure_conditioned_flow"
DEFAULT_INSTALL_DIR = THIRD_PARTY / "opencv_install_pressure_conditioned_flow"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("\n+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def is_opencv_source(path: Path) -> bool:
    return (
        (path / "CMakeLists.txt").is_file()
        and (
            path
            / "modules"
            / "core"
            / "include"
            / "opencv2"
            / "core"
            / "version.hpp"
        ).is_file()
        and (
            path
            / "modules"
            / "video"
            / "src"
            / "variational_refinement.cpp"
        ).is_file()
    )


def read_version(source: Path) -> tuple[int, int, int]:
    version_file = (
        source
        / "modules"
        / "core"
        / "include"
        / "opencv2"
        / "core"
        / "version.hpp"
    )
    values: dict[str, int] = {}
    for line in version_file.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == "#define":
            if fields[1] in {
                "CV_VERSION_MAJOR",
                "CV_VERSION_MINOR",
                "CV_VERSION_REVISION",
            }:
                values[fields[1]] = int(fields[2])
    return (
        values["CV_VERSION_MAJOR"],
        values["CV_VERSION_MINOR"],
        values["CV_VERSION_REVISION"],
    )


def git_head(source: Path) -> str | None:
    if not (source / ".git").exists() or shutil.which("git") is None:
        return None
    completed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def fetch_clean_opencv(source: Path, *, replace: bool) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required by --fetch-opencv")

    if source.exists():
        if not replace:
            if is_opencv_source(source):
                print("Using existing OpenCV source:", source)
                return
            raise FileExistsError(
                f"Source path exists but is not OpenCV: {source}. "
                "Use --reset-opencv-source to replace it."
            )
        shutil.rmtree(source)

    source.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            OPENCV_TAG,
            OPENCV_REPOSITORY,
            str(source),
        ]
    )


def resolve_source(
    explicit: Path | None,
    *,
    fetch: bool,
    reset: bool,
) -> Path:
    source = (
        explicit.expanduser().resolve()
        if explicit is not None
        else DEFAULT_SOURCE_DIR.resolve()
    )

    if fetch or reset:
        fetch_clean_opencv(source, replace=reset)

    if not is_opencv_source(source):
        raise FileNotFoundError(
            "A clean OpenCV 4.13.0 source tree is required.\n"
            f"Expected: {source}\n"
            "Either run with --fetch-opencv or provide --opencv-source."
        )
    return source


def validate_release_sources(source: Path) -> None:
    if not MODULE_SOURCE.is_dir():
        raise FileNotFoundError(
            f"Release native module is missing: {MODULE_SOURCE}"
        )

    version = read_version(source)
    print("OpenCV version:", ".".join(map(str, version)))
    if version != OPENCV_VERSION:
        raise RuntimeError(
            "The validated implementation requires OpenCV 4.13.0; "
            f"found {version[0]}.{version[1]}.{version[2]}"
        )

    head = git_head(source)
    if head is not None:
        print("OpenCV commit:", head)
        if head != OPENCV_COMMIT:
            raise RuntimeError(
                "OpenCV 4.13.0 source commit does not match the validated "
                f"commit {OPENCV_COMMIT}; found {head}."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencv-source", type=Path, default=None)
    parser.add_argument(
        "--fetch-opencv",
        action="store_true",
        help="Fetch the validated OpenCV 4.13.0 tag into _third_party.",
    )
    parser.add_argument(
        "--reset-opencv-source",
        action="store_true",
        help="Delete and freshly fetch the default/selected OpenCV source.",
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--generator", default=None, help="CMake generator, for example Ninja")
    parser.add_argument("--c-compiler", default=None, help="Optional C compiler executable")
    parser.add_argument("--cxx-compiler", default=None, help="Optional C++ compiler executable")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--install-dir", type=Path, default=DEFAULT_INSTALL_DIR)
    args = parser.parse_args()

    source = resolve_source(
        args.opencv_source,
        fetch=args.fetch_opencv,
        reset=args.reset_opencv_source,
    )
    build_dir = args.build_dir.expanduser().resolve()
    install_dir = args.install_dir.expanduser().resolve()

    print("Repository root:", REPO_ROOT)
    print("Custom module source:", MODULE_SOURCE)
    print("OpenCV source:", source)
    print("Build directory:", build_dir)
    print("Install directory:", install_dir)

    validate_release_sources(source)

    if args.clean:
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(install_dir, ignore_errors=True)

    build_dir.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)

    generator: list[str] = []
    if args.generator:
        generator = ["-G", args.generator]
    elif shutil.which("ninja"):
        generator = ["-G", "Ninja"]

    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build_dir),
        *generator,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DBUILD_LIST=core,imgproc,video,amv_vr_masked",
        f"-DOPENCV_EXTRA_MODULES_PATH={EXTRA_MODULES_PATH}",
        "-DBUILD_SHARED_LIBS=ON",
        f"-DOPENCV_GENERATE_PKGCONFIG={'OFF' if os.name == 'nt' else 'ON'}",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_DOCS=OFF",
        "-DBUILD_JAVA=OFF",
        "-DBUILD_opencv_python2=OFF",
        "-DBUILD_opencv_python3=OFF",
        "-DBUILD_opencv_python_bindings_generator=OFF",
        "-DBUILD_opencv_world=OFF",
        "-DWITH_IPP=OFF",
        "-DWITH_OPENCL=OFF",
        "-DWITH_TBB=OFF",
        "-DWITH_OPENMP=OFF",
        f"-DWITH_PTHREADS_PF={'OFF' if os.name == 'nt' else 'ON'}",
        "-DWITH_FFMPEG=OFF",
        "-DWITH_GSTREAMER=OFF",
        "-DWITH_QT=OFF",
        "-DWITH_GTK=OFF",
        "-DWITH_V4L=OFF",
        "-DWITH_1394=OFF",
        "-DWITH_JASPER=OFF",
        "-DWITH_OPENEXR=OFF",
        "-DWITH_EIGEN=OFF",
        "-DWITH_LAPACK=OFF",
        "-DBUILD_opencv_highgui=OFF",
        "-DBUILD_opencv_videoio=OFF",
        "-DBUILD_opencv_imgcodecs=OFF",
    ]
    if args.c_compiler:
        configure.append(f"-DCMAKE_C_COMPILER={args.c_compiler}")
    if args.cxx_compiler:
        configure.append(f"-DCMAKE_CXX_COMPILER={args.cxx_compiler}")
    run(configure)

    if args.configure_only:
        print("Configuration completed; build was not started.")
        return

    jobs = str(max(1, int(args.jobs)))
    run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "opencv_amv_vr_masked",
            "--parallel",
            jobs,
        ]
    )
    run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "install",
            "--parallel",
            jobs,
        ]
    )

    patterns = ("*opencv_amv_vr_masked*.dll",) if os.name == "nt" else (
        "libopencv_amv_vr_masked.so*", "libopencv_amv_vr_masked*.dylib",
    )
    libraries = sorted({p for pattern in patterns for p in install_dir.rglob(pattern)})
    if not libraries:
        raise FileNotFoundError(
            "Build finished but the masked OpenCV shared library was not installed"
        )

    print("\nPressure-conditioned release native build complete.")
    for path in libraries:
        print("  ", path)
    print("Custom implementation was compiled directly from:")
    print("  ", MODULE_SOURCE)


if __name__ == "__main__":
    main()
