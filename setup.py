from __future__ import annotations

import platform
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self) -> None:
        import mujoco
        import numpy
        import pybind11

        mujoco_dir = Path(mujoco.__file__).resolve().parent
        lib_candidates = sorted(mujoco_dir.glob("libmujoco*"))
        if not lib_candidates:
            raise RuntimeError(f"Could not find libmujoco in {mujoco_dir}")
        libmujoco = lib_candidates[0]

        for ext in self.extensions:
            ext.include_dirs.extend(
                [
                    pybind11.get_include(),
                    numpy.get_include(),
                    str(mujoco_dir / "include"),
                    str(Path(__file__).resolve().parent / "src" / "mujoco_uni" / "native"),
                ]
            )
            ext.extra_objects.append(str(libmujoco))
            if platform.system() == "Darwin":
                ext.extra_compile_args.extend(["-std=c++17", "-stdlib=libc++"])
                ext.extra_link_args.extend(
                    ["-stdlib=libc++", "-Wl,-rpath,@loader_path/../../mujoco"]
                )
            elif platform.system() == "Linux":
                ext.extra_compile_args.extend(["-std=c++17"])
                ext.extra_link_args.extend(["-Wl,-rpath,$ORIGIN/../../mujoco"])
            else:
                ext.extra_compile_args.extend(["/std:c++17"])
        super().build_extensions()


setup(
    ext_modules=[
        Extension(
            "mujoco_uni.compiled._batch_env",
            sources=[
                "src/mujoco_uni/native/batch_env.cc",
                "src/mujoco_uni/native/threadpool.cc",
            ],
            language="c++",
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
