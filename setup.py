import os
import pathlib
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # All statically linked libraries will be placed here
        lib_output_dir = os.path.abspath(
            os.path.join(self.build_temp, 'lib'))
        if not os.path.exists(lib_output_dir):
            os.makedirs(lib_output_dir)

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        import torch

        pytorch_dir = os.path.dirname(torch.__file__)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' +
            str(extdir.parent.absolute()),
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_BUILD_TYPE=' + config,
            '-DCMAKE_PREFIX_PATH=' + pytorch_dir,
            '-DCMAKE_EXPORT_COMPILE_COMMANDS=1'
            # '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(config.upper(),
            #                                                lib_output_dir)
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))


setup(
    name='ragdoll',
    version='0.1',
    # packages=['ragdoll'],
    packages=find_packages(),
    ext_modules=[CMakeExtension('ragdoll/ragdoll_core')],
    cmdclass={
        'build_ext': build_ext,
    }
)
