#!/usr/bin/env python3
"""Minimal setup.py kept only for the custom install command.
All declarative metadata lives in pyproject.toml.
"""

from setuptools import setup
from setuptools.command.install import install
import shutil
import os
import subprocess
import sys


class CustomInstallCommand(install):
    def run(self):
        install.run(self)

        # Copy the DLL file to the installation directory
        source_path = os.path.join("CPORLib", "obj", "Debug", "netstandard2.0", "CPORLib.dll")
        target_path_dir = os.path.join(self.install_lib, "up_cpor")
        target_path = os.path.join(target_path_dir, "CPORLib.dll")
        shutil.copy(source_path, target_path)

        # Microsoft.Z3.dll — platform-independent managed wrapper
        z3_managed_src = os.path.join(
            os.path.expanduser("~"), ".nuget", "packages",
            "microsoft.z3", "4.12.2", "lib", "netstandard2.0", "Microsoft.Z3.dll",
        )
        if os.path.isfile(z3_managed_src):
            shutil.copy(z3_managed_src, target_path_dir)


setup(cmdclass={"install": CustomInstallCommand})