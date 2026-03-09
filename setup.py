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
        
        target_path_dir = os.path.join(self.install_lib, "up_cpor")

        # Copy the main DLL file to the installation directory
        source_path = os.path.join("CPORLib", "obj", "Debug", "netstandard2.0", "CPORLib.dll")
        target_path = os.path.join(target_path_dir, "CPORLib.dll")
        shutil.copy(source_path, target_path)

        # copy the Microsoft.Solver.Foundation.dll to the installation directory
        solver_path = os.path.join("CPORLib", "RequiredDLLs", "Microsoft.Solver.Foundation.dll")
        target_path = os.path.join(target_path_dir, "Microsoft.Solver.Foundation.dll")
        shutil.copy(solver_path, target_path)


setup(cmdclass={"install": CustomInstallCommand})