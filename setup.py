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
        target_path = os.path.join(self.install_lib, "up_cpor", "CPORLib.dll")
        shutil.copy(source_path, target_path)


setup(cmdclass={"install": CustomInstallCommand})