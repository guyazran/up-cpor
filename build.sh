#!/bin/bash

# requires `conda install -c condo-forge botnet`
dotnet build CPORLib/CPORLibSolution.sln

pip uninstall -y up-cpor
pip install .
