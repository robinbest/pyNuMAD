import importlib


def _optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("pynumad"):
            raise
        return None


from pynumad.objects.blade import Blade
from pynumad.objects.airfoil import Airfoil
from pynumad.objects.component import Component
from pynumad.objects.material import Material
from pynumad.objects.station import Station
from pynumad.io.mesh_to_yaml import mesh_to_yaml

mesh_gen = _optional_import("pynumad.mesh_gen")
utils = importlib.import_module("pynumad.utils")
analysis = importlib.import_module("pynumad.analysis")
graphics = _optional_import("pynumad.graphics")

from pynumad.paths import SOFTWARE_PATHS, DATA_PATH, set_path


__version__ = "1.0.0"

__copyright__ = """Copyright 2023 National Technology & Engineering 
Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 
with NTESS, the U.S. Government retains certain rights in this software."""

__license__ = "Revised BSD License"
