import pynumad
import numpy as np

from pynumad.analysis.freecad import write_freecad_cross_sections


blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")


def get_cs_params(blade):
    total_stations = np.asarray(blade.ispan).size
    adhesive_thickness = np.full((total_stations,), 0.001)

    return {
        "adhesive_mat_name": "Adhesive",
        "web_fore_adhesive_thickness": adhesive_thickness,
        "web_aft_adhesive_thickness": adhesive_thickness,
    }


script_path = write_freecad_cross_sections(
    blade,
    "myBlade_Modified",
    station_list=[0, 10, 20],
    directory=".",
    move_le_to_origin=True,
    make_faces=True,
    detailed=True,
    cs_params=get_cs_params(blade),
    export_step=False,
)

print(f"Wrote FreeCAD script: {script_path}")
