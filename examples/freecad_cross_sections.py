import pynumad
import numpy as np

from pynumad.analysis.freecad import write_freecad_cross_sections

"""
How to run:
python examples/freecad_cross_sections.py 
~/projs/FreeCAD/squashfs-root/usr/bin/freecadcmd  TurbineBlade_freecad_cross_sections.py
"""

yaml_file = "examples/example_data/myBlade_Modified.yaml"
st_list=[0, 3, 5, 7, 10, 15, 20, 25, 28, 29]
#yaml_file = "examples/example_data/IEA-22-280-RWT.yaml"
#st_list=[0, 3, 5, 7, 10, 15, 20, 25, 28, 29, 39, 49, 59]
blade = pynumad.Blade(yaml_file)


def get_cs_params(blade):
    total_stations = np.asarray(blade.ispan).size
    adhesive_thickness = np.full((total_stations,), 0.001)

    return {
        "adhesive_mat_name": "Adhesive",
        "web_fore_adhesive_thickness": adhesive_thickness,
        "web_aft_adhesive_thickness": adhesive_thickness,

        # Shell spar-boundary adhesive
        "shell_component_adhesive_width": 0.001,
        "shell_component_adhesive_mat_name": "Adhesive",

        # Smooth/resample shell curves before layer offsets.
        "shell_resample_enabled": True,
        "shell_resample_scope": "side",
        "shell_resample_spacing": 0.005,
        "shell_resample_min_points": 12,
        "shell_resample_max_points": 400,
    }


script_path = write_freecad_cross_sections(
    blade,
    "TurbineBlade",
    station_list=st_list,
    directory=".",
    move_le_to_origin=True,
    make_faces=True,
    detailed=True,
    cs_params=get_cs_params(blade),
    export_step=False,
)

print(f"Wrote FreeCAD script: {script_path}")
