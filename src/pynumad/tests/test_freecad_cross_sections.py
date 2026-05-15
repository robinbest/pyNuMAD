import json

import pynumad
import numpy as np

from pynumad.analysis.freecad import (
    blade_station_count,
    face_material_metadata,
    get_cross_section,
    get_detailed_cross_section,
    get_yaml_station_count,
    global_laminate_definitions,
    laminate_definitions,
    load_blade_for_freecad,
    material_definitions,
    make_freecad_cross_section_parts,
    make_freecad_section_part,
    record_turbine_message,
    station_frame_definition,
    write_freecad_cross_sections,
    yaml_station_count,
)
from pynumad.analysis.freecad.make_cross_sections import (
    _regions_in_shape_face_order,
    _regions_in_shape_face_order_with_messages,
    _shell_laminate_vertex_contact_messages,
)


class _FakeCenter:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _FakeFace:
    def __init__(self, area, center):
        self.Area = area
        self.CenterOfMass = _FakeCenter(*center)


class _FakeShape:
    def __init__(self, faces):
        self.Faces = faces


class _FakeFaceWithoutSignature:
    pass


class _FakeFreeCADObject:
    def __init__(self, name):
        self.Name = name
        self.Label = name
        self.PropertiesList = []

    def addProperty(self, _type_name, name, _group="", _description=""):
        if name not in self.PropertiesList:
            self.PropertiesList.append(name)


class _FakeFreeCADDoc:
    def __init__(self):
        self.objects = {}

    def getObject(self, name):
        return self.objects.get(name)

    def addObject(self, _type_name, name):
        obj = _FakeFreeCADObject(name)
        self.objects[name] = obj
        return obj


def test_get_cross_section_splits_hp_and_lp_surfaces():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    section = get_cross_section(blade, 0)

    assert section.station == 0
    assert section.hp_points.shape[1] == 3
    assert section.lp_points.shape[1] == 3
    assert len(section.hp_points) == blade.geometry.LEindex
    assert len(section.lp_points) == blade.geometry.LEindex


def test_write_freecad_cross_sections_script(tmp_path):
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    script_path = write_freecad_cross_sections(
        blade,
        "blade",
        station_list=[0],
        directory=tmp_path,
        move_le_to_origin=True,
    )

    contents = script_path.read_text(encoding="utf-8")
    assert script_path.name == "blade_freecad_cross_sections.py"
    assert "import FreeCAD as App" in contents
    assert '"station": 0' in contents
    assert "Station{:03d}_wire" in contents
    assert '"station_frame"' in contents
    assert "StationFrame" in contents
    assert "WarningMessages" in contents
    assert "ErrorMessages" in contents


def test_get_detailed_cross_section_has_shell_and_web_regions():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")
    cs_params = {
        "adhesive_mat_name": "Adhesive",
        "web_fore_adhesive_thickness": np.full((len(blade.ispan),), 0.001),
        "web_aft_adhesive_thickness": np.full((len(blade.ispan),), 0.001),
    }

    section = get_detailed_cross_section(blade, 10, cs_params=cs_params, move_le_to_origin=True)

    assert section.regions
    assert any("HP" in region.name for region in section.regions)
    assert any("web" in region.name for region in section.regions)
    assert all(region.points is not None or region.outer_points is not None or region.edge_points is not None for region in section.regions)
    assert section.station_frame["station"] == 10
    assert "lcs" in section.station_frame


def test_station_frame_definition_contains_planar_lcs_and_reference_lcs():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    frame = station_frame_definition(blade, 10)
    section_basis = np.column_stack(
        (
            frame["lcs"]["x_axis"],
            frame["lcs"]["y_axis"],
            frame["lcs"]["z_axis"],
        )
    )
    reference_basis = np.column_stack(
        (
            frame["reference_lcs"]["x_axis"],
            frame["reference_lcs"]["y_axis"],
            frame["reference_lcs"]["z_axis"],
        )
    )

    assert frame["station"] == 10
    assert "reference_axis" not in frame
    np.testing.assert_allclose(frame["origin"], [0.0, blade.geometry.iprebend[10], blade.ispan[10]])
    assert frame["origin_units"] == "m"
    np.testing.assert_allclose(frame["section_translation"], [0.0, 0.0, 0.0])
    assert set(frame["rotations"]) == {
        "prebend_angle_deg",
        "sweep_angle_deg",
        "twist_deg",
        "prebend_slope",
        "sweep_slope",
    }
    assert abs(frame["rotations"]["twist_deg"] - blade.geometry.idegreestwist[10]) < 1e-12
    np.testing.assert_allclose(section_basis.T @ section_basis, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(reference_basis.T @ reference_basis, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(section_basis[2, :2], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(section_basis[:, 2], [0.0, 0.0, 1.0], atol=1e-12)
    assert abs(reference_basis[2, 2] - 1.0) > 1e-6


def test_cs_params_geometry_scaling_scales_station_frame_origin():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    meter_section = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    mm_section = get_detailed_cross_section(
        blade,
        10,
        move_le_to_origin=True,
        cs_params={"geometry_scaling": 1000.0},
    )

    np.testing.assert_allclose(
        mm_section.station_frame["origin"],
        1000.0 * np.array(meter_section.station_frame["origin"]),
    )
    assert mm_section.station_frame["span"] == 1000.0 * meter_section.station_frame["span"]
    assert mm_section.station_frame["origin_units"] == "mm"


def test_arbitrary_geometry_scaling_uses_scaled_station_frame_units():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(
        blade,
        10,
        move_le_to_origin=True,
        cs_params={"geometry_scaling": 25.0},
    )

    assert section.station_frame["geometry_scaling"] == 25.0
    assert section.station_frame["origin_units"] == "scaled"


def test_freecad_direct_api_is_importable_without_freecad():
    assert callable(make_freecad_section_part)
    assert callable(make_freecad_cross_section_parts)


def test_face_material_metadata_splits_region_name_fields():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")
    cs_params = _web_adhesive_cs_params(blade)

    section = get_detailed_cross_section(blade, 10, cs_params=cs_params, move_le_to_origin=True)
    metadata = face_material_metadata(section.regions)

    assert len(metadata) == len(section.regions)
    shell_item = next(
        item
        for item in metadata
        if item["region_name"].startswith("Station010_HP_03_") and item["assignment_type"] == "laminate"
    )
    assert shell_item["station"] == 10
    assert shell_item["side"] == "HP"
    assert shell_item["layer"] >= 0
    assert shell_item["stack_index"] == "03"
    assert shell_item["web_index"] is None
    assert shell_item["material_name"]
    assert shell_item["assignment_type"] == "laminate"
    assert shell_item["assignment_name"].startswith("Laminate")
    assert "plies" not in shell_item

    adhesive_item = next(item for item in metadata if item["region_name"] == "Station010_web0_hp_adhesive")
    assert adhesive_item["station"] == 10
    assert adhesive_item["web_index"] == 0
    assert adhesive_item["side"] == "HP"
    assert adhesive_item["stack_index"] is None
    assert adhesive_item["stack_name"] is None
    assert adhesive_item["component_name"] is None
    assert adhesive_item["assignment_type"] == "material"
    assert adhesive_item["assignment_name"] == adhesive_item["material_name"]


def test_face_material_metadata_uses_fixed_snake_case_schema():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(blade, 7, move_le_to_origin=True)
    metadata = face_material_metadata(section.regions)
    expected_keys = {
        "station",
        "layer",
        "side",
        "stack_index",
        "stack_name",
        "component_name",
        "web_index",
        "face_index",
        "region_name",
        "material_name",
        "assignment_type",
        "assignment_index",
        "assignment_name",
    }

    assert metadata
    assert all(set(item) == expected_keys for item in metadata)
    assert all(key == key.lower() for key in expected_keys)
    assert all("_" in key or key in {"side", "layer", "station"} for key in expected_keys)
    assert all("RegionName" not in item and "Feature" not in item for item in metadata)


def test_face_material_metadata_can_reference_global_turbine_tables():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(blade, 7, move_le_to_origin=True)
    laminate_table = global_laminate_definitions(blade)
    material_table = material_definitions(blade)
    metadata = face_material_metadata(
        section.regions,
        laminate_table=laminate_table,
        material_table=material_table,
    )
    laminate_names = {item["laminate_name"] for item in laminate_table}
    material_indices = {item["material_name"]: item["material_index"] for item in material_table}

    assert laminate_table
    assert all(item["assignment_index"] is not None for item in metadata)
    assert all(
        item["assignment_name"] in laminate_names
        for item in metadata
        if item["assignment_type"] == "laminate"
    )
    assert all(
        item["assignment_index"] == material_indices[item["material_name"]]
        for item in metadata
        if item["assignment_type"] == "material"
    )


def test_face_material_metadata_expands_repeated_plygroups_for_homogen():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(blade, 5, move_le_to_origin=True)
    metadata = face_material_metadata(section.regions)
    laminates = laminate_definitions(section.regions)
    laminates_by_name = {laminate["laminate_name"]: laminate for laminate in laminates}
    repeated = next(
        item
        for item in metadata
        if item["assignment_type"] == "laminate" and len(laminates_by_name[item["assignment_name"]]["plies"]) > 1
    )
    repeated_laminate = laminates_by_name[repeated["assignment_name"]]

    assert repeated["assignment_type"] == "laminate"
    assert repeated["assignment_name"] == repeated_laminate["laminate_name"]
    assert len(repeated_laminate["plies"]) > 1
    assert len({ply["material"] for ply in repeated_laminate["plies"]}) == 1
    assert len({ply["angle"] for ply in repeated_laminate["plies"]}) == 1
    assert all(ply["thickness"] > 0 for ply in repeated_laminate["plies"])
    assert all(set(ply) == {"material", "angle", "thickness"} for ply in repeated_laminate["plies"])


def test_laminate_definitions_deduplicate_shared_ply_stacks():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    metadata = face_material_metadata(section.regions)
    laminates = laminate_definitions(section.regions)
    laminate_face_count = sum(1 for item in metadata if item["assignment_type"] == "laminate")
    laminate_names = {item["laminate_name"] for item in laminates}

    assert laminates
    assert len(laminates) < laminate_face_count
    assert [item["laminate_index"] for item in laminates] == list(range(len(laminates)))
    assert all(item["assignment_name"] in laminate_names for item in metadata if item["assignment_type"] == "laminate")


def test_foam_core_faces_are_material_assignments_not_laminates():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(blade, 7, move_le_to_origin=True)
    metadata = face_material_metadata(section.regions)
    laminates = laminate_definitions(section.regions)
    foam_faces = [item for item in metadata if item["material_name"] == "medium_density_foam"]

    assert foam_faces
    assert all(item["assignment_type"] == "material" for item in foam_faces)
    assert all(item["assignment_name"] == "medium_density_foam" for item in foam_faces)
    assert all(
        all(ply["material"] != "medium_density_foam" for ply in laminate["plies"])
        for laminate in laminates
    )


def test_station_10_web0_core_is_face43_material_assignment():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    metadata = face_material_metadata(
        section.regions,
        laminate_table=global_laminate_definitions(blade),
        material_table=material_definitions(blade),
    )
    web0_items = [item for item in metadata if item["region_name"].startswith("Station010_web0_layer")]

    assert [(item["face_index"], item["region_name"], item["material_name"]) for item in web0_items] == [
        (41, "Station010_web0_layer00", "glass_biax"),
        (42, "Station010_web0_layer01", "medium_density_foam"),
        (43, "Station010_web0_layer02", "glass_biax"),
    ]
    assert web0_items[1]["assignment_type"] == "material"
    assert web0_items[1]["assignment_index"] == 5
    assert web0_items[1]["assignment_name"] == "medium_density_foam"


def test_face_material_regions_can_follow_reordered_freecad_faces():
    regions = ["skin_a", "core", "skin_b"]
    source_faces = [
        _FakeFace(1.0, (0.0, 0.0)),
        _FakeFace(3.0, (1.0, 0.0)),
        _FakeFace(1.2, (2.0, 0.0)),
    ]
    stitched_shape = _FakeShape([source_faces[2], source_faces[0], source_faces[1]])

    ordered = _regions_in_shape_face_order(regions, source_faces, stitched_shape)

    assert ordered == ["skin_b", "skin_a", "core"]


def test_face_material_reorder_returns_no_messages_when_verified():
    regions = ["skin_a", "core", "skin_b"]
    source_faces = [
        _FakeFace(1.0, (0.0, 0.0)),
        _FakeFace(3.0, (1.0, 0.0)),
        _FakeFace(1.2, (2.0, 0.0)),
    ]
    stitched_shape = _FakeShape([source_faces[2], source_faces[0], source_faces[1]])

    ordered, messages = _regions_in_shape_face_order_with_messages(
        regions,
        source_faces,
        stitched_shape,
        station=10,
    )

    assert ordered == ["skin_b", "skin_a", "core"]
    assert messages == []


def test_face_material_reorder_returns_warning_when_signatures_are_unavailable():
    regions = ["skin_a", "core", "skin_b"]
    source_faces = [
        _FakeFace(1.0, (0.0, 0.0)),
        _FakeFaceWithoutSignature(),
        _FakeFace(1.2, (2.0, 0.0)),
    ]
    stitched_shape = _FakeShape(source_faces)

    ordered, messages = _regions_in_shape_face_order_with_messages(
        regions,
        source_faces,
        stitched_shape,
        station=10,
    )

    assert ordered == regions
    assert len(messages) == 1
    assert messages[0]["severity"] == "warning"
    assert messages[0]["code"] == "face_signature_unavailable"
    assert messages[0]["station"] == 10
    assert messages[0]["source"] == "freecad_cross_sections.face_material_map"
    assert "region-generation order" in messages[0]["message"]


def test_shell_laminate_vertex_contact_warning_identifies_spar_cap_skin_step():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True)

    messages = _shell_laminate_vertex_contact_messages(section.regions)
    target = next(
        message
        for message in messages
        if message["details"]["first_region_name"] == "Station010_HP_03_10_HP_SPAR_layer02"
        and message["details"]["second_region_name"] == "Station010_HP_02_10_HP_TE_PANEL_layer03"
    )

    assert target["severity"] == "warning"
    assert target["code"] == "shell_laminate_vertex_contact"
    assert target["station"] == 10
    assert target["details"]["first_face_index"] == 22
    assert target["details"]["second_face_index"] == 31
    assert np.isclose(target["details"]["first_thickness"], 0.095)
    assert np.isclose(target["details"]["second_thickness"], 0.003)


def test_shell_component_adhesive_inserts_four_spar_boundary_faces():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    cs_params = {
        "shell_component_adhesive_width": 0.001,
        "shell_component_adhesive_mat_name": "Adhesive",
    }

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True, cs_params=cs_params)
    metadata = face_material_metadata(
        section.regions,
        laminate_table=global_laminate_definitions(blade),
        material_table=material_definitions(blade),
    )
    shell_adhesives = [
        item
        for item in metadata
        if "_to_" in item["region_name"] and item["material_name"] == "Adhesive"
    ]

    assert len(shell_adhesives) == 4
    assert all("_adhesive_layer00" not in item["region_name"] for item in metadata)
    assert all("_adhesive_layer01" not in item["region_name"] for item in metadata)
    assert {item["layer"] for item in shell_adhesives} == {2}
    assert all(item["material_name"] == "Adhesive" for item in shell_adhesives)
    assert all(item["assignment_type"] == "material" for item in shell_adhesives)
    assert all("SPAR" in item["region_name"] for item in shell_adhesives)
    assert _shell_laminate_vertex_contact_messages(section.regions) == []


def test_shell_component_adhesive_preserves_common_outer_layer_areas():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    cs_params = {
        "shell_component_adhesive_width": 0.001,
        "shell_component_adhesive_mat_name": "Adhesive",
    }

    baseline = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    with_adhesive = get_detailed_cross_section(blade, 10, move_le_to_origin=True, cs_params=cs_params)
    baseline_areas = {
        region.name: _polygon_area(_region_polygon(region))
        for region in baseline.regions
        if ("_HP_" in region.name or "_LP_" in region.name)
        and (region.name.endswith("_layer00") or region.name.endswith("_layer01"))
    }

    for region in with_adhesive.regions:
        if "_to_" in region.name:
            assert not region.name.endswith("_layer00")
            assert not region.name.endswith("_layer01")
            continue
        if region.name in baseline_areas:
            assert np.isclose(_polygon_area(_region_polygon(region)), baseline_areas[region.name])


def test_shell_component_adhesive_keeps_spar_boundary_colinear():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    cs_params = {
        "shell_component_adhesive_width": 0.001,
        "shell_component_adhesive_mat_name": "Adhesive",
    }

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True, cs_params=cs_params)
    checks = [
        (
            "Station010_HP_03_10_HP_SPAR_layer01",
            "start_connector",
            "Station010_HP_02_10_HP_TE_PANEL_to_03_10_HP_SPAR_adhesive_layer02",
            "end_connector",
        ),
        (
            "Station010_HP_03_10_HP_SPAR_layer01",
            "end_connector",
            "Station010_HP_03_10_HP_SPAR_to_04_10_HP_LE_PANEL_adhesive_layer02",
            "start_connector",
        ),
        (
            "Station010_LP_08_10_LP_SPAR_layer01",
            "start_connector",
            "Station010_LP_07_10_LP_LE_PANEL_to_08_10_LP_SPAR_adhesive_layer02",
            "end_connector",
        ),
        (
            "Station010_LP_08_10_LP_SPAR_layer01",
            "end_connector",
            "Station010_LP_08_10_LP_SPAR_to_09_10_LP_TE_PANEL_adhesive_layer02",
            "start_connector",
        ),
    ]

    for spar_layer_name, spar_connector_name, adhesive_name, adhesive_connector_name in checks:
        spar_layer = next(region for region in section.regions if region.name == spar_layer_name)
        adhesive = next(region for region in section.regions if region.name == adhesive_name)
        spar_connector = getattr(spar_layer, spar_connector_name)
        adhesive_connector = getattr(adhesive, adhesive_connector_name)
        assert _segments_colinear(spar_connector[0], spar_connector[-1], adhesive_connector[0], adhesive_connector[-1])


def test_shell_component_adhesive_splits_previous_layer_shared_edges():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    cs_params = {
        "shell_component_adhesive_width": 0.001,
        "shell_component_adhesive_mat_name": "Adhesive",
    }

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True, cs_params=cs_params)
    layer01 = next(region for region in section.regions if region.name == "Station010_HP_02_10_HP_TE_PANEL_layer01")
    layer02 = next(region for region in section.regions if region.name == "Station010_HP_02_10_HP_TE_PANEL_layer02")
    adhesive = next(
        region
        for region in section.regions
        if region.name == "Station010_HP_02_10_HP_TE_PANEL_to_03_10_HP_SPAR_adhesive_layer02"
    )

    assert layer01.edge_points is not None
    assert _regions_share_edge(layer01, layer02)
    assert _regions_share_edge(layer01, adhesive)


def test_all_shell_component_adhesive_splits_are_shared():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    cs_params = {
        "geometry_scaling": 1000.0,
        "shell_component_adhesive_width": 0.001,
        "shell_component_adhesive_mat_name": "Adhesive",
        "skip_shell_gelcoat_layer": True,
    }

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True, cs_params=cs_params)
    shell_regions = [region for region in section.regions if ("_HP_" in region.name or "_LP_" in region.name)]
    shell_adhesives = [
        region
        for region in shell_regions
        if "_to_" in region.name and region.material_name == "Adhesive"
    ]

    assert len(shell_adhesives) == 4
    split_previous_layers = [
        region
        for region in shell_regions
        if region.edge_points is not None and region.name.endswith("_layer01")
    ]
    assert split_previous_layers
    for region in split_previous_layers:
        for edge in region.edge_points[2:-1]:
            matches = [
                candidate
                for candidate in shell_regions
                if candidate is not region and _region_has_edge(candidate, edge)
            ]
            assert matches, region.name


def test_skip_shell_gelcoat_layer_omits_layer00_but_preserves_inner_geometry():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    baseline = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    skipped = get_detailed_cross_section(
        blade,
        10,
        move_le_to_origin=True,
        cs_params={"skip_shell_gelcoat_layer": True},
    )

    assert not any(
        ("_HP_" in region.name or "_LP_" in region.name) and region.name.endswith("_layer00")
        for region in skipped.regions
    )
    baseline_layer01 = next(region for region in baseline.regions if region.name == "Station010_HP_02_10_HP_TE_PANEL_layer01")
    skipped_layer01 = next(region for region in skipped.regions if region.name == baseline_layer01.name)
    assert np.allclose(skipped_layer01.outer_points, baseline_layer01.outer_points)
    assert np.allclose(skipped_layer01.inner_points, baseline_layer01.inner_points)


def test_skip_shell_gelcoat_layer_keeps_trailing_edge_adhesive():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(
        blade,
        10,
        move_le_to_origin=True,
        cs_params={"skip_shell_gelcoat_layer": True},
    )

    te_adhesive = next(region for region in section.regions if region.name == "Station010_TE_adhesive")
    assert te_adhesive.material_name == "Adhesive"
    assert not any(
        ("_HP_" in region.name or "_LP_" in region.name) and region.name.endswith("_layer00")
        for region in section.regions
    )


def test_scaled_skip_shell_gelcoat_layer_keeps_trailing_edge_adhesive():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(
        blade,
        10,
        move_le_to_origin=True,
        cs_params={"skip_shell_gelcoat_layer": True, "geometry_scaling": 1000.0},
    )

    te_adhesive = next(region for region in section.regions if region.name == "Station010_TE_adhesive")
    assert te_adhesive.material_name == "Adhesive"


def test_shell_stack_boundary_connector_is_normal_with_scaled_gelcoat_skip():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    cs_params = {
        "geometry_scaling": 1000.0,
        "shell_component_adhesive_width": 0.001,
        "shell_component_adhesive_mat_name": "Adhesive",
        "skip_shell_gelcoat_layer": True,
    }

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True, cs_params=cs_params)
    te_panel = next(region for region in section.regions if region.name == "Station010_HP_02_10_HP_TE_PANEL_layer01")
    spar = next(region for region in section.regions if region.name == "Station010_HP_03_10_HP_SPAR_layer01")

    assert np.allclose(te_panel.outer_points[-1], spar.outer_points[0])
    assert np.allclose(te_panel.inner_points[-1], spar.inner_points[0])
    tangent = _unit_2d(
        _unit_2d(te_panel.outer_points[-1, :2] - te_panel.outer_points[-2, :2])
        + _unit_2d(spar.outer_points[1, :2] - spar.outer_points[0, :2])
    )
    connector = _unit_2d(te_panel.inner_points[-1, :2] - te_panel.outer_points[-1, :2])
    assert abs(np.dot(tangent, connector)) < 1e-8


def test_stair_step_squaring_preserves_leading_edge_topology_with_near_normal_connector():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    cs_params = {
        "geometry_scaling": 1000.0,
        "shell_component_adhesive_width": 0.001,
        "shell_component_adhesive_mat_name": "Adhesive",
        "skip_shell_gelcoat_layer": True,
    }

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True, cs_params=cs_params)
    adjacent_region = next(region for region in section.regions if region.name == "Station010_HP_04_10_HP_LE_PANEL_layer02")
    previous_layer = next(region for region in section.regions if region.name == "Station010_HP_05_10_HP_LE_layer02")
    le_region = next(region for region in section.regions if region.name == "Station010_HP_05_10_HP_LE_layer03")
    tangent = _unit_2d(le_region.outer_points[1, :2] - le_region.outer_points[0, :2])
    connector = _unit_2d(le_region.inner_points[0, :2] - le_region.outer_points[0, :2])
    stair_direction = _unit_2d(adjacent_region.end_connector[-1, :2] - le_region.outer_points[0, :2])

    assert np.allclose(le_region.outer_points[0], previous_layer.inner_points[0])
    assert np.allclose(le_region.outer_points, previous_layer.inner_points)
    assert np.dot(le_region.outer_points[1, :2] - le_region.outer_points[0, :2], le_region.outer_points[2, :2] - le_region.outer_points[0, :2]) > 0.0
    assert np.dot(le_region.inner_points[1, :2] - le_region.inner_points[0, :2], le_region.inner_points[2, :2] - le_region.inner_points[0, :2]) > 0.0
    assert abs(np.cross(stair_direction, connector)) < 1e-8
    assert abs(np.dot(tangent, connector)) < np.sin(np.deg2rad(10.0))


def test_skip_shell_gelcoat_layer_keeps_non_gelcoat_layer00():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    blade.stackdb.stacks[1, 10].plygroups[0].materialid = "glass_triax"

    section = get_detailed_cross_section(
        blade,
        10,
        move_le_to_origin=True,
        cs_params={"skip_shell_gelcoat_layer": True},
    )

    assert any(region.name == "Station010_HP_01_10_HP_TE_REINF_layer00" for region in section.regions)


def test_record_turbine_message_creates_metadata_error_channel():
    doc = _FakeFreeCADDoc()

    metadata_obj = record_turbine_message(
        doc,
        "error",
        "blade_yaml_read_failed",
        "YAML web missing arcs",
        source="freecad_cross_sections.blade_load",
        details={"yaml_file": "examples/example_data/V27_fromScan.yaml"},
    )

    assert metadata_obj is doc.getObject("TurbineMetadata")
    assert json.loads(metadata_obj.WarningMessages) == []
    errors = json.loads(metadata_obj.ErrorMessages)
    assert errors[0]["code"] == "blade_yaml_read_failed"
    assert errors[0]["source"] == "freecad_cross_sections.blade_load"
    assert errors[0]["details"]["yaml_file"] == "examples/example_data/V27_fromScan.yaml"
    assert json.loads(metadata_obj.MaterialDefinitions) == []
    assert json.loads(metadata_obj.LaminateDefinitions) == []


def test_load_blade_for_freecad_records_yaml_load_failure():
    doc = _FakeFreeCADDoc()

    blade = load_blade_for_freecad("examples/example_data/V27_fromScan.yaml", doc=doc)

    assert blade is None
    errors = json.loads(doc.getObject("TurbineMetadata").ErrorMessages)
    assert errors[0]["code"] == "blade_yaml_read_failed"
    assert errors[0]["details"]["exception_type"] == "ValueError"
    assert "start_nd_arc and end_nd_arc" in errors[0]["message"]


def test_material_definitions_include_elastic_density_and_thermal_data():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    materials = material_definitions(blade)
    by_name = {item["material_name"]: item for item in materials}

    assert [item["material_index"] for item in materials] == list(range(len(materials)))
    assert by_name["glass_triax"]["material_type"] == "orthotropic"
    assert by_name["glass_triax"]["density"] == 1940.0
    assert by_name["glass_triax"]["elastic"]["e1"] == 28211400000.0
    assert by_name["glass_triax"]["elastic"]["g23"] == 3491240000.0
    assert by_name["Gelcoat"]["material_type"] == "isotropic"
    assert by_name["Gelcoat"]["elastic"]["youngs_modulus"] == 3440000000.0
    assert by_name["Gelcoat"]["thermal"]["expansion_coefficient"] == 0.0
    assert by_name["Gelcoat"]["strength"]["compressive"] == 10000000000.0
    assert "thermal" not in by_name["glass_triax"]


def test_yaml_station_count_is_lightweight_and_matches_imported_blade():
    yaml_path = "examples/example_data/myBlade_Modified.yaml"
    blade = pynumad.Blade(yaml_path)

    assert yaml_station_count(yaml_path) == 30
    assert get_yaml_station_count(yaml_path) == 30
    assert yaml_station_count(yaml_path) == blade_station_count(blade)


def test_detailed_webs_connect_spar_boundaries():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    section = get_detailed_cross_section(blade, 20, move_le_to_origin=True)
    web_layers = [region for region in section.regions if "_web" in region.name and "_layer00" in region.name]

    assert len(web_layers) == 2
    for web_layer in web_layers:
        hp_center, lp_center = _web_layer_centers(web_layer)
        assert abs(hp_center[0] - lp_center[0]) < 0.1
        assert abs(hp_center[1] - lp_center[1]) > 0.1


def test_detailed_webs_are_inset_within_spar_regions():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    section = get_detailed_cross_section(blade, 20, move_le_to_origin=True)
    web_layers = [region for region in section.regions if "_web" in region.name and "_layer00" in region.name]
    hp_spar = next(region for region in section.regions if region.name.startswith("Station020_HP_03_20_HP_SPAR_layer00"))
    lp_spar = next(region for region in section.regions if region.name.startswith("Station020_LP_08_20_LP_SPAR_layer00"))
    hp_inner_spar = next(region for region in section.regions if region.name.startswith("Station020_HP_03_20_HP_SPAR_layer03"))
    lp_inner_spar = next(region for region in section.regions if region.name.startswith("Station020_LP_08_20_LP_SPAR_layer03"))

    for web_layer in web_layers:
        hp_center, lp_center = _web_layer_centers(web_layer)

        assert 0.0 < _project_point_to_polyline_for_test(hp_inner_spar.inner_points, hp_center) < _polyline_length_for_test(hp_inner_spar.inner_points)
        assert 0.0 < _project_point_to_polyline_for_test(lp_inner_spar.inner_points, lp_center) < _polyline_length_for_test(lp_inner_spar.inner_points)
        assert _distance_to_polyline_for_test(hp_inner_spar.inner_points, hp_center) < _distance_to_polyline_for_test(hp_spar.outer_points, hp_center)
        assert _distance_to_polyline_for_test(lp_inner_spar.inner_points, lp_center) < _distance_to_polyline_for_test(lp_spar.outer_points, lp_center)


def test_detailed_webs_share_edges_with_inner_spar_layers():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    for station in [10, 20]:
        cs_params = _web_adhesive_cs_params(blade)
        section = get_detailed_cross_section(blade, station, cs_params=cs_params, move_le_to_origin=True)
        web_layers = [
            region
            for region in section.regions
            if "_web" in region.name and "_layer" in region.name and "_adhesive" not in region.name
        ]
        hp_inner_spar = next(
            region
            for region in section.regions
            if region.name.startswith(f"Station{station:03d}_HP_03_") and region.name.endswith("layer03")
        )
        lp_inner_spar = next(
            region
            for region in section.regions
            if region.name.startswith(f"Station{station:03d}_LP_08_") and region.name.endswith("layer03")
        )
        hp_inner_regions = [
            region
            for region in section.regions
            if region.name.startswith(f"Station{station:03d}_HP_") and "_layer" in region.name and region.edge_points is not None
        ]
        lp_inner_regions = [
            region
            for region in section.regions
            if region.name.startswith(f"Station{station:03d}_LP_") and "_layer" in region.name and region.edge_points is not None
        ]

        assert hp_inner_spar.edge_points is not None
        assert lp_inner_spar.edge_points is not None

        for web_layer in web_layers:
            web_prefix = web_layer.name.rsplit("_layer", 1)[0]
            hp_adhesive = next(region for region in section.regions if region.name == f"{web_prefix}_hp_adhesive")
            lp_adhesive = next(region for region in section.regions if region.name == f"{web_prefix}_lp_adhesive")

            assert web_layer.edge_points is not None
            assert hp_adhesive.edge_points is not None
            assert lp_adhesive.edge_points is not None
            assert any(_regions_share_edge(region, hp_adhesive) for region in hp_inner_regions)
            assert any(_regions_share_edge(region, lp_adhesive) for region in lp_inner_regions)
            assert _regions_share_edge(hp_adhesive, web_layer)
            assert _regions_share_edge(lp_adhesive, web_layer)


def test_detailed_web_adhesive_is_one_face_per_side():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")
    cs_params = _web_adhesive_cs_params(blade)

    section = get_detailed_cross_section(blade, 10, cs_params=cs_params, move_le_to_origin=True)
    adhesive_regions = [region for region in section.regions if "_web" in region.name and "_adhesive" in region.name]

    assert len(adhesive_regions) == 4
    assert all("_layer" not in region.name for region in adhesive_regions)
    assert all(region.edge_kinds.count("spline") == 4 for region in adhesive_regions)


def test_detailed_web_outer_layers_do_not_cross():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")
    cs_params = _web_adhesive_cs_params(blade)

    for station in [10, 20]:
        section = get_detailed_cross_section(blade, station, cs_params=cs_params, move_le_to_origin=True)
        for i_web in [0, 1]:
            first = next(region for region in section.regions if region.name == f"Station{station:03d}_web{i_web}_layer00")
            last = next(region for region in section.regions if region.name == f"Station{station:03d}_web{i_web}_layer02")
            first_hp, first_lp = _web_layer_centers(first)
            last_hp, last_lp = _web_layer_centers(last)

            assert not _segments_intersect(first_hp, first_lp, last_hp, last_lp)


def test_iea_station_025_web_layers_keep_matching_ply_widths():
    blade = pynumad.Blade("examples/example_data/IEA-22-280-RWT.yaml")
    cs_params = _web_adhesive_cs_params(blade)

    section = get_detailed_cross_section(blade, 25, cs_params=cs_params, move_le_to_origin=True)
    web_layers = [
        region
        for region in section.regions
        if region.name.startswith("Station025_web") and "_layer" in region.name
    ]

    assert len(web_layers) == 9
    for layer in web_layers:
        web_edges = [edge for edge in layer.edge_points if np.linalg.norm(edge[-1, :2] - edge[0, :2]) < 0.1]
        lengths = sorted(np.linalg.norm(edge[-1, :2] - edge[0, :2]) for edge in web_edges)

        assert len(lengths) == 2
        assert lengths[1] / lengths[0] < 1.01
        assert not _has_self_intersection(_region_polygon(layer))


def test_iea_yaml_web_components_use_explicit_web_assignments():
    blade = pynumad.Blade("examples/example_data/IEA-22-280-RWT.yaml")

    assert blade.stackdb.swstacks.shape[0] == 3
    for web_index, web_name in enumerate(["web0", "web1", "web2"]):
        components = [
            plygroup.component
            for plygroup in blade.stackdb.swstacks[web_index, 25].plygroups
        ]

        assert components == [
            f"{web_name}_skin00",
            f"{web_name}_filler",
            f"{web_name}_skin01",
        ]


def test_iea_station_025_draws_three_distinct_yaml_webs():
    blade = pynumad.Blade("examples/example_data/IEA-22-280-RWT.yaml")

    section = get_detailed_cross_section(blade, 25, move_le_to_origin=True)
    foam_layers = [
        region
        for region in section.regions
        if region.name.startswith("Station025_web") and region.name.endswith("layer01")
    ]
    centers = [np.vstack(region.edge_points)[:, :2].mean(axis=0) for region in foam_layers]
    separations = [
        np.linalg.norm(first - second)
        for i, first in enumerate(centers)
        for second in centers[i + 1 :]
    ]

    assert len(foam_layers) == 3
    assert min(separations) > 0.1


def test_iea_station_059_active_webs_attach_to_spar_caps():
    blade = pynumad.Blade("examples/example_data/IEA-22-280-RWT.yaml")
    cs_params = _web_adhesive_cs_params(blade)

    for station in [15, 25, 59]:
        section = get_detailed_cross_section(blade, station, cs_params=cs_params, move_le_to_origin=True)
        hp_inner_spar = next(
            region
            for region in section.regions
            if region.name.startswith(f"Station{station:03d}_HP_03_") and region.name.endswith("layer03")
        )
        lp_inner_spar = next(
            region
            for region in section.regions
            if region.name.startswith(f"Station{station:03d}_LP_08_") and region.name.endswith("layer03")
        )

        for web_index in [1, 2]:
            hp_adhesive = next(region for region in section.regions if region.name == f"Station{station:03d}_web{web_index}_hp_adhesive")
            lp_adhesive = next(region for region in section.regions if region.name == f"Station{station:03d}_web{web_index}_lp_adhesive")

            assert _regions_share_edge(hp_inner_spar, hp_adhesive)
            assert _regions_share_edge(lp_inner_spar, lp_adhesive)


def test_myblade_station_028_webs_do_not_intersect_neighbor_shell_panels():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    cs_params = _web_adhesive_cs_params(blade)

    section = get_detailed_cross_section(blade, 28, cs_params=cs_params, move_le_to_origin=True)
    web_regions = [region for region in section.regions if region.name.startswith("Station028_web")]
    shell_regions = [
        region
        for region in section.regions
        if region.name.startswith("Station028_HP_") or region.name.startswith("Station028_LP_")
    ]

    assert web_regions
    for web_region in web_regions:
        web_polygon = _region_polygon(web_region)
        for shell_region in shell_regions:
            assert not _polygons_have_crossing_edges(web_polygon, _region_polygon(shell_region))


def test_myblade_station_029_has_no_webs_when_web_stack_has_tapered_out():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")
    cs_params = _web_adhesive_cs_params(blade)

    section = get_detailed_cross_section(blade, 29, cs_params=cs_params, move_le_to_origin=True)

    assert not any(region.name.startswith("Station029_web") for region in section.regions)


def test_hp_le_final_layer_does_not_self_intersect():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    layer = next(region for region in section.regions if region.name.startswith("Station010_HP_05_10_HP_LE_layer03"))
    polygon = _region_polygon(layer)

    assert not _has_self_intersection(polygon)
    assert min(np.linalg.norm(np.diff(layer.outer_points[:, :2], axis=0), axis=1)) > 1e-3


def test_le_shell_layers_share_hp_lp_tip_offsets():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    for i_layer in range(4):
        hp_layer = next(region for region in section.regions if region.name.startswith(f"Station010_HP_05_10_HP_LE_layer{i_layer:02d}"))
        lp_layer = next(region for region in section.regions if region.name.startswith(f"Station010_LP_06_10_LP_LE_layer{i_layer:02d}"))

        np.testing.assert_allclose(hp_layer.outer_points[-1], lp_layer.outer_points[0])
        np.testing.assert_allclose(hp_layer.inner_points[-1], lp_layer.inner_points[0])


def test_station_010_le_points_do_not_protrude_past_shared_tip():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_cross_section(blade, 10, move_le_to_origin=True)
    le_x = section.hp_points[-1, 0]

    for points in (section.hp_points, section.lp_points):
        assert np.max(points[:, 0]) <= le_x + 1e-9

    detailed = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    for region in detailed.regions:
        if not (
            region.name.startswith("Station010_HP_05_")
            or region.name.startswith("Station010_LP_06_")
        ):
            continue
        for points in (region.outer_points, region.inner_points):
            assert np.max(points[:, 0]) <= le_x + 1e-9


def test_iea_station_002_le_keeps_real_rounded_nose_points():
    blade = pynumad.Blade("examples/example_data/IEA-22-280-RWT.yaml")

    section = get_cross_section(blade, 2, move_le_to_origin=True)
    detailed = get_detailed_cross_section(blade, 2, move_le_to_origin=True)
    lp_le_panel = next(
        region
        for region in detailed.regions
        if region.name.startswith("Station002_LP_07_") and region.name.endswith("layer00")
    )

    assert np.max(section.lp_points[:, 0]) > 0.01 * blade.geometry.ichord[2]
    assert np.max(lp_le_panel.outer_points[:, 0]) > 0.01 * blade.geometry.ichord[2]
    assert not _has_self_intersection(_region_polygon(lp_le_panel))


def test_shell_layers_terminate_at_trailing_edge_adhesive():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    for station in [10, 20]:
        section = get_detailed_cross_section(blade, station, move_le_to_origin=True)
        te_adhesive = next(region for region in section.regions if region.name == f"Station{station:03d}_TE_adhesive")
        shell_layers = [
            region
            for region in section.regions
            if region.outer_points is not None and region.name.endswith("layer00")
        ]
        first_layer = shell_layers[0]
        last_layer = shell_layers[-1]

        assert not np.allclose(first_layer.outer_points[0], last_layer.outer_points[-1])
        assert _region_boundary_contains_points(te_adhesive, first_layer.start_connector)
        assert _region_boundary_contains_points(te_adhesive, last_layer.end_connector)
        assert te_adhesive.material_name == "Adhesive"


def test_modified_blade_station_020_trailing_edge_adhesive_stays_open():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    for station in [10, 15, 20, 29]:
        section = get_detailed_cross_section(blade, station, move_le_to_origin=True)
        te_adhesive = next(region for region in section.regions if region.name == f"Station{station:03d}_TE_adhesive")
        trailing_edge_cap = te_adhesive.edge_points[-1]
        cut_gap = np.linalg.norm(te_adhesive.edge_points[0][-1] - te_adhesive.edge_points[4][0])

        assert te_adhesive.material_name == "Adhesive"
        assert len(te_adhesive.edge_points) == 6
        assert cut_gap > 0.01
        assert np.linalg.norm(trailing_edge_cap[0] - trailing_edge_cap[-1]) > 1e-6
        assert not _has_self_intersection(_region_polygon(te_adhesive))


def test_modified_blade_station_005_trailing_edge_adhesive_uses_small_gap():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(blade, 5, move_le_to_origin=True)
    te_adhesive = next(region for region in section.regions if region.name == "Station005_TE_adhesive")
    cut_gap = np.linalg.norm(te_adhesive.edge_points[0][-1] - te_adhesive.edge_points[4][0])

    assert cut_gap < 0.15
    assert not _has_self_intersection(_region_polygon(te_adhesive))


def test_iea_flatback_station_uses_flatback_trailing_edge_adhesive():
    blade = pynumad.Blade("examples/example_data/IEA-22-280-RWT.yaml")

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    flatback_adhesive = next(region for region in section.regions if region.name == "Station010_flatTEadhesive")
    flatback_opening = np.linalg.norm(section.hp_points[0] - section.lp_points[0])
    hp_first_layer = next(
        region
        for region in section.regions
        if region.name.startswith("Station010_HP_01_") and region.name.endswith("layer00")
    )
    lp_first_layer = next(
        region
        for region in section.regions
        if region.name.startswith("Station010_LP_10_") and region.name.endswith("layer00")
    )

    assert flatback_opening > 0.05 * blade.geometry.ichord[10]
    assert flatback_adhesive.material_name == "Adhesive"
    assert len(flatback_adhesive.edge_points) == 6
    adhesive_opening = np.linalg.norm(
        flatback_adhesive.edge_points[-1][0] - flatback_adhesive.edge_points[-1][-1]
    )
    assert np.isclose(adhesive_opening, flatback_opening)
    assert _region_boundary_contains_points(flatback_adhesive, hp_first_layer.start_connector)
    assert _region_boundary_contains_points(flatback_adhesive, lp_first_layer.end_connector)
    assert not any(region.name == "Station010_TE_adhesive" for region in section.regions)
    assert not _has_self_intersection(_region_polygon(flatback_adhesive))


def test_adjacent_shell_regions_use_stair_step_boundaries():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    regions_by_name = {region.name: region for region in section.regions}

    for i_segment in range(5):
        first_stack = blade.stackdb.stacks[i_segment, 10]
        second_stack = blade.stackdb.stacks[i_segment + 1, 10]
        first = regions_by_name.get(f"Station010_HP_{first_stack.name}_layer02")
        second = regions_by_name.get(f"Station010_HP_{second_stack.name}_layer02")
        if first is None or second is None:
            continue
        _assert_stair_boundary(first, second)

    for i_segment in range(6, 11):
        first_stack = blade.stackdb.stacks[i_segment, 10]
        second_stack = blade.stackdb.stacks[i_segment + 1, 10]
        first = regions_by_name.get(f"Station010_LP_{first_stack.name}_layer02")
        second = regions_by_name.get(f"Station010_LP_{second_stack.name}_layer02")
        if first is None or second is None:
            continue
        _assert_stair_boundary(first, second)


def test_shell_stair_step_short_edges_are_perpendicular():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    for station in [10, 20]:
        section = get_detailed_cross_section(blade, station, move_le_to_origin=True)
        for region in section.regions:
            if region.outer_points is None or region.inner_points is None:
                continue
            if region.start_connector is not None:
                if len(region.start_connector) == 3:
                    _assert_step_perpendicular(region.start_connector[0] - region.start_connector[1], region.inner_points[1] - region.inner_points[0])
                    _assert_step_perpendicular(region.start_connector[1] - region.start_connector[2], region.outer_points[1] - region.outer_points[0])
            if region.end_connector is not None:
                if len(region.end_connector) == 3:
                    _assert_step_perpendicular(region.end_connector[-1] - region.end_connector[-2], region.inner_points[-1] - region.inner_points[-2])
                    _assert_step_perpendicular(region.end_connector[1] - region.end_connector[0], region.outer_points[-1] - region.outer_points[-2])


def test_disconnected_shell_boundaries_use_square_caps():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    targets = [
        ("Station010_HP_03_10_HP_SPAR_layer03", "start"),
        ("Station010_HP_03_10_HP_SPAR_layer03", "end"),
        ("Station010_HP_04_10_HP_LE_PANEL_layer03", "start"),
        ("Station010_HP_04_10_HP_LE_PANEL_layer03", "end"),
    ]
    for name, side in targets:
        region = next(region for region in section.regions if region.name == name)
        if side == "start":
            _assert_step_perpendicular(region.start_connector[0] - region.start_connector[1], region.inner_points[1] - region.inner_points[0])
        else:
            _assert_step_perpendicular(region.end_connector[-1] - region.end_connector[-2], region.inner_points[-1] - region.inner_points[-2])


def test_detailed_shell_regions_do_not_include_zero_length_boundaries():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    for station in [10, 20]:
        section = get_detailed_cross_section(blade, station, move_le_to_origin=True)
        for region in section.regions:
            if region.edge_points is not None:
                for edge_points in region.edge_points:
                    assert np.linalg.norm(np.diff(edge_points[:, :2], axis=0), axis=1).sum() > 1e-9
                continue
            if region.outer_points is None:
                continue
            assert np.linalg.norm(np.diff(region.outer_points[:, :2], axis=0), axis=1).sum() > 1e-9
            assert np.linalg.norm(np.diff(region.inner_points[:, :2], axis=0), axis=1).sum() > 1e-9
            assert np.linalg.norm(np.diff(region.start_connector[:, :2], axis=0), axis=1).min() > 1e-9
            assert np.linalg.norm(np.diff(region.end_connector[:, :2], axis=0), axis=1).min() > 1e-9


def test_detailed_shell_regions_do_not_self_intersect():
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    for station in [10, 20]:
        section = get_detailed_cross_section(blade, station, move_le_to_origin=True)
        for region in section.regions:
            if region.outer_points is None and region.edge_points is None:
                continue
            assert not _has_self_intersection(_region_polygon(region)), region.name


def test_write_detailed_freecad_cross_sections_script(tmp_path):
    blade = pynumad.Blade("src/pynumad/tests/test_data/blades/blade.yaml")

    script_path = write_freecad_cross_sections(
        blade,
        "blade",
        station_list=[10],
        directory=tmp_path,
        move_le_to_origin=True,
        detailed=True,
    )

    contents = script_path.read_text(encoding="utf-8")
    assert '"detailed": true' in contents
    assert '"regions": [' in contents
    assert "face_between_curves" in contents
    assert "sewShape" in contents
    assert "FaceMaterialMap" in contents
    assert "TurbineMetadata" in contents
    assert "LaminateDefinitions" in contents
    assert "MaterialDefinitions" in contents
    assert "StationCount" in contents
    assert "face_index" in contents
    assert "assignment_index" in contents
    assert '"station"' in contents
    assert '"side"' in contents
    assert '"web_index"' in contents
    assert "_section" in contents
    assert '"debug_faces": false' in contents
    assert '"start_connector": [' in contents


def test_cs_params_geometry_scaling_generates_millimeter_sections():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    meter_section = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    mm_section = get_detailed_cross_section(
        blade,
        10,
        move_le_to_origin=True,
        cs_params={"geometry_scaling": 1000.0},
    )

    assert np.allclose(mm_section.hp_points, 1000.0 * meter_section.hp_points)
    assert np.allclose(mm_section.lp_points, 1000.0 * meter_section.lp_points)
    meter_region = next(region for region in meter_section.regions if region.name == "Station010_HP_02_10_HP_TE_PANEL_layer01")
    mm_region = next(region for region in mm_section.regions if region.name == meter_region.name)
    assert np.isclose(mm_region.plies[0]["thickness"], 1000.0 * meter_region.plies[0]["thickness"])


def test_cs_params_move_le_to_origin_overrides_function_argument():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(
        blade,
        10,
        move_le_to_origin=True,
        cs_params={"geometry_scaling": 1000.0, "move_le_to_origin": False},
    )

    assert not np.allclose(section.hp_points[-1, :2], [0.0, 0.0])


def test_move_le_to_origin_false_keeps_station_coordinates():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    shifted = get_detailed_cross_section(blade, 10, move_le_to_origin=True)
    unshifted = get_detailed_cross_section(blade, 10, move_le_to_origin=False)

    assert np.allclose(shifted.hp_points[-1, :2], [0.0, 0.0])
    assert not np.allclose(unshifted.hp_points[-1, :2], [0.0, 0.0])


def test_move_le_to_origin_shifts_station_frame_by_section_translation():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    unshifted = get_detailed_cross_section(
        blade,
        10,
        move_le_to_origin=False,
        cs_params={"geometry_scaling": 1000.0},
    )
    shifted = get_detailed_cross_section(
        blade,
        10,
        move_le_to_origin=True,
        cs_params={"geometry_scaling": 1000.0},
    )

    translation = np.array(shifted.station_frame["section_translation"])
    np.testing.assert_allclose(translation, -unshifted.hp_points[-1])
    np.testing.assert_allclose(
        shifted.station_frame["origin"],
        np.array(unshifted.station_frame["origin"]) + translation,
    )
    np.testing.assert_allclose(
        shifted.station_frame["lcs"]["origin"],
        shifted.station_frame["origin"],
    )
    np.testing.assert_allclose(
        shifted.station_frame["reference_lcs"]["origin"],
        shifted.station_frame["origin"],
    )


def test_station_20_homogen_lcs_keeps_section_in_xy_plane():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    section = get_detailed_cross_section(
        blade,
        20,
        move_le_to_origin=True,
        cs_params={"geometry_scaling": 1000.0},
    )
    basis = np.column_stack(
        (
            section.station_frame["lcs"]["x_axis"],
            section.station_frame["lcs"]["y_axis"],
            section.station_frame["lcs"]["z_axis"],
        )
    )
    points = np.vstack((section.hp_points, section.lp_points))
    origin = np.array(section.station_frame["origin"])
    transformed_points = origin + points @ basis.T
    local_points = (transformed_points - origin) @ basis

    np.testing.assert_allclose(points[:, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(local_points[:, 2], 0.0, atol=1e-9)
    np.testing.assert_allclose(basis[2, :2], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(basis[:, 2], [0.0, 0.0, 1.0], atol=1e-12)


def test_write_detailed_script_uses_cs_params_geometry_scaling_for_laminates(tmp_path):
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    script_path = write_freecad_cross_sections(
        blade,
        "blade",
        station_list=[10],
        directory=tmp_path,
        move_le_to_origin=True,
        detailed=True,
        cs_params={"geometry_scaling": 1000.0},
    )

    contents = script_path.read_text(encoding="utf-8")
    assert '"thickness": 1.0' in contents


def _has_self_intersection(points):
    for i_point in range(len(points)):
        first_start = points[i_point]
        first_end = points[(i_point + 1) % len(points)]
        for j_point in range(i_point + 1, len(points)):
            if abs(i_point - j_point) <= 1 or (i_point == 0 and j_point == len(points) - 1):
                continue
            second_start = points[j_point]
            second_end = points[(j_point + 1) % len(points)]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return True
    return False


def _assert_stair_boundary(first, second):
    np.testing.assert_allclose(first.outer_points[-1], second.outer_points[0])
    if np.allclose(first.inner_points[-1], second.inner_points[0]):
        return

    first_has_second_inner = np.any(np.all(np.isclose(first.end_connector, second.inner_points[0]), axis=1))
    second_has_first_inner = np.any(np.all(np.isclose(second.start_connector, first.inner_points[-1]), axis=1))
    assert first_has_second_inner or second_has_first_inner


def _assert_step_perpendicular(step, tangent):
    step_norm = np.linalg.norm(step[:2])
    tangent_norm = np.linalg.norm(tangent[:2])
    assert step_norm > 1e-9
    assert tangent_norm > 1e-9
    assert abs(np.dot(step[:2] / step_norm, tangent[:2] / tangent_norm)) < 1e-6


def _web_layer_centers(web_layer):
    if web_layer.edge_points is not None:
        hp_edge = web_layer.edge_points[0]
        lp_edge = web_layer.edge_points[2]
        return hp_edge.mean(axis=0), lp_edge.mean(axis=0)
    return (web_layer.points[0] + web_layer.points[3]) / 2, (web_layer.points[1] + web_layer.points[2]) / 2


def _web_adhesive_cs_params(blade):
    adhesive_thickness = np.full((len(blade.ispan),), 0.001)
    return {
        "adhesive_mat_name": "Adhesive",
        "web_fore_adhesive_thickness": adhesive_thickness,
        "web_aft_adhesive_thickness": adhesive_thickness,
    }


def _region_has_edge(region, edge):
    for candidate in _region_edges(region):
        if _edges_match(candidate, edge):
            return True
    return False


def _regions_share_edge(first, second):
    return any(_region_has_edge(first, edge) for edge in _region_edges(second))


def _region_edges(region):
    if region.edge_points is not None:
        return region.edge_points
    if region.outer_points is None:
        return []
    return [
        region.outer_points,
        region.end_connector,
        region.inner_points,
        region.start_connector,
    ]


def _region_boundary_contains_points(region, points):
    return all(
        any(np.allclose(point, candidate) for edge in region.edge_points or [] for candidate in edge)
        for point in points
    )


def _edges_match(first, second):
    if first.shape != second.shape:
        return False
    return np.allclose(first, second) or np.allclose(first, np.flip(second, axis=0))


def _region_polygon(region):
    if region.edge_points is not None:
        points = []
        for edge_points in region.edge_points:
            if not points:
                points.extend(edge_points)
            else:
                points.extend(edge_points[1:])
        return np.array(points)
    return np.vstack(
        (
            region.outer_points,
            region.end_connector[1:],
            np.flip(region.inner_points, axis=0),
            region.start_connector[1:],
        )
    )


def _polygon_area(points):
    if not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    return 0.5 * abs(
        np.dot(points[:-1, 0], points[1:, 1]) - np.dot(points[1:, 0], points[:-1, 1])
    )


def _unit_2d(vector):
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return np.array([1.0, 0.0])
    return vector / norm


def _segments_colinear(first_start, first_end, second_start, second_end, tolerance=1e-9):
    first = first_end[:2] - first_start[:2]
    second = second_end[:2] - second_start[:2]
    if np.linalg.norm(first) <= tolerance or np.linalg.norm(second) <= tolerance:
        return False
    return abs(np.cross(first, second)) / (np.linalg.norm(first) * np.linalg.norm(second)) <= tolerance


def _segments_intersect(first_start, first_end, second_start, second_end):
    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)
    return first_orientation * second_orientation < -1e-12 and third_orientation * fourth_orientation < -1e-12


def _polygons_have_crossing_edges(first, second):
    for first_start, first_end in zip(first[:-1], first[1:]):
        for second_start, second_end in zip(second[:-1], second[1:]):
            if _segments_share_endpoint(first_start, first_end, second_start, second_end):
                continue
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return True
    return False


def _segments_share_endpoint(first_start, first_end, second_start, second_end):
    return (
        np.linalg.norm(first_start[:2] - second_start[:2]) < 1e-7
        or np.linalg.norm(first_start[:2] - second_end[:2]) < 1e-7
        or np.linalg.norm(first_end[:2] - second_start[:2]) < 1e-7
        or np.linalg.norm(first_end[:2] - second_end[:2]) < 1e-7
    )


def _orientation(start, end, point):
    return np.cross(end[:2] - start[:2], point[:2] - start[:2])


def _polyline_length_for_test(points):
    return np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum()


def _project_point_to_polyline_for_test(points, point):
    cumulative = 0.0
    best_distance = float("inf")
    best_s = 0.0
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared == 0:
            continue
        t = np.clip(float(np.dot(point - start, segment) / length_squared), 0.0, 1.0)
        projected = start + t * segment
        distance = np.linalg.norm(point - projected)
        if distance < best_distance:
            best_distance = distance
            best_s = cumulative + t * np.sqrt(length_squared)
        cumulative += np.sqrt(length_squared)
    return best_s


def _distance_to_polyline_for_test(points, point):
    best_distance = float("inf")
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared == 0:
            continue
        t = np.clip(float(np.dot(point - start, segment) / length_squared), 0.0, 1.0)
        projected = start + t * segment
        best_distance = min(best_distance, np.linalg.norm(point - projected))
    return best_distance
