import pynumad
import numpy as np

from pynumad.analysis.freecad import (
    face_material_metadata,
    get_cross_section,
    get_detailed_cross_section,
    laminate_definitions,
    material_definitions,
    make_freecad_cross_section_parts,
    make_freecad_section_part,
    write_freecad_cross_sections,
)


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
        "assignment_name",
    }

    assert metadata
    assert all(set(item) == expected_keys for item in metadata)
    assert all(key == key.lower() for key in expected_keys)
    assert all("_" in key or key in {"side", "layer", "station"} for key in expected_keys)
    assert all("RegionName" not in item and "Feature" not in item for item in metadata)


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


def test_material_definitions_include_elastic_density_and_thermal_data():
    blade = pynumad.Blade("examples/example_data/myBlade_Modified.yaml")

    materials = material_definitions(blade)
    by_name = {item["material_name"]: item for item in materials}

    assert by_name["glass_triax"]["material_type"] == "orthotropic"
    assert by_name["glass_triax"]["density"] == 1940.0
    assert by_name["glass_triax"]["elastic"]["e1"] == 28211400000.0
    assert by_name["glass_triax"]["elastic"]["g23"] == 3491240000.0
    assert by_name["Gelcoat"]["material_type"] == "isotropic"
    assert by_name["Gelcoat"]["elastic"]["youngs_modulus"] == 3440000000.0
    assert by_name["Gelcoat"]["thermal"]["expansion_coefficient"] == 0.0
    assert by_name["Gelcoat"]["strength"]["compressive"] == 10000000000.0
    assert "thermal" not in by_name["glass_triax"]


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

        assert hp_inner_spar.edge_points is not None
        assert lp_inner_spar.edge_points is not None

        for web_layer in web_layers:
            web_prefix = web_layer.name.rsplit("_layer", 1)[0]
            hp_adhesive = next(region for region in section.regions if region.name == f"{web_prefix}_hp_adhesive")
            lp_adhesive = next(region for region in section.regions if region.name == f"{web_prefix}_lp_adhesive")

            assert web_layer.edge_points is not None
            assert hp_adhesive.edge_points is not None
            assert lp_adhesive.edge_points is not None
            assert _regions_share_edge(hp_inner_spar, hp_adhesive)
            assert _regions_share_edge(lp_inner_spar, lp_adhesive)
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
    web_layers = [region for region in section.regions if region.name.startswith("Station025_web1_layer")]

    assert len(web_layers) == 7
    for layer in web_layers:
        web_edges = [edge for edge in layer.edge_points if np.linalg.norm(edge[-1, :2] - edge[0, :2]) < 0.1]
        lengths = sorted(np.linalg.norm(edge[-1, :2] - edge[0, :2]) for edge in web_edges)

        assert len(lengths) == 2
        assert lengths[1] / lengths[0] < 1.01
        assert not _has_self_intersection(_region_polygon(layer))


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
    assert "LaminateDefinitions" in contents
    assert "MaterialDefinitions" in contents
    assert "face_index" in contents
    assert '"station"' in contents
    assert '"side"' in contents
    assert '"web_index"' in contents
    assert "_section" in contents
    assert '"debug_faces": false' in contents
    assert '"start_connector": [' in contents


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
    for candidate in region.edge_points or []:
        if _edges_match(candidate, edge):
            return True
    return False


def _regions_share_edge(first, second):
    return any(_region_has_edge(first, edge) for edge in second.edge_points or [])


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


def _segments_intersect(first_start, first_end, second_start, second_end):
    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)
    return first_orientation * second_orientation < -1e-12 and third_orientation * fourth_orientation < -1e-12


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
