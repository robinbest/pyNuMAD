"""FreeCAD export helpers for pyNuMAD blade cross sections.

The public entry points build either a simple airfoil wire or a detailed 2D
section made from shell laminate faces, shear-web faces, and adhesive faces.
The detailed path follows this sequence:

* read station-local HP/LP curves and keypoints from the blade object;
* split the outer perimeter into stack regions;
* trim round/small trailing edges so HP and LP laminates terminate before
  touching, or preserve large flatback trailing edges as supplied;
* offset each ply layer inward by its material thickness;
* square layer boundaries where adjacent stacks have different thicknesses;
* add shear-web laminates and adhesive regions where web stacks exist;
* add either a round-TE adhesive face or a flatback-TE adhesive face;
* convert each region into FreeCAD faces and store material metadata by face;
* store per-station reference-axis and local-coordinate-system metadata for
  downstream curved-blade placement.

Geometry is represented as NumPy arrays of 3D points, but all intersection,
projection, orientation, and offset logic is intentionally 2D in the station
cross-section plane using x/y coordinates.  The z coordinate is carried through
unchanged so FreeCAD receives valid 3D points.

This module intentionally covers 2D cross-section construction for FreeCAD and
HomoGen workflows.  It is not a drop-in replacement for the Cubit meshing
workflow in ``pynumad.analysis.cubit``.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import yaml

from pynumad.objects.stack import Ply, Stack


@dataclass
class FreeCADCrossSection:
    """Station-local blade section data arranged for FreeCAD curve creation."""

    station: int
    te_point: np.ndarray
    hp_points: np.ndarray
    lp_points: np.ndarray
    station_frame: dict = None

    @property
    def closed_points(self):
        """Return points ordered around the actual airfoil perimeter.

        The YAML TE point is often the midpoint of the HP/LP trailing-edge
        endpoints.  For flatback stations it is not an OML vertex; including it
        would make offset logic see a pointed tail instead of the physical
        flatback wall, which creates overlaps where flatback adhesive meets the
        shell.
        """

        return _clean_polygon_points(
            np.vstack((self.hp_points, np.flip(self.lp_points[:-1], axis=0)))
        )


@dataclass
class FreeCADFaceRegion:
    """A named cross-section face region for the generated FreeCAD script.

    A region is described in one of three ways:

    * ``points``: one closed polygonal face boundary;
    * ``outer_points``/``inner_points`` plus optional end connectors: a strip
      between two curves, usually one laminate ply region;
    * ``edge_points``/``edge_kinds``: an explicit ordered boundary with spline
      and line edges, used for adhesives, webs, and split spar faces.

    ``material_name`` is passed through to the FreeCAD face metadata so
    downstream tools can recover material assignments.  Composite regions can
    also carry ``laminate_name`` and ``plies`` for tools such as HomoGen that
    assign a face to a named laminate made from ply entries.
    """

    name: str
    material_name: str
    ply_angle: float
    laminate_name: str = None
    plies: list = None
    points: np.ndarray = None
    outer_points: np.ndarray = None
    inner_points: np.ndarray = None
    start_connector: np.ndarray = None
    end_connector: np.ndarray = None
    edge_points: list = None
    edge_kinds: list = None


@dataclass
class FreeCADDetailedCrossSection(FreeCADCrossSection):
    """Detailed station data with shell, web, and adhesive face regions."""

    regions: list = None
    material_definitions: list = None


def get_cross_section(
    blade,
    station,
    *,
    geometry_scaling=1.0,
    normalize_chord=False,
    move_le_to_origin=False,
):
    """Return HP and LP points for a blade station without using Cubit.

    Parameters
    ----------
    blade : pynumad.objects.blade.Blade
        Blade object with populated geometry.
    station : int
        Geometry station index.
    geometry_scaling : float, optional
        Scale applied to coordinates before optional chord normalization.
    normalize_chord : bool, optional
        If true, divide coordinates by the station chord after scaling.
    move_le_to_origin : bool, optional
        If true, translate all points so the leading edge is at ``(0, 0, 0)``.

    Returns
    -------
    FreeCADCrossSection
        High-pressure and low-pressure point arrays.  HP runs TE-to-LE, LP runs
        TE-to-LE, matching the Cubit cross-section setup.
    """

    geometry = blade.geometry
    i_le = geometry.LEindex + 1
    xyz = np.array(
        [
            geometry.coordinates[:, 0, station],
            geometry.coordinates[:, 1, station],
            geometry.coordinates[:, 2, station],
        ]
    ).transpose()
    xyz = xyz * geometry_scaling

    if normalize_chord:
        chord = geometry.ichord[station] * geometry_scaling
        xyz = xyz / chord

    if move_le_to_origin:
        xyz = xyz - xyz[i_le - 1, :]

    hp_points = xyz[1:i_le, :]
    lp_points = np.flip(xyz, axis=0)[1:i_le, :]
    _clamp_le_surface_protrusion(hp_points, lp_points, xyz[0, :], xyz[i_le - 1, :])
    return FreeCADCrossSection(
        station=station,
        te_point=xyz[0, :],
        hp_points=hp_points,
        lp_points=lp_points,
        station_frame=station_frame_definition(blade, station),
    )


def get_detailed_cross_section(
    blade,
    station,
    *,
    cs_params=None,
    geometry_scaling=1.0,
    normalize_chord=False,
    move_le_to_origin=False,
):
    """Return shell, web, and adhesive face regions for one blade station.

    This is a FreeCAD-oriented approximation of the Cubit 2D section setup.  It
    uses pyNuMAD keypoints to split the shell into material stack regions, offsets
    each region inward by the local laminate thickness, and adds shear-web faces
    for stations where web stacks are present.
    """

    section = get_cross_section(
        blade,
        station,
        geometry_scaling=geometry_scaling,
        normalize_chord=normalize_chord,
        move_le_to_origin=move_le_to_origin,
    )
    transformer = _StationTransformer(
        blade,
        station,
        geometry_scaling=geometry_scaling,
        normalize_chord=normalize_chord,
        move_le_to_origin=move_le_to_origin,
    )

    regions = _shell_regions(blade, station, section, transformer, cs_params or {})
    regions.extend(_web_regions(blade, station, transformer, cs_params or {}, regions))

    return FreeCADDetailedCrossSection(
        station=section.station,
        te_point=section.te_point,
        hp_points=section.hp_points,
        lp_points=section.lp_points,
        station_frame=section.station_frame,
        regions=regions,
        material_definitions=material_definitions(blade),
    )


def write_freecad_cross_sections(
    blade,
    wt_name,
    *,
    station_list=None,
    directory=".",
    geometry_scaling=1.0,
    normalize_chord=False,
    move_le_to_origin=False,
    make_faces=True,
    export_step=False,
    detailed=False,
    cs_params=None,
):
    """Write a FreeCAD Python script that creates selected 2D cross sections.

    The generated script is meant to be run with FreeCAD or ``freecadcmd``.  It
    creates B-splines for the HP and LP surfaces, a straight trailing-edge line,
    and optionally a planar face for each section.

    Returns
    -------
    pathlib.Path
        Path to the generated FreeCAD script.
    """

    if station_list is None or len(station_list) == 0:
        station_list = list(range(len(blade.definition.ispan)))

    out_dir = Path(directory).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    section_builder = get_detailed_cross_section if detailed else get_cross_section
    sections = []
    for station in station_list:
        kwargs = {
            "geometry_scaling": geometry_scaling,
            "normalize_chord": normalize_chord,
            "move_le_to_origin": move_le_to_origin,
        }
        if detailed:
            kwargs["cs_params"] = cs_params
        sections.append(section_builder(blade, station, **kwargs))

    script_path = out_dir / f"{wt_name}_freecad_cross_sections.py"
    payload = {
        "wt_name": wt_name,
        "make_faces": make_faces,
        "detailed": detailed,
        "debug_faces": bool((cs_params or {}).get("debug_faces", False)) if detailed else False,
        "export_step": export_step,
        "fcstd_path": os.fspath(out_dir / f"{wt_name}_cross_sections.FCStd"),
        "step_path": os.fspath(out_dir / f"{wt_name}_cross_sections.step"),
        "station_count": blade_station_count(blade),
        "material_definitions": material_definitions(blade) if detailed else [],
        "laminate_definitions": global_laminate_definitions(
            blade,
            geometry_scaling=geometry_scaling,
            normalize_chord=normalize_chord,
        )
        if detailed
        else [],
        "sections": [
            {
                "station": section.station,
                "hp": section.hp_points.tolist(),
                "lp": section.lp_points.tolist(),
                "station_frame": section.station_frame,
                "regions": _serialize_regions(getattr(section, "regions", None)),
            }
            for section in sections
        ],
    }

    script_path.write_text(_freecad_script(payload), encoding="utf-8")
    return script_path


def make_freecad_cross_section_parts(
    blade,
    *,
    station_list=None,
    doc=None,
    geometry_scaling=1.0,
    normalize_chord=False,
    move_le_to_origin=False,
    detailed=True,
    cs_params=None,
    debug_faces=False,
):
    """Create FreeCAD cross-section objects directly in a document.

    This is the API intended for FreeCAD workbenches and modules such as
    HomoGen.  It avoids the intermediate generated Python script and STEP file:
    pyNuMAD builds the section topology, FreeCAD receives one stitched object
    per station, and material metadata is stored on each object as a JSON
    ``FaceMaterialMap`` property.

    Parameters are intentionally aligned with :func:`write_freecad_cross_sections`.

    Returns
    -------
    list
        The created FreeCAD ``Part::Feature`` objects.
    """

    if station_list is None or len(station_list) == 0:
        station_list = list(range(len(blade.definition.ispan)))

    if doc is None:
        App, _ = _require_freecad_modules()
        doc = App.ActiveDocument or App.newDocument("pyNuMAD_cross_sections")

    material_table = material_definitions(blade) if detailed else []
    laminate_table = (
        global_laminate_definitions(
            blade,
            geometry_scaling=geometry_scaling,
            normalize_chord=normalize_chord,
        )
        if detailed
        else []
    )
    if detailed:
        metadata_obj = _make_turbine_metadata_object(
            doc,
            material_table=material_table,
            laminate_table=laminate_table,
            station_count=blade_station_count(blade),
        )
    else:
        metadata_obj = None

    section_builder = get_detailed_cross_section if detailed else get_cross_section
    created = []
    messages = []
    for station in station_list:
        kwargs = {
            "geometry_scaling": geometry_scaling,
            "normalize_chord": normalize_chord,
            "move_le_to_origin": move_le_to_origin,
        }
        if detailed:
            kwargs["cs_params"] = cs_params
        section = section_builder(blade, station, **kwargs)
        created.append(
            make_freecad_section_part(
                section,
                doc=doc,
                debug_faces=debug_faces,
                laminate_table=laminate_table,
                material_table=material_table,
                message_log=messages,
            )
        )

    if metadata_obj is not None:
        _set_turbine_messages(metadata_obj, messages)
    doc.recompute()
    return created


def load_blade_for_freecad(yaml_file, *, doc=None):
    """Load a pyNuMAD blade and record YAML/load failures on ``TurbineMetadata``.

    This helper is intended for FreeCAD/HomoGen callers that need user-facing
    error messages even when blade construction fails before section generation
    can create station objects.
    """

    import pynumad

    try:
        return pynumad.Blade(yaml_file)
    except Exception as exc:
        record_turbine_message(
            doc,
            "error",
            "blade_yaml_read_failed",
            str(exc),
            source="freecad_cross_sections.blade_load",
            details={
                "yaml_file": os.fspath(yaml_file),
                "exception_type": type(exc).__name__,
            },
        )
        return None


def record_turbine_message(doc, severity, code, message, *, station=None, source=None, details=None):
    """Append a warning/error message to the document-level ``TurbineMetadata``.

    Messages are stored as JSON strings in ``WarningMessages`` and
    ``ErrorMessages`` so downstream tools have one stable place to inspect
    FreeCAD section-generation issues.
    """

    if doc is None:
        App, _ = _require_freecad_modules()
        doc = App.ActiveDocument or App.newDocument("pyNuMAD_cross_sections")
    metadata_obj = _get_or_create_turbine_metadata_object(doc)
    _initialize_turbine_metadata_defaults(metadata_obj)
    _append_turbine_messages(
        metadata_obj,
        [
            _generation_message(
                severity,
                code,
                message,
                station=station,
                source=source,
                **(details or {}),
            )
        ],
    )
    return metadata_obj


def make_freecad_section_part(
    section,
    *,
    doc=None,
    name=None,
    debug_faces=False,
    laminate_table=None,
    material_table=None,
    message_log=None,
):
    """Create one FreeCAD object from a pyNuMAD cross-section data object.

    Detailed sections become a sewn shell with one face per material region.
    The returned object gets station-local JSON string properties:

    * ``FaceMaterialMap``: one entry per ``obj.Shape.Faces`` item, with each
      face assigned to either a material or laminate table entry;
    * ``StationFrame``: station reference-axis origin, bend/sweep rotations,
      twist, and local coordinate system.

    When ``laminate_table`` and ``material_table`` are provided, face metadata
    references those turbine-level tables by index.  Standalone calls without
    tables still get a section-local laminate table for diagnostics.
    """

    App, Part = _require_freecad_modules()
    if doc is None:
        doc = App.ActiveDocument or App.newDocument("pyNuMAD_cross_sections")

    if getattr(section, "regions", None):
        face_shapes = []
        for region in section.regions:
            face_shape = _freecad_face_from_region(region, App, Part)
            face_shapes.append(face_shape)
            if debug_faces:
                face_obj = doc.addObject("Part::Feature", region.name)
                face_obj.Shape = face_shape
                face_obj.Label = f"{region.name} | {region.material_name} | angle {region.ply_angle}"
                _set_view_color(face_obj, _color_for_material(region.material_name))

        obj_name = name or f"Station{section.station:03d}_section"
        section_obj = doc.addObject("Part::Feature", obj_name)
        section_obj.Shape = _freecad_stitched_section_shape(face_shapes, Part)
        section_obj.Label = f"Station {section.station:03d}"
        face_ordered_regions, face_map_messages = _regions_in_shape_face_order_with_messages(
            section.regions,
            face_shapes,
            section_obj.Shape,
            station=section.station,
        )
        face_map_messages.extend(_shell_laminate_vertex_contact_messages(face_ordered_regions))
        if message_log is not None:
            message_log.extend(face_map_messages)
        elif face_map_messages:
            metadata_obj = doc.getObject("TurbineMetadata") if hasattr(doc, "getObject") else None
            if metadata_obj is not None:
                _append_turbine_messages(metadata_obj, face_map_messages)
        _set_string_property(
            section_obj,
            "FaceMaterialMap",
            json.dumps(
                face_material_metadata(
                    face_ordered_regions,
                    laminate_table=laminate_table,
                    material_table=material_table,
                )
            ),
            group="Turbine",
            description="JSON map from face index to material metadata",
        )
        _set_string_property(
            section_obj,
            "StationFrame",
            json.dumps(getattr(section, "station_frame", None) or {}),
            group="Turbine",
            description="JSON station reference-axis origin, rotations, and local coordinate system",
        )
        _set_view_color(section_obj, (0.78, 0.82, 0.86, 0.0))
        return section_obj

    hp_edge = _freecad_bspline_edge(section.hp_points, App, Part)
    lp_edge = _freecad_bspline_edge(list(reversed(section.lp_points)), App, Part)
    te_edge = Part.LineSegment(
        _freecad_vector(section.lp_points[0], App),
        _freecad_vector(section.hp_points[0], App),
    ).toShape()
    wire_obj = doc.addObject("Part::Feature", name or f"Station{section.station:03d}_wire")
    wire_obj.Shape = Part.Wire([hp_edge, lp_edge, te_edge])
    _set_string_property(
        wire_obj,
        "StationFrame",
        json.dumps(getattr(section, "station_frame", None) or {}),
        group="Turbine",
        description="JSON station reference-axis origin, rotations, and local coordinate system",
    )
    return wire_obj


def face_material_metadata(regions, *, laminate_table=None, material_table=None):
    """Return face-index/material metadata for serialized or object regions.

    Laminate faces reference a laminate table by ``assignment_index`` and
    ``assignment_name``.  If no table is supplied, a section-local table is
    built for diagnostic/backward-compatible calls.  FreeCAD export paths pass
    turbine-level laminate and material tables so every station references the
    same definitions.
    """

    metadata = []
    laminate_table = laminate_table if laminate_table is not None else laminate_definitions(regions)
    laminate_index_by_key = {
        _laminate_key(item["plies"]): item["laminate_index"]
        for item in laminate_table
    }
    material_index_by_name = {
        item["material_name"]: item.get("material_index")
        for item in material_table or []
    }
    for index, region in enumerate(regions or []):
        region_name = _region_value(region, "name")
        plies = _region_value(region, "plies") or []
        laminate_index = laminate_index_by_key.get(_laminate_key(plies)) if plies else None
        assignment_type = "laminate" if laminate_index is not None else "material"
        material_name = _region_value(region, "material_name")
        assignment_index = (
            laminate_index
            if laminate_index is not None
            else material_index_by_name.get(material_name)
        )
        assignment_name = (
            laminate_table[laminate_index]["laminate_name"]
            if laminate_index is not None
            else material_name
        )
        item = _parsed_region_name(region_name)
        item.update(
            dict(
                face_index=index,
                region_name=region_name,
                material_name=material_name,
                assignment_type=assignment_type,
                assignment_index=assignment_index,
                assignment_name=assignment_name,
            )
        )
        metadata.append(item)
    return metadata


def _regions_in_shape_face_order(regions, source_faces, stitched_shape):
    """Return regions reordered to match the actual FreeCAD ``Shape.Faces`` list."""

    ordered_regions, _ = _regions_in_shape_face_order_with_messages(
        regions,
        source_faces,
        stitched_shape,
    )
    return ordered_regions


def _regions_in_shape_face_order_with_messages(regions, source_faces, stitched_shape, *, station=None):
    """Return regions reordered to match the actual FreeCAD ``Shape.Faces`` list.

    ``Part.makeCompound`` followed by ``sewShape`` can reorder faces.  The
    material map is consumed through FreeCAD face indices, so match each sewn
    face back to its source face using simple geometric signatures.  Failures
    are returned as lightweight generation messages for ``TurbineMetadata``.
    """

    shape_faces = list(getattr(stitched_shape, "Faces", []) or [])
    regions = list(regions or [])
    source_faces = list(source_faces or [])
    messages = []
    if len(shape_faces) != len(regions) or len(source_faces) != len(regions):
        messages.append(
            _generation_message(
                "error",
                "face_count_mismatch",
                (
                    "FaceMaterialMap face-order verification failed because the generated region count, "
                    "source face count, and stitched FreeCAD face count do not match. The map was written "
                    "in region-generation order and may not match FreeCAD Face indices."
                ),
                station=station,
                source="freecad_cross_sections.face_material_map",
                region_count=len(regions),
                source_face_count=len(source_faces),
                stitched_face_count=len(shape_faces),
            )
        )
        return regions, messages

    source_signatures = [_face_signature(face) for face in source_faces]
    shape_signatures = [_face_signature(face) for face in shape_faces]
    if any(signature is None for signature in source_signatures + shape_signatures):
        messages.append(
            _generation_message(
                "warning",
                "face_signature_unavailable",
                (
                    "FaceMaterialMap face-order verification was skipped because at least one FreeCAD "
                    "face did not expose area and center-of-mass data. The map was written in "
                    "region-generation order and should be checked before assigning materials."
                ),
                station=station,
                source="freecad_cross_sections.face_material_map",
                region_count=len(regions),
                source_face_count=len(source_faces),
                stitched_face_count=len(shape_faces),
            )
        )
        return regions, messages

    unused = set(range(len(source_signatures)))
    ordered = []
    for shape_signature in shape_signatures:
        best_index = min(
            unused,
            key=lambda index: _face_signature_distance(shape_signature, source_signatures[index]),
        )
        unused.remove(best_index)
        ordered.append(regions[best_index])
    return ordered, messages


def _generation_message(severity, code, message, *, station=None, source=None, **details):
    item = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if station is not None:
        item["station"] = station
    if source is not None:
        item["source"] = source
    if details:
        item["details"] = details
    return item


def _shell_laminate_vertex_contact_messages(regions, *, tolerance=1e-8, thickness_ratio=5.0):
    messages = []
    indexed_regions = list(enumerate(regions or []))
    for first_pos, (first_index, first) in enumerate(indexed_regions):
        first_thickness = _region_laminate_thickness(first)
        if first_thickness is None:
            continue
        first_parsed = _parsed_region_name(_region_value(first, "name"))
        if first_parsed.get("side") not in ("HP", "LP") or first_parsed.get("web_index") is not None:
            continue
        if "_to_" in _region_value(first, "name"):
            continue
        first_polygon = _region_boundary_points(first)
        if first_polygon is None:
            continue
        for second_index, second in indexed_regions[first_pos + 1 :]:
            second_thickness = _region_laminate_thickness(second)
            if second_thickness is None:
                continue
            second_parsed = _parsed_region_name(_region_value(second, "name"))
            if second_parsed.get("side") != first_parsed.get("side") or second_parsed.get("web_index") is not None:
                continue
            if "_to_" in _region_value(second, "name"):
                continue
            if "SPAR" not in _region_value(first, "name").upper() and "SPAR" not in _region_value(second, "name").upper():
                continue
            second_polygon = _region_boundary_points(second)
            if second_polygon is None:
                continue
            common_vertices = _common_boundary_vertices(first_polygon, second_polygon, tolerance=tolerance)
            if not common_vertices:
                continue
            if _regions_share_boundary_edge(first_polygon, second_polygon, tolerance=tolerance):
                continue
            if _shell_component_adhesive_covers_vertices(indexed_regions, common_vertices, tolerance=tolerance):
                continue
            thick = max(first_thickness, second_thickness)
            thin = min(first_thickness, second_thickness)
            if thin <= 0 or thick / thin < thickness_ratio:
                continue
            messages.append(
                _generation_message(
                    "warning",
                    "shell_laminate_vertex_contact",
                    (
                        "Two shell laminate regions with a large thickness mismatch meet only at a vertex. "
                        "This can create an unmeshable interface; consider adding an adhesive or transition "
                        "region at this shell component boundary."
                    ),
                    station=first_parsed.get("station"),
                    source="freecad_cross_sections.shell_interfaces",
                    first_face_index=first_index,
                    first_region_name=_region_value(first, "name"),
                    first_material_name=_region_value(first, "material_name"),
                    first_thickness=first_thickness,
                    second_face_index=second_index,
                    second_region_name=_region_value(second, "name"),
                    second_material_name=_region_value(second, "material_name"),
                    second_thickness=second_thickness,
                    common_vertices=[point.tolist() for point in common_vertices],
                )
            )
    return messages


def _shell_component_adhesive_covers_vertices(indexed_regions, vertices, *, tolerance):
    for _, region in indexed_regions:
        name = _region_value(region, "name")
        if "_to_" not in name or "adhesive" not in name.lower():
            continue
        polygon = _region_boundary_points(region)
        if polygon is None:
            continue
        if any(
            any(np.linalg.norm(vertex[:2] - point[:2]) <= tolerance for point in polygon)
            for vertex in vertices
        ):
            return True
    return False


def _region_laminate_thickness(region):
    plies = _region_value(region, "plies") or []
    if not plies:
        return None
    return sum(float(ply.get("thickness", 0.0) or 0.0) for ply in plies)


def _region_boundary_points(region):
    edge_points = _region_value(region, "edge_points")
    if edge_points is not None:
        points = []
        for edge in edge_points:
            points.extend(np.asarray(edge, dtype=float))
        return _clean_boundary_points(points)
    outer_points = _region_value(region, "outer_points")
    inner_points = _region_value(region, "inner_points")
    if outer_points is None or inner_points is None:
        return None
    points = list(np.asarray(outer_points, dtype=float))
    end_connector = _region_value(region, "end_connector")
    if end_connector is not None:
        points.extend(np.asarray(end_connector, dtype=float))
    points.extend(np.flip(np.asarray(inner_points, dtype=float), axis=0))
    start_connector = _region_value(region, "start_connector")
    if start_connector is not None:
        points.extend(np.asarray(start_connector, dtype=float))
    return _clean_boundary_points(points)


def _clean_boundary_points(points, tolerance=1e-12):
    cleaned = []
    for point in points:
        point = np.asarray(point, dtype=float)
        if cleaned and np.linalg.norm(point - cleaned[-1]) <= tolerance:
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tolerance:
        cleaned.pop()
    return np.asarray(cleaned)


def _common_boundary_vertices(first_points, second_points, *, tolerance):
    common = []
    for first in first_points:
        if any(np.linalg.norm(first[:2] - second[:2]) <= tolerance for second in second_points):
            if not any(np.linalg.norm(first[:2] - existing[:2]) <= tolerance for existing in common):
                common.append(first)
    return common


def _regions_share_boundary_edge(first_points, second_points, *, tolerance):
    for first_start, first_end in zip(first_points, np.roll(first_points, -1, axis=0)):
        if np.linalg.norm(first_start[:2] - first_end[:2]) <= tolerance:
            continue
        for second_start, second_end in zip(second_points, np.roll(second_points, -1, axis=0)):
            if np.linalg.norm(second_start[:2] - second_end[:2]) <= tolerance:
                continue
            if (
                np.linalg.norm(first_start[:2] - second_end[:2]) <= tolerance
                and np.linalg.norm(first_end[:2] - second_start[:2]) <= tolerance
            ) or (
                np.linalg.norm(first_start[:2] - second_start[:2]) <= tolerance
                and np.linalg.norm(first_end[:2] - second_end[:2]) <= tolerance
            ):
                return True
    return False


def _set_turbine_messages(metadata_obj, messages):
    warnings = [message for message in messages if message.get("severity") == "warning"]
    errors = [message for message in messages if message.get("severity") == "error"]
    _set_string_property(
        metadata_obj,
        "WarningMessages",
        json.dumps(warnings),
        group="Turbine",
        description="JSON warning messages from pyNuMAD FreeCAD section generation",
    )
    _set_string_property(
        metadata_obj,
        "ErrorMessages",
        json.dumps(errors),
        group="Turbine",
        description="JSON error messages from pyNuMAD FreeCAD section generation",
    )


def _append_turbine_messages(metadata_obj, messages):
    existing = _existing_turbine_messages(metadata_obj)
    _set_turbine_messages(metadata_obj, existing + list(messages or []))


def _existing_turbine_messages(metadata_obj):
    existing = []
    for property_name in ("WarningMessages", "ErrorMessages"):
        try:
            existing.extend(json.loads(getattr(metadata_obj, property_name, "[]") or "[]"))
        except (TypeError, ValueError):
            pass
    return existing


def _face_signature(face):
    try:
        center = getattr(face, "CenterOfMass")
        return (
            float(getattr(face, "Area")),
            float(center.x),
            float(center.y),
            float(center.z),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _face_signature_distance(first, second):
    area_scale = max(abs(first[0]), abs(second[0]), 1.0)
    area_error = abs(first[0] - second[0]) / area_scale
    center_error = sum((first[index] - second[index]) ** 2 for index in range(1, 4)) ** 0.5
    return area_error + center_error


def laminate_definitions(regions):
    """Return unique laminate definitions referenced by face metadata.

    Faces often share identical ply stacks.  This table deduplicates those
    stacks by ply material, angle, and thickness and assigns a compact integer
    index for HomoGen-style face assignments.
    """

    definitions = []
    index_by_key = {}
    for region in regions or []:
        plies = _region_value(region, "plies") or []
        if not plies:
            continue
        key = _laminate_key(plies)
        if key in index_by_key:
            continue
        laminate_index = len(definitions)
        index_by_key[key] = laminate_index
        definitions.append(
            {
                "laminate_index": laminate_index,
                "laminate_name": f"Laminate{laminate_index:03d}",
                "plies": plies,
            }
        )
    return definitions


def global_laminate_definitions(
    blade,
    *,
    geometry_scaling=1.0,
    normalize_chord=False,
):
    """Return turbine-level laminate definitions across every blade station.

    The global table is built from the full ``StackDatabase``, not from the
    subset of stations requested for FreeCAD export.  This keeps HomoGen's
    material model at turbine scope while station ``FaceMaterialMap`` entries
    only carry compact table indices.  With chord-normalized geometry, ply
    thickness is also normalized station-by-station, so physically identical
    laminates can become distinct exported definitions.
    """

    definitions = []
    index_by_key = {}

    def add_plygroup(plygroup, station):
        transformer = _StationTransformer(
            blade,
            station,
            geometry_scaling=geometry_scaling,
            normalize_chord=normalize_chord,
            move_le_to_origin=False,
        )
        plies = _plies_from_plygroup(plygroup, transformer)
        if not plies:
            return
        key = _laminate_key(plies)
        if key in index_by_key:
            return
        laminate_index = len(definitions)
        index_by_key[key] = laminate_index
        definitions.append(
            {
                "laminate_index": laminate_index,
                "laminate_name": f"Laminate{laminate_index:03d}",
                "plies": plies,
            }
        )

    stackdb = getattr(blade, "stackdb", None)
    for stack_array_name in ("stacks", "swstacks"):
        stack_array = getattr(stackdb, stack_array_name, None)
        if stack_array is None:
            continue
        for station in range(stack_array.shape[1]):
            for stack in stack_array[:, station]:
                for plygroup in getattr(stack, "plygroups", []) or []:
                    add_plygroup(plygroup, station)

    return definitions


def material_definitions(blade):
    """Return project-local material property definitions for HomoGen.

    The table uses SI units from pyNuMAD/YAML: density in ``kg/m^3``, elastic
    moduli in ``Pa``, thermal expansion in ``1/K``, thermal conductivity in
    ``W/m/K``, specific heat in ``J/kg/K``, and reference temperature in ``K``.
    Only thermal fields present in the input data are emitted.
    """

    materials = getattr(getattr(blade, "definition", None), "materials", {}) or {}
    material_iter = materials.values() if isinstance(materials, dict) else materials
    definitions = []
    for material_index, material in enumerate(material_iter):
        item = {
            "material_index": material_index,
            "material_name": material.name,
            "material_type": material.type,
            "density": _json_value(material.density),
            "elastic": _elastic_definition(material),
        }

        thermal = _thermal_definition(material)
        if thermal:
            item["thermal"] = thermal

        strength = _strength_definition(material)
        if strength:
            item["strength"] = strength

        fracture = _fracture_definition(material)
        if fracture:
            item["fracture"] = fracture

        definitions.append(item)
    return definitions


def blade_station_count(blade):
    """Return the number of imported blade stations."""

    span = getattr(getattr(blade, "definition", None), "ispan", None)
    return 0 if span is None else len(span)


def yaml_station_count(yaml_path):
    """Return the number of blade stations declared in a WindIO/pyNuMAD YAML.

    This lightweight helper reads only the YAML station arrays, so HomoGen can
    populate a station-range UI before constructing the full pyNuMAD blade
    geometry.  ``outer_shape_bem.reference_axis.z.values`` is the preferred
    source because it defines the spanwise station locations; common fallback
    grids are checked for older or partial files.
    """

    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    try:
        outer_shape = data["components"]["blade"]["outer_shape_bem"]
    except (TypeError, KeyError) as exc:
        raise ValueError("YAML file does not contain components.blade.outer_shape_bem station data") from exc

    candidates = [
        ("outer_shape_bem.reference_axis.z.values", outer_shape.get("reference_axis", {}).get("z", {}).get("values")),
        ("outer_shape_bem.chord.values", outer_shape.get("chord", {}).get("values")),
        ("outer_shape_bem.twist.values", outer_shape.get("twist", {}).get("values")),
        ("outer_shape_bem.reference_axis.z.grid", outer_shape.get("reference_axis", {}).get("z", {}).get("grid")),
        ("outer_shape_bem.chord.grid", outer_shape.get("chord", {}).get("grid")),
    ]
    for _label, values in candidates:
        if values:
            return len(values)
    raise ValueError("YAML file does not define any recognized blade station arrays")


def get_yaml_station_count(yaml_path):
    """Lightweight public API for HomoGen station-range dialogs.

    This is an explicit alias for :func:`yaml_station_count`, provided so UI
    code can ask for the station count before importing/generating FreeCAD
    cross-section geometry.
    """

    return yaml_station_count(yaml_path)


def station_frame_definition(blade, station):
    """Return reference-axis orientation data for one blade station.

    WindIO stores the blade generating line in
    ``outer_shape_bem.reference_axis.x/y/z`` and the section twist in
    ``outer_shape_bem.twist``.  pyNuMAD imports those as sweep/prebend/span and
    twist arrays.  This table keeps the physical reference-axis origin and a
    right-handed local coordinate system so downstream tools can place a 2D
    cross section in the curved/twisted blade frame.

    The frame data always come from ``outer_shape_bem.reference_axis`` and use
    a right-handed convention: ``z_axis`` follows the reference-axis tangent,
    while ``x_axis``/``y_axis`` are twisted about ``z_axis``.  Those constants
    are documented here instead of repeated in every station's JSON output.
    """

    geometry = blade.geometry
    span = np.asarray(blade.definition.ispan, dtype=float)
    origin = np.array(
        [
            -blade.definition.rotorspin * geometry.isweep[station],
            geometry.iprebend[station],
            span[station],
        ],
        dtype=float,
    )
    dx_dz = _station_derivative(span, -blade.definition.rotorspin * geometry.isweep, station)
    dy_dz = _station_derivative(span, geometry.iprebend, station)
    twist_deg = float(geometry.idegreestwist[station])
    basis = _station_lcs_basis(dx_dz, dy_dz, twist_deg, blade.definition.rotorspin)

    return {
        "station": int(station),
        "span": _json_value(span[station]),
        "origin": _json_value(origin),
        "origin_units": "m",
        "reference_axis": {
            "x": _json_value(origin[0]),
            "y": _json_value(origin[1]),
            "z": _json_value(origin[2]),
            "units": "m",
        },
        "rotations": {
            "prebend_angle_deg": _json_value(np.rad2deg(np.arctan2(dy_dz, 1.0))),
            "sweep_angle_deg": _json_value(np.rad2deg(np.arctan2(dx_dz, 1.0))),
            "twist_deg": _json_value(twist_deg),
            "prebend_slope": _json_value(dy_dz),
            "sweep_slope": _json_value(dx_dz),
        },
        "lcs": {
            "origin": _json_value(origin),
            "x_axis": _json_value(basis[:, 0]),
            "y_axis": _json_value(basis[:, 1]),
            "z_axis": _json_value(basis[:, 2]),
        },
    }


def _station_derivative(span, values, station):
    """Return d(values)/d(span) at a station using neighboring stations."""

    values = np.asarray(values, dtype=float)
    if span.size < 2:
        return 0.0
    if station <= 0:
        denominator = span[1] - span[0]
        return 0.0 if denominator == 0 else float((values[1] - values[0]) / denominator)
    if station >= span.size - 1:
        denominator = span[-1] - span[-2]
        return 0.0 if denominator == 0 else float((values[-1] - values[-2]) / denominator)
    denominator = span[station + 1] - span[station - 1]
    return 0.0 if denominator == 0 else float((values[station + 1] - values[station - 1]) / denominator)


def _station_lcs_basis(dx_dz, dy_dz, twist_deg, rotorspin):
    """Build a right-handed station LCS from reference-axis slope and twist."""

    z_axis = _unit(np.array([dx_dz, dy_dz, 1.0], dtype=float))
    x_seed = np.array([1.0, 0.0, 0.0])
    x_axis = x_seed - np.dot(x_seed, z_axis) * z_axis
    if np.linalg.norm(x_axis) <= 1e-12:
        x_seed = np.array([0.0, 1.0, 0.0])
        x_axis = x_seed - np.dot(x_seed, z_axis) * z_axis
    x_axis = _unit(x_axis)
    y_axis = _unit(np.cross(z_axis, x_axis))

    twist = np.deg2rad(-rotorspin * twist_deg)
    x_twisted = np.cos(twist) * x_axis + np.sin(twist) * y_axis
    y_twisted = -np.sin(twist) * x_axis + np.cos(twist) * y_axis
    return np.column_stack((_unit(x_twisted), _unit(y_twisted), z_axis))


def _elastic_definition(material):
    if material.type == "orthotropic":
        return {
            "e1": _json_value(material.ex),
            "e2": _json_value(material.ey),
            "e3": _json_value(material.ez),
            "g12": _json_value(material.gxy),
            "g13": _json_value(material.gxz),
            "g23": _json_value(material.gyz),
            "nu12": _json_value(material.prxy),
            "nu13": _json_value(material.prxz),
            "nu23": _json_value(material.pryz),
        }
    return {
        "youngs_modulus": _json_value(material.ex),
        "shear_modulus": _json_value(material.gxy),
        "poisson_ratio": _json_value(material.prxy),
    }


def _thermal_definition(material):
    fields = {
        "expansion_coefficient": getattr(material, "thermal_expansion", None),
        "conductivity": getattr(material, "thermal_conductivity", None),
        "specific_heat": getattr(material, "specific_heat", None),
        "reference_temperature": getattr(material, "thermal_reference_temperature", None),
    }
    return {
        key: _json_value(value)
        for key, value in fields.items()
        if value is not None
    }


def _strength_definition(material):
    fields = {
        "tensile": material.uts,
        "compressive": _abs_json_value(material.ucs),
        "shear": material.uss,
    }
    return {
        key: _json_value(value)
        for key, value in fields.items()
        if value is not None
    }


def _fracture_definition(material):
    fields = {
        "g1g2": material.g1g2,
        "alp0": material.alp0,
    }
    return {
        key: _json_value(value)
        for key, value in fields.items()
        if value is not None and not _is_nan(value)
    }


class _StationTransformer:
    """Convert station geometry and lengths into the requested output units.

    pyNuMAD stores station geometry in meters and laminate thicknesses in
    millimeters.  The transformer applies the same scaling, optional chord
    normalization, and optional leading-edge translation to both points and
    lengths so all later geometric tolerances are compared in output units.
    """

    def __init__(
        self,
        blade,
        station,
        *,
        geometry_scaling,
        normalize_chord,
        move_le_to_origin,
    ):
        """Store station conversion options and cached chord/LE geometry."""

        self.blade = blade
        self.station = station
        self.geometry_scaling = geometry_scaling
        self.normalize_chord = normalize_chord
        self.move_le_to_origin = move_le_to_origin

        geometry = blade.geometry
        self.chord = geometry.ichord[station]
        self.le = (
            np.array(geometry.coordinates[geometry.LEindex, :, station])
            * geometry_scaling
        )

    def points(self, points):
        """Return input points after scaling, chord normalization, and LE shift."""

        xyz = np.array(points, dtype=float) * self.geometry_scaling
        if self.normalize_chord:
            xyz = xyz / (self.chord * self.geometry_scaling)
        if self.move_le_to_origin:
            xyz = xyz - (self.le / (self.chord * self.geometry_scaling) if self.normalize_chord else self.le)
        return xyz

    def length_from_m(self, length_m):
        """Convert a length stored in meters to the section output units."""

        length = length_m * self.geometry_scaling
        if self.normalize_chord:
            length = length / (self.chord * self.geometry_scaling)
        return length

    def length_from_mm(self, length_mm):
        """Convert a laminate thickness stored in millimeters to output units."""

        return self.length_from_m(0.001 * length_mm)


def _shell_regions(blade, station, section, transformer, cs_params):
    """Build all perimeter shell and trailing-edge adhesive regions.

    The outer airfoil is split by keypoints into six HP and six LP stack
    segments.  Zero-length segments are discarded.  Round or very small TE
    openings are shortened to leave an adhesive gap.  Large flatback openings
    are kept at their input geometry and get a separate flatback adhesive strip,
    matching the Cubit workflow's distinction between round and flatback
    trailing edges.
    """

    stackdb = blade.stackdb
    if stackdb.stacks is None:
        return []

    hp_segments, lp_segments = _shell_segments(blade, station, section, transformer)
    stack_station = min(station, stackdb.stacks.shape[1] - 1)

    stacks, sides, segments = _remove_zero_length_shell_segments(
        list(stackdb.stacks[:6, stack_station]) + list(stackdb.stacks[6:12, stack_station]),
        ["HP"] * 6 + ["LP"] * 6,
        hp_segments + lp_segments,
    )
    flatback_te = _flatback_trailing_edge(section, transformer, cs_params)
    trailing_edge = None
    if flatback_te is None:
        stacks, sides, segments, trailing_edge = _trim_trailing_edge_segments(
            stacks, sides, segments, station, transformer, cs_params
        )
    else:
        stacks, sides, segments, flatback_te = _trim_flatback_trailing_edge_segments(
            stacks,
            sides,
            segments,
            station,
            transformer,
            cs_params,
            flatback_te,
        )
    shell_component_adhesives = _shell_component_adhesive_specs(
        stacks,
        sides,
        segments,
        station,
        transformer,
        cs_params,
    )
    regions = _perimeter_shell_regions(
        stacks,
        sides,
        segments,
        station,
        section,
        transformer,
        shell_component_adhesives=shell_component_adhesives,
        skip_gelcoat_layer=bool(cs_params.get("skip_shell_gelcoat_layer", False)),
    )
    if flatback_te is None:
        regions.extend(_trailing_edge_adhesive_regions(station, trailing_edge, regions, cs_params))
    else:
        regions.extend(_flatback_te_adhesive_regions(station, flatback_te, regions, cs_params))
    return regions


def _flatback_trailing_edge(section, transformer, cs_params):
    """Return flatback TE endpoints when the station has a broad blunt tail.

    Cubit treats stations past ``last_round_station`` as flatbacks instead of
    trimming them like sharp trailing edges.  The FreeCAD path does not receive
    that station classification, so it detects the same geometry locally: HP
    and LP must start at a broad, nearly vertical TE wall around the nominal TE
    midpoint.  The default threshold is ``5%`` of chord in output units.  A
    station-specific ``flatback_te_threshold`` in meters can override it, and
    ``enable_flatback_te=False`` disables this branch.
    """

    if not cs_params.get("enable_flatback_te", True):
        return None
    if len(section.hp_points) == 0 or len(section.lp_points) == 0:
        return None

    hp_outer = section.hp_points[0]
    lp_outer = section.lp_points[0]
    opening = np.linalg.norm(hp_outer - lp_outer)
    requested_threshold = _station_value(
        cs_params.get("flatback_te_threshold"),
        section.station,
        default=0.0,
    )
    threshold = (
        transformer.length_from_m(requested_threshold)
        if requested_threshold > 0
        else transformer.length_from_m(0.05 * transformer.chord)
    )
    if opening <= threshold:
        return None

    midpoint = 0.5 * (hp_outer + lp_outer)
    if np.linalg.norm(section.te_point - midpoint) > 0.15 * opening:
        return None

    return {
        "hp_outer": hp_outer,
        "lp_outer": lp_outer,
        "opening": opening,
    }


def _clamp_le_surface_protrusion(
    hp_points,
    lp_points,
    te_point,
    le_point,
    tolerance=1e-9,
    max_protrusion_fraction=0.002,
):
    """Clamp HP/LP points that numerically protrude past the leading edge.

    Some input station coordinates place the last HP/LP points a tiny distance
    beyond the nominal LE in x.  That can make the LE face self-intersect after
    offsetting.  Only small protrusions are pulled back to the LE x-coordinate.
    Larger protrusions are treated as real rounded-nose geometry; clamping them
    would create an artificial flatfront.  ``1e-9`` is a geometric noise
    tolerance in output units, and ``0.2%`` of the chord-line length is the
    default boundary between numerical cleanup and physical geometry.
    """

    if abs(le_point[0] - te_point[0]) <= tolerance:
        return

    le_is_x_maximum = le_point[0] > te_point[0]
    max_protrusion = max(max_protrusion_fraction * np.linalg.norm(le_point - te_point), tolerance)
    _clamp_trailing_points_to_le_x(
        hp_points,
        le_point[0],
        le_is_x_maximum,
        tolerance,
        max_protrusion,
    )
    _clamp_trailing_points_to_le_x(
        lp_points,
        le_point[0],
        le_is_x_maximum,
        tolerance,
        max_protrusion,
    )
    hp_points[-1] = le_point
    lp_points[-1] = le_point


def _clamp_trailing_points_to_le_x(points, le_x, le_is_x_maximum, tolerance, max_protrusion):
    """Clamp the trailing run of points to the leading-edge x limit."""

    protrusions = points[:, 0] - le_x if le_is_x_maximum else le_x - points[:, 0]
    if np.max(protrusions) > max_protrusion:
        return

    for i_point in reversed(range(len(points))):
        excess = points[i_point, 0] - le_x if le_is_x_maximum else le_x - points[i_point, 0]
        if excess <= tolerance:
            if i_point != len(points) - 1:
                break
            continue
        points[i_point, 0] = le_x


def _shell_segments(blade, station, section, transformer):
    """Split HP and LP airfoil curves into stack segments using keypoints.

    HP segments are ordered from trailing edge to leading edge.  LP segments are
    ordered from leading edge to trailing edge so the combined shell path walks
    continuously around the perimeter.
    """

    keypoints = transformer.points(blade.keypoints.key_points[:, :, station])
    hp_boundaries = np.vstack((section.hp_points[0], keypoints[0:5], section.hp_points[-1]))
    lp_boundaries = np.vstack((section.lp_points[-1], keypoints[5:10], section.lp_points[0]))

    hp_segments = _split_polyline_at_points(section.hp_points, hp_boundaries)
    lp_segments = _split_polyline_at_points(np.flip(section.lp_points, axis=0), lp_boundaries)
    return hp_segments, lp_segments


def _remove_zero_length_shell_segments(stacks, sides, segments):
    """Drop stack segments whose curve length is effectively zero.

    A ``1e-9`` length tolerance prevents degenerate faces when keypoints collapse
    together at small or highly tapered stations.
    """

    filtered = [
        (stack, side, _clean_polyline(segment))
        for stack, side, segment in zip(stacks, sides, segments)
        if _polyline_lengths(segment)[-1] > 1e-9
    ]
    if not filtered:
        return [], [], []
    filtered_stacks, filtered_sides, filtered_segments = zip(*filtered)
    return list(filtered_stacks), list(filtered_sides), list(filtered_segments)


def _trim_trailing_edge_segments(stacks, sides, segments, station, transformer, cs_params):
    """Trim HP/LP shell paths at the TE and return the removed adhesive edges.

    The detailed model intentionally does not let HP and LP laminates meet
    directly at the trailing edge.  This function removes equal path distance
    from the HP start and LP end, possibly across more than one stack segment,
    so a separate TE adhesive face can close the section.  If the split width is
    too small, or either side cannot be trimmed, the input segments are returned
    unchanged and no TE adhesive is generated.
    """

    if len(segments) < 2:
        return stacks, sides, segments, None

    hp_count = 0
    while hp_count < len(sides) and sides[hp_count] == "HP":
        hp_count += 1
    lp_start = len(sides)
    while lp_start > 0 and sides[lp_start - 1] == "LP":
        lp_start -= 1
    if hp_count == 0 or lp_start == len(sides):
        return stacks, sides, segments, None

    requested_width = _station_value(cs_params.get("te_adhesive_width"), station, default=0.0)
    split_width = (
        transformer.length_from_m(requested_width)
        if requested_width > 0
        else _trailing_edge_split_width(
            stacks[:hp_count],
            segments[:hp_count],
            stacks[lp_start:],
            segments[lp_start:],
            transformer,
            cs_params,
            station,
        )
    )
    if split_width <= 1e-9:
        return stacks, sides, segments, None

    hp_stacks, hp_sides, hp_segments, hp_outer = _trim_segments_from_start(
        stacks[:hp_count], sides[:hp_count], segments[:hp_count], split_width
    )
    lp_stacks, lp_sides, lp_segments, lp_outer = _trim_segments_from_end(
        stacks[lp_start:], sides[lp_start:], segments[lp_start:], split_width
    )
    middle_stacks = stacks[hp_count:lp_start]
    middle_sides = sides[hp_count:lp_start]
    middle_segments = segments[hp_count:lp_start]

    if hp_outer is None or lp_outer is None:
        return stacks, sides, segments, None

    trailing_edge = {
        "hp_outer": hp_outer,
        "lp_outer": lp_outer,
    }

    return (
        hp_stacks + middle_stacks + lp_stacks,
        hp_sides + middle_sides + lp_sides,
        hp_segments + middle_segments + lp_segments,
        trailing_edge,
    )


def _trim_flatback_trailing_edge_segments(
    stacks,
    sides,
    segments,
    station,
    transformer,
    cs_params,
    flatback_te,
):
    """Trim shell ends near a flatback wall and retain removed adhesive edges.

    The flatback wall itself remains in the adhesive face.  The HP and LP shell
    ends are moved a short distance away from that wall so the adhesive can
    share the shell cut connectors instead of overlapping the first shell
    elements at the flatback corners.
    """

    if len(segments) < 2:
        return stacks, sides, segments, flatback_te

    hp_count = 0
    while hp_count < len(sides) and sides[hp_count] == "HP":
        hp_count += 1
    lp_start = len(sides)
    while lp_start > 0 and sides[lp_start - 1] == "LP":
        lp_start -= 1
    if hp_count == 0 or lp_start == len(sides):
        return stacks, sides, segments, flatback_te

    trim_width = _flatback_trailing_edge_trim_width(
        stacks[:hp_count],
        segments[:hp_count],
        stacks[lp_start:],
        segments[lp_start:],
        transformer,
        cs_params,
        station,
    )
    if trim_width <= 1e-9:
        return stacks, sides, segments, flatback_te

    hp_stacks, hp_sides, hp_segments, hp_outer = _trim_segments_from_start(
        stacks[:hp_count], sides[:hp_count], segments[:hp_count], trim_width
    )
    lp_stacks, lp_sides, lp_segments, lp_outer = _trim_segments_from_end(
        stacks[lp_start:], sides[lp_start:], segments[lp_start:], trim_width
    )
    if hp_outer is None or lp_outer is None:
        return stacks, sides, segments, flatback_te

    flatback_te = dict(flatback_te)
    flatback_te["hp_outer"] = hp_outer
    flatback_te["lp_outer"] = lp_outer

    middle_stacks = stacks[hp_count:lp_start]
    middle_sides = sides[hp_count:lp_start]
    middle_segments = segments[hp_count:lp_start]
    return (
        hp_stacks + middle_stacks + lp_stacks,
        hp_sides + middle_sides + lp_sides,
        hp_segments + middle_segments + lp_segments,
        flatback_te,
    )


def _flatback_trailing_edge_trim_width(
    hp_stacks,
    hp_segments,
    lp_stacks,
    lp_segments,
    transformer,
    cs_params,
    station,
):
    """Return shell path distance reserved for flatback adhesive."""

    hp_total = sum(_polyline_lengths(segment)[-1] for segment in hp_segments)
    lp_total = sum(_polyline_lengths(segment)[-1] for segment in lp_segments)
    max_width = 0.9 * min(hp_total, lp_total)
    if max_width <= 1e-9:
        return 0.0

    requested_width = _station_value(
        cs_params.get(
            "flatback_te_adhesive_width",
            cs_params.get("flatback_te_adhesive_depth"),
        ),
        station,
        default=0.0,
    )
    if requested_width > 0:
        return min(transformer.length_from_m(requested_width), max_width)

    hp_thickness = transformer.length_from_mm(sum(hp_stacks[0].layer_thicknesses()))
    lp_thickness = transformer.length_from_mm(sum(lp_stacks[-1].layer_thicknesses()))
    thickness_width = max(hp_thickness, lp_thickness)
    chord_width = transformer.length_from_m(0.005 * transformer.chord)
    return min(max(thickness_width, chord_width), max_width)


def _trailing_edge_split_width(hp_stacks, hp_segments, lp_stacks, lp_segments, transformer, cs_params, station):
    """Find the path distance to remove from each TE side.

    If ``cs_params["te_adhesive_width"]`` is not supplied, the split width is
    chosen by solving for a target HP/LP gap.  A bisection search over path
    distance is robust for curved and tapered stations.  The upper bound is
    ``90%`` of the shorter available TE path so trimming cannot consume an
    entire side.
    """

    hp_lengths = [_polyline_lengths(segment)[-1] for segment in hp_segments]
    lp_lengths = [_polyline_lengths(segment)[-1] for segment in lp_segments]
    hp_total = sum(hp_lengths)
    lp_total = sum(lp_lengths)
    max_width = 0.9 * min(hp_total, lp_total)
    if max_width <= 1e-9:
        return 0.0

    def gap_at(width):
        hp_point = _point_at_path_distance(hp_segments, width)
        lp_point = _point_at_reversed_path_distance(lp_segments, width)
        return np.linalg.norm(hp_point - lp_point)

    initial_gap = gap_at(0.0)
    target_gap = _trailing_edge_target_gap(
        hp_stacks[0],
        lp_stacks[-1],
        transformer,
        cs_params,
        station,
        initial_gap,
    )
    local_width = min(hp_lengths[0], lp_lengths[-1])
    min_width = min(0.02 * local_width, max_width)

    if gap_at(0.0) >= target_gap:
        return min_width
    if gap_at(max_width) <= target_gap:
        return max_width

    low = 0.0
    high = max_width
    for _ in range(32):
        mid = 0.5 * (low + high)
        if gap_at(mid) < target_gap:
            low = mid
        else:
            high = mid
    return max(high, min_width)


def _trim_segments_from_start(stacks, sides, segments, distance):
    """Remove ``distance`` along the beginning of a segmented path.

    Returns kept stacks/sides/segments plus the removed edge path.  Whole
    segments are removed when necessary; the first partially kept segment is
    split by arc length.
    """

    remaining = distance
    kept_stacks = []
    kept_sides = []
    kept_segments = []
    removed_parts = []
    trimming = True
    for stack, side, segment in zip(stacks, sides, segments):
        if not trimming:
            kept_stacks.append(stack)
            kept_sides.append(side)
            kept_segments.append(segment)
            continue

        length = _polyline_lengths(segment)[-1]
        if remaining >= length - 1e-9:
            removed_parts.append(segment)
            remaining -= length
            continue

        removed_parts.append(_polyline_between(segment, 0.0, remaining))
        rest = _polyline_between(segment, remaining, length)
        kept_stacks.append(stack)
        kept_sides.append(side)
        kept_segments.append(rest)
        trimming = False

    return kept_stacks, kept_sides, kept_segments, _join_connected_edges(removed_parts) if removed_parts else None


def _trim_segments_from_end(stacks, sides, segments, distance):
    """Remove ``distance`` along the end of a segmented path.

    This is the mirror of :func:`_trim_segments_from_start`; removed parts are
    returned in geometric order so they can become one adhesive boundary.
    """

    remaining = distance
    kept = [(stack, side, segment) for stack, side, segment in zip(stacks, sides, segments)]
    removed_parts = []
    for i_segment in reversed(range(len(kept))):
        stack, side, segment = kept[i_segment]
        length = _polyline_lengths(segment)[-1]
        if remaining >= length - 1e-9:
            removed_parts.insert(0, segment)
            kept.pop(i_segment)
            remaining -= length
            continue

        removed_parts.insert(0, _polyline_between(segment, length - remaining, length))
        kept[i_segment] = (stack, side, _polyline_between(segment, 0.0, length - remaining))
        break

    if kept:
        kept_stacks, kept_sides, kept_segments = zip(*kept)
        kept_stacks, kept_sides, kept_segments = list(kept_stacks), list(kept_sides), list(kept_segments)
    else:
        kept_stacks, kept_sides, kept_segments = [], [], []
    return kept_stacks, kept_sides, kept_segments, _join_connected_edges(removed_parts) if removed_parts else None


def _shell_component_adhesive_specs(stacks, sides, segments, station, transformer, cs_params):
    """Return optional adhesive shell inserts at spar/component boundaries."""

    width = transformer.length_from_m(
        _station_value(cs_params.get("shell_component_adhesive_width"), station, default=0.0)
    )
    if width <= 0:
        return []

    material_name = cs_params.get("shell_component_adhesive_mat_name", cs_params.get("adhesive_mat_name", "Adhesive"))
    adhesive_specs = []
    i_segment = 0
    while i_segment < len(stacks):
        stack = stacks[i_segment]
        side = sides[i_segment]
        segment = segments[i_segment]
        if (
            i_segment < len(stacks) - 1
            and sides[i_segment + 1] == side
            and _is_spar_component_boundary(stack, stacks[i_segment + 1])
            and _polyline_lengths(segment)[-1] > 1e-9
            and _polyline_lengths(segments[i_segment + 1])[-1] > 1e-9
        ):
            next_stack = stacks[i_segment + 1]
            next_segment = segments[i_segment + 1]
            common_layers = _common_outer_plygroups(stack, next_stack)
            if common_layers and len(common_layers) < max(len(stack.plygroups), len(next_stack.plygroups)):
                adhesive_specs.append(
                    {
                        "side": side,
                        "first_stack_name": stack.name,
                        "second_stack_name": next_stack.name,
                        "insert_layer": len(common_layers),
                        "width": width,
                        "material_name": material_name,
                    }
                )
        i_segment += 1

    return adhesive_specs


def _apply_shell_component_adhesives_for_layer(stacks, sides, current_segments, i_layer, adhesive_specs):
    """Insert shell adhesive segments at the current laminate depth."""

    new_stacks = []
    new_sides = []
    new_segments = []
    i_segment = 0
    while i_segment < len(stacks):
        stack = stacks[i_segment]
        side = sides[i_segment]
        segment = current_segments[i_segment]
        spec = None
        if i_segment < len(stacks) - 1:
            next_stack = stacks[i_segment + 1]
            for candidate in adhesive_specs:
                if (
                    candidate["insert_layer"] == i_layer
                    and candidate["side"] == side
                    and candidate["first_stack_name"] == stack.name
                    and candidate["second_stack_name"] == next_stack.name
                    and sides[i_segment + 1] == side
                ):
                    spec = candidate
                    break

        if spec is not None:
            next_stack = stacks[i_segment + 1]
            next_segment = current_segments[i_segment + 1]
            left_length = _polyline_lengths(segment)[-1]
            right_length = _polyline_lengths(next_segment)[-1]
            stack_is_spar = "SPAR" in stack.name.upper()
            next_stack_is_spar = "SPAR" in next_stack.name.upper()
            left_trim = min(spec["width"], 0.45 * left_length) if next_stack_is_spar else 0.0
            right_trim = min(spec["width"], 0.45 * right_length) if stack_is_spar else 0.0
            if left_trim > 1e-9 or right_trim > 1e-9:
                kept_left = _polyline_between(segment, 0.0, left_length - left_trim)
                kept_right = _polyline_between(next_segment, right_trim, right_length)
                if next_stack_is_spar:
                    adhesive_segment = _clean_polyline(np.vstack((kept_left[-1], segment[-1])))
                else:
                    adhesive_segment = _clean_polyline(np.vstack((segment[-1], kept_right[0])))
                new_stacks.append(stack)
                new_sides.append(side)
                new_segments.append(kept_left)
                new_stacks.append(_shell_component_adhesive_stack(stack, next_stack, spec["material_name"]))
                new_sides.append(side)
                new_segments.append(adhesive_segment)
                current_segments[i_segment + 1] = kept_right
                i_segment += 1
                continue

        new_stacks.append(stack)
        new_sides.append(side)
        new_segments.append(segment)
        i_segment += 1

    return new_stacks, new_sides, new_segments


def _is_spar_component_boundary(first_stack, second_stack):
    first_is_spar = "SPAR" in first_stack.name.upper()
    second_is_spar = "SPAR" in second_stack.name.upper()
    return first_is_spar != second_is_spar


def _shell_component_adhesive_stack(first_stack, second_stack, material_name):
    stack = Stack()
    stack.name = f"{first_stack.name}_to_{second_stack.name}_adhesive"
    common_layers = _common_outer_plygroups(first_stack, second_stack)
    first_remaining = sum(first_stack.layer_thicknesses()[len(common_layers) :])
    second_remaining = sum(second_stack.layer_thicknesses()[len(common_layers) :])
    stack.plygroups = common_layers + [
        Ply(
            component=stack.name,
            materialid=material_name,
            thickness=max(first_remaining, second_remaining),
            angle=0.0,
            nPlies=1,
        )
    ]
    return stack


def _common_outer_plygroups(first_stack, second_stack, tolerance=1e-12):
    common = []
    for first, second in zip(first_stack.plygroups, second_stack.plygroups):
        first_thickness = first.nPlies * first.thickness
        second_thickness = second.nPlies * second.thickness
        if (
            first.materialid != second.materialid
            or abs(first_thickness - second_thickness) > tolerance
            or abs(float(first.angle or 0.0) - float(second.angle or 0.0)) > tolerance
        ):
            break
        common.append(
            Ply(
                component=_shell_component_bridge_marker(first_stack, second_stack),
                materialid=first.materialid,
                thickness=first.thickness,
                angle=first.angle,
                nPlies=first.nPlies,
            )
        )
    return common


def _shell_component_bridge_marker(first_stack, second_stack):
    return f"{first_stack.name}_to_{second_stack.name}_bridge"


def _is_shell_component_bridge_plygroup(plygroup):
    return str(getattr(plygroup, "component", "")).endswith("_bridge")


def _point_at_path_distance(segments, distance):
    """Return a point at arc length ``distance`` across connected segments."""

    remaining = distance
    for segment in segments:
        length = _polyline_lengths(segment)[-1]
        if remaining <= length:
            return _point_at_distance(segment, remaining)
        remaining -= length
    return segments[-1][-1]


def _point_at_reversed_path_distance(segments, distance):
    """Return a point measured backward from the end of connected segments."""

    remaining = distance
    for segment in reversed(segments):
        length = _polyline_lengths(segment)[-1]
        if remaining <= length:
            return _point_at_distance(segment, length - remaining)
        remaining -= length
    return segments[0][0]


def _trailing_edge_target_gap(hp_stack, lp_stack, transformer, cs_params, station, initial_gap):
    """Return the desired HP/LP opening at the TE trim location.

    A station-specific ``te_adhesive_gap`` overrides the heuristic.  Otherwise
    the gap is based on local combined laminate thickness, lightly increased
    from the initial TE opening, and bounded between chord-relative limits.  The
    formula favors robustness for meshing by avoiding both overlaps and tiny
    adhesive edges.
    """

    requested_gap = _station_value(cs_params.get("te_adhesive_gap"), station, default=0.0)
    if requested_gap > 0:
        return transformer.length_from_m(requested_gap)

    hp_thickness = transformer.length_from_mm(sum(hp_stack.layer_thicknesses()))
    lp_thickness = transformer.length_from_mm(sum(lp_stack.layer_thicknesses()))
    combined_thickness = hp_thickness + lp_thickness
    thickness_gap = 1.35 * combined_thickness
    opening_limited_gap = min(1.75 * combined_thickness, initial_gap + transformer.length_from_m(0.008 * transformer.chord))
    chord_gap = transformer.length_from_m(0.01 * transformer.chord)
    max_gap = transformer.length_from_m(0.06 * transformer.chord)
    return min(max(thickness_gap, opening_limited_gap, chord_gap), max_gap)


def _shell_regions_from_stack(stack, station, side, outer_points, section, transformer):
    """Build ply faces for one isolated shell stack segment.

    This simpler path is used when a paired LE treatment cannot be applied.  It
    repeatedly offsets the current outer curve inward by each plygroup
    thickness and returns strip regions between successive curves.
    """

    if len(outer_points) < 2 or _polyline_lengths(outer_points)[-1] <= 1e-9:
        return []

    regions = []
    current_outer = _clean_polyline(outer_points)

    for i_layer, plygroup in enumerate(stack.plygroups):
        thickness = transformer.length_from_mm(plygroup.nPlies * plygroup.thickness)
        if thickness <= 0:
            continue

        inner_points = _clean_polyline(_offset_open_polyline_inward(current_outer, section.closed_points, thickness))
        regions.append(
            FreeCADFaceRegion(
                name=f"Station{station:03d}_{side}_{stack.name}_layer{i_layer:02d}",
                material_name=plygroup.materialid,
                ply_angle=plygroup.angle,
                laminate_name=_laminate_name(station, side, stack.name, i_layer),
                plies=_plies_from_plygroup(plygroup, transformer),
                outer_points=current_outer,
                inner_points=inner_points,
            )
        )
        current_outer = inner_points

    if regions:
        return regions

    thickness = transformer.length_from_mm(sum(stack.layer_thicknesses()))
    if thickness <= 0:
        return []

    return [
        FreeCADFaceRegion(
            name=f"Station{station:03d}_{side}_{stack.name}",
            material_name=stack.name,
            ply_angle=0,
            outer_points=outer_points,
            inner_points=_offset_open_polyline_inward(outer_points, section.closed_points, thickness),
        )
    ]


def _perimeter_shell_regions(
    stacks,
    sides,
    segments,
    station,
    section,
    transformer,
    shell_component_adhesives=None,
    skip_gelcoat_layer=False,
):
    """Expand perimeter stack segments into layer-by-layer shell face regions.

    For each ply layer, all current outer segments are combined into one path so
    offsets remain consistent across stack boundaries.  Segment slices map the
    combined path back to individual stack regions.  Boundaries are squared when
    adjacent stacks have different thicknesses so the resulting faces are easier
    to mesh and do not contain sliver-like diagonal closures.
    """

    shell_component_adhesives = shell_component_adhesives or []
    current_segments = [_clean_polyline(segment) for segment in segments]
    stack_name_counts = {
        (side, stack.name): sum(1 for other_side, other_stack in zip(sides, stacks) if other_side == side and other_stack.name == stack.name)
        for side, stack in zip(sides, stacks)
    }
    max_layers = max((len(stack.plygroups) for stack in stacks), default=0)
    max_layers = max(max_layers, max((spec["insert_layer"] + 1 for spec in shell_component_adhesives), default=0))
    regions = []

    for i_layer in range(max_layers):
        if any(spec["insert_layer"] == i_layer for spec in shell_component_adhesives):
            stacks, sides, current_segments = _apply_shell_component_adhesives_for_layer(
                stacks,
                sides,
                current_segments,
                i_layer,
                shell_component_adhesives,
            )
            stack_name_counts = {
                (side, stack.name): sum(1 for other_side, other_stack in zip(sides, stacks) if other_side == side and other_stack.name == stack.name)
                for side, stack in zip(sides, stacks)
            }
            max_layers = max(max_layers, max((len(stack.plygroups) for stack in stacks), default=0))

        combined_outer, segment_slices = _combine_connected_segments(current_segments)
        thicknesses = [
            transformer.length_from_mm(stack.plygroups[i_layer].nPlies * stack.plygroups[i_layer].thickness)
            if i_layer < len(stack.plygroups)
            else 0.0
            for stack in stacks
        ]
        offset_curves = _offset_curves_by_thickness(combined_outer, section.closed_points, thicknesses)
        _close_matching_curve_endpoints(offset_curves)
        _square_disconnected_segment_boundaries(offset_curves, segment_slices, current_segments)
        _square_stair_step_boundaries(
            offset_curves,
            thicknesses,
            segment_slices,
            current_segments,
            _is_closed_polyline(combined_outer),
        )

        for i_segment, stack in enumerate(stacks):
            if i_layer >= len(stack.plygroups) or thicknesses[i_segment] <= 0:
                continue
            start, end = segment_slices[i_segment]
            thickness = thicknesses[i_segment]
            inner_segment = _clean_polyline(offset_curves[thickness][start:end])
            outer_segment = current_segments[i_segment]
            _square_stair_step_endpoint(
                inner_segment,
                outer_segment,
                offset_curves,
                thicknesses,
                thickness,
                i_segment,
                start,
                "start",
                _is_closed_polyline(combined_outer),
            )
            _square_stair_step_endpoint(
                inner_segment,
                outer_segment,
                offset_curves,
                thicknesses,
                thickness,
                i_segment,
                end - 1,
                "end",
                _is_closed_polyline(combined_outer),
            )
            if stack_name_counts[(sides[i_segment], stack.name)] > 1 and (i_segment == 0 or i_segment == len(stacks) - 1):
                outer_segment = np.vstack((outer_segment[0], outer_segment[-1]))
                inner_segment = np.vstack((inner_segment[0], inner_segment[-1]))
            if not _valid_face_boundary(outer_segment, inner_segment):
                continue

            plygroup = stack.plygroups[i_layer]
            start_connector, end_connector = _stair_step_connectors(
                offset_curves,
                thicknesses,
                i_segment,
                start,
                end - 1,
                inner_segment[0],
                inner_segment[-1],
                _is_closed_polyline(combined_outer),
            )
            if (
                not _is_shell_component_bridge_plygroup(plygroup)
                and not _skip_shell_plygroup_face(plygroup, i_layer, skip_gelcoat_layer)
            ):
                regions.append(
                    FreeCADFaceRegion(
                        name=f"Station{station:03d}_{sides[i_segment]}_{_shell_region_stack_name(stack, sides, stack_name_counts, i_segment)}_layer{i_layer:02d}",
                        material_name=plygroup.materialid,
                        ply_angle=plygroup.angle,
                        laminate_name=_laminate_name(
                            station,
                            sides[i_segment],
                            _shell_region_stack_name(stack, sides, stack_name_counts, i_segment),
                            i_layer,
                        ),
                        plies=_plies_from_plygroup(plygroup, transformer),
                        outer_points=outer_segment,
                        inner_points=inner_segment,
                        start_connector=start_connector,
                        end_connector=end_connector,
                    )
                )
            current_segments[i_segment] = inner_segment

    return regions


def _skip_shell_plygroup_face(plygroup, i_layer, skip_gelcoat_layer):
    """Return whether a shell plygroup should be offset but not emitted."""

    return skip_gelcoat_layer and i_layer == 0 and _is_gelcoat_plygroup(plygroup)


def _is_gelcoat_plygroup(plygroup):
    """Return whether a plygroup looks like a gelcoat/coating layer."""

    text = " ".join(
        str(value).lower()
        for value in (
            getattr(plygroup, "materialid", ""),
            getattr(plygroup, "component", ""),
            getattr(plygroup, "name", ""),
        )
    )
    return any(marker in text for marker in ("gelcoat", "gel coat", "coating", "coat"))


def _shell_region_stack_name(stack, sides, stack_name_counts, i_segment):
    """Return a stable region stack name, disambiguating repeated TE stacks."""

    name = stack.name
    if stack_name_counts[(sides[i_segment], stack.name)] <= 1:
        return name
    if i_segment == 0 or i_segment == len(sides) - 1:
        return name + "_TE"
    return name


def _trailing_edge_adhesive_regions(station, trailing_edge, shell_regions, cs_params):
    """Create the adhesive face that closes the trimmed trailing edge.

    The face boundary uses the removed HP/LP outer curves, the laminate cut
    connectors through all shell layers, and a final outer TE cap.  If the shell
    cut connectors cannot be recovered, no adhesive is emitted rather than
    creating an invalid face.
    """

    if trailing_edge is None:
        return []

    hp_outer = _clean_polyline(trailing_edge["hp_outer"])
    lp_outer = _clean_polyline(trailing_edge["lp_outer"])
    if _polyline_lengths(hp_outer)[-1] <= 1e-9 or _polyline_lengths(lp_outer)[-1] <= 1e-9:
        return []

    hp_connector = _shell_cut_connector(shell_regions, hp_outer[-1], "start", side="HP")
    lp_connector = _shell_cut_connector(shell_regions, lp_outer[0], "end", side="LP")
    if hp_connector is None or lp_connector is None:
        return []

    if len(hp_connector) < 2 or len(lp_connector) < 2:
        return []

    edge_points = _trailing_edge_adhesive_edges(hp_outer, hp_connector, lp_outer, lp_connector)
    if edge_points is None:
        return []

    return [
        FreeCADFaceRegion(
            name=f"Station{station:03d}_TE_adhesive",
            material_name=cs_params.get("adhesive_mat_name", "Adhesive"),
            ply_angle=0.0,
            edge_points=edge_points,
            edge_kinds=["spline", "line", "line", "line", "spline", "line"],
        )
    ]


def _trailing_edge_adhesive_edges(hp_outer, hp_connector, lp_outer, lp_connector):
    """Choose a non-self-intersecting ordered boundary for the TE adhesive.

    Connector depth is tried from innermost to outermost.  This lets the
    adhesive close through as many laminate layers as possible while falling
    back gracefully if a deeper boundary would cross itself at a difficult
    station.
    """

    max_depth = min(len(hp_connector), len(lp_connector))
    for depth in reversed(range(2, max_depth + 1)):
        hp_cut = hp_connector[:depth]
        lp_cut = lp_connector[:depth]
        inner_bridge = np.vstack((hp_cut[-1], lp_cut[-1]))
        if np.linalg.norm(inner_bridge[0] - inner_bridge[-1]) <= 1e-9:
            continue
        trailing_edge_cap = np.vstack((lp_outer[-1], hp_outer[0]))
        edge_points = [hp_outer, hp_cut, inner_bridge, np.flip(lp_cut, axis=0), lp_outer, trailing_edge_cap]
        if not _edge_boundary_self_intersects(edge_points):
            return edge_points
    return None


def _flatback_te_adhesive_regions(station, flatback_te, shell_regions, cs_params):
    """Create the adhesive face behind a preserved flatback TE wall.

    For flatback sections the OML already contains a physical trailing-edge
    wall.  The shell ends are trimmed a short distance away from that wall, and
    this face fills the transition from the flatback wall to the shell cut
    connectors.  That gives the adhesive and shell regions shared edges at the
    HP/LP corners instead of overlapping faces.
    """

    hp_outer = flatback_te["hp_outer"]
    lp_outer = flatback_te["lp_outer"]
    hp_connector = _shell_cut_connector(shell_regions, hp_outer[-1], "start", side="HP")
    lp_connector = _shell_cut_connector(shell_regions, lp_outer[0], "end", side="LP")
    if hp_connector is None or lp_connector is None:
        return []
    face_edges = _trailing_edge_adhesive_edges(hp_outer, hp_connector, lp_outer, lp_connector)
    if face_edges is None:
        return []

    return [
        FreeCADFaceRegion(
            name=f"Station{station:03d}_flatTEadhesive",
            material_name=cs_params.get("adhesive_mat_name", "Adhesive"),
            ply_angle=0.0,
            edge_points=face_edges,
            edge_kinds=["spline", "line", "line", "line", "spline", "line"],
        )
    ]


def _shell_cut_connector(shell_regions, outer_point, connector_end, side=None, tolerance=1e-8, skipped_outer_tolerance=1e-3):
    """Collect a through-thickness connector at a trimmed shell end.

    Regions are sorted by layer index and chained from the supplied outer point
    toward the innermost layer.  Points that do not match within ``1e-8`` output
    units are skipped, which keeps unrelated or numerically disconnected layers
    out of the adhesive boundary.  If a thin gelcoat face is intentionally
    omitted, the first emitted shell layer begins just inward of ``outer_point``;
    that small initial gap is accepted so TE adhesive still closes through the
    skipped thickness.
    """

    candidates_by_layer = []
    for region in shell_regions:
        if region.outer_points is None or region.inner_points is None:
            continue
        if side is not None and f"_{side}_" not in region.name:
            continue
        connector = region.start_connector if connector_end == "start" else region.end_connector
        if connector is None:
            continue
        layer = _layer_index_from_name(region.name)
        candidates_by_layer.append((layer if layer is not None else 0, region))

    if not candidates_by_layer:
        return None

    candidates_by_layer.sort(key=lambda item: item[0])
    points = [outer_point]
    for _, region in candidates_by_layer:
        region_outer_point = region.outer_points[0] if connector_end == "start" else region.outer_points[-1]
        gap = np.linalg.norm(region_outer_point - points[-1])
        if gap > tolerance:
            if len(points) == 1 and gap <= skipped_outer_tolerance:
                points.append(region_outer_point)
            else:
                continue
        connector = region.start_connector if connector_end == "start" else region.end_connector
        ordered = np.flip(connector, axis=0) if connector_end == "start" else connector
        if np.linalg.norm(ordered[0] - points[-1]) > tolerance:
            ordered = np.flip(ordered, axis=0)
        points.extend(ordered[1:])
    return np.array(points)


def _combine_connected_segments(segments):
    """Concatenate adjacent shell segments and record each segment slice."""

    combined = []
    segment_slices = []
    for segment in segments:
        if not combined:
            start = 0
            combined.extend(segment)
        elif np.linalg.norm(np.asarray(combined[-1]) - segment[0]) <= 1e-9:
            start = len(combined) - 1
            combined.extend(segment[1:])
        else:
            start = len(combined)
            combined.extend(segment)
        end = len(combined)
        segment_slices.append((start, end))
    return np.array(combined), segment_slices


def _offset_curves_by_thickness(points, closed_points, thicknesses):
    """Create inward offsets of a combined perimeter path for each thickness."""

    offset_curves = {0.0: points}
    for thickness in sorted(set(thicknesses)):
        if thickness > 0:
            offset_curves[thickness] = _offset_open_polyline_inward(points, closed_points, thickness)
    return offset_curves


def _close_matching_curve_endpoints(offset_curves):
    """Force matching start/end points on offset curves for closed paths."""

    if not _is_closed_polyline(offset_curves[0.0]):
        return
    for thickness, points in offset_curves.items():
        if thickness <= 0:
            continue
        intersection = _curve_end_intersection(points, "start", points, "end")
        points[0] = intersection
        points[-1] = intersection


def _is_closed_polyline(points, tolerance=1e-9):
    """Return whether a polyline's endpoints are coincident within tolerance."""

    return len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= tolerance


def _curve_end_intersection(first_points, first_end, second_points, second_end):
    """Intersect endpoint tangent lines, with a local midpoint fallback.

    If the tangent lines are parallel or their intersection lies too far from
    the local endpoints, the midpoint is used.  The ``0.75 * local_length`` bound
    avoids long miter spikes at sharp or noisy endpoints.
    """

    first_start, first_next = _endpoint_tangent_points(first_points, first_end)
    second_start, second_next = _endpoint_tangent_points(second_points, second_end)
    fallback = 0.5 * (first_start + second_start)
    intersection = _line_intersection_2d(first_start, first_next, second_start, second_next)
    if intersection is None:
        return fallback
    local_length = max(
        np.linalg.norm(first_next - first_start),
        np.linalg.norm(second_next - second_start),
        np.linalg.norm(first_start - second_start),
        1e-9,
    )
    if np.linalg.norm(intersection - fallback) > 0.75 * local_length:
        return fallback
    return intersection


def _endpoint_tangent_points(points, end):
    """Return endpoint and adjacent point used to define an endpoint tangent."""

    if end == "start":
        return points[0], points[1]
    if end == "end":
        return points[-1], points[-2]
    raise ValueError(f"Unknown curve end: {end}")


def _square_stair_step_boundaries(offset_curves, thicknesses, segment_slices, current_segments, closed=False):
    """Adjust offset endpoints at stack boundaries before regions are sliced."""

    for i_segment, (start, end) in enumerate(segment_slices):
        thickness = thicknesses[i_segment]
        if thickness <= 0:
            continue
        _square_stair_step_boundary(
            offset_curves,
            thicknesses,
            current_segments[i_segment],
            thickness,
            i_segment,
            start,
            "start",
            closed,
        )
        _square_stair_step_boundary(
            offset_curves,
            thicknesses,
            current_segments[i_segment],
            thickness,
            i_segment,
            end - 1,
            "end",
            closed,
        )


def _square_disconnected_segment_boundaries(offset_curves, segment_slices, current_segments, tolerance=1e-9):
    """Square offsets at gaps between non-connected neighboring segments."""

    for i_segment, (start, end) in enumerate(segment_slices):
        if i_segment > 0 and np.linalg.norm(current_segments[i_segment - 1][-1] - current_segments[i_segment][0]) > tolerance:
            _square_offset_curve_endpoint(offset_curves, current_segments[i_segment], start, "start")
        if i_segment < len(segment_slices) - 1 and np.linalg.norm(current_segments[i_segment][-1] - current_segments[i_segment + 1][0]) > tolerance:
            _square_offset_curve_endpoint(offset_curves, current_segments[i_segment], end - 1, "end")


def _square_offset_curve_endpoint(offset_curves, outer_segment, boundary_index, boundary_end):
    """Place offset endpoints normal to the local outer-segment tangent."""

    outer_point = offset_curves[0.0][boundary_index]
    tangent = _segment_end_tangent(outer_segment, boundary_end)
    for thickness, points in offset_curves.items():
        if thickness <= 0:
            continue
        current_offset = points[boundary_index] - outer_point
        normal = _perp(tangent)
        if np.dot(normal, current_offset) < 0:
            normal = -normal
        points[boundary_index] = outer_point + normal * thickness


def _square_stair_step_boundary(
    offset_curves,
    thicknesses,
    outer_segment,
    thickness,
    i_segment,
    boundary_index,
    boundary_end,
    closed=False,
):
    """Square a boundary where this segment is thicker than its neighbor.

    The thinner adjacent offset is first placed on the local normal, then the
    current thicker offset is placed on the same normal.  This creates a stepped
    through-thickness boundary instead of a diagonal connector between unequal
    laminate thicknesses.
    """

    adjacent_index = i_segment - 1 if boundary_end == "start" else i_segment + 1
    if adjacent_index < 0:
        if not closed:
            return
        adjacent_index = len(thicknesses) - 1
    if adjacent_index >= len(thicknesses):
        if not closed:
            return
        adjacent_index = 0

    adjacent_thickness = thicknesses[adjacent_index]
    if not thickness > adjacent_thickness > 0:
        return

    outer_point = offset_curves[0.0][boundary_index]
    tangent = _segment_end_tangent(outer_segment, boundary_end)
    current_offset = offset_curves[thickness][boundary_index] - outer_point
    normal = _perp(tangent)
    if np.dot(normal, current_offset) < 0:
        normal = -normal

    offset_curves[adjacent_thickness][boundary_index] = outer_point + normal * adjacent_thickness
    offset_curves[thickness][boundary_index] = outer_point + normal * thickness


def _square_stair_step_endpoint(
    inner_segment,
    outer_segment,
    offset_curves,
    thicknesses,
    thickness,
    i_segment,
    boundary_index,
    boundary_end,
    closed=False,
):
    """Apply the squared stair-step endpoint to the sliced inner segment."""

    adjacent_index = i_segment - 1 if boundary_end == "start" else i_segment + 1
    if adjacent_index < 0:
        if not closed:
            return
        adjacent_index = len(thicknesses) - 1
    if adjacent_index >= len(thicknesses):
        if not closed:
            return
        adjacent_index = 0

    adjacent_thickness = thicknesses[adjacent_index]
    if not thickness > adjacent_thickness > 0:
        return

    base_point = offset_curves[adjacent_thickness][boundary_index]
    tangent = _segment_end_tangent(outer_segment, boundary_end)
    current_offset = offset_curves[thickness][boundary_index] - offset_curves[0.0][boundary_index]
    normal = _perp(tangent)
    if np.dot(normal, current_offset) < 0:
        normal = -normal
    squared_point = base_point + normal * (thickness - adjacent_thickness)

    if boundary_end == "start":
        inner_segment[0] = squared_point
    else:
        inner_segment[-1] = squared_point


def _segment_end_tangent(points, boundary_end):
    """Return a unit tangent at the requested segment end.

    Degenerate one-point inputs use ``[1, 0, 0]`` so callers still have a stable
    normal direction rather than dividing by zero.
    """

    if len(points) < 2:
        return np.array([1.0, 0.0, 0.0])
    if boundary_end == "start":
        return _unit(points[1] - points[0])
    return _unit(points[-1] - points[-2])


def _stair_step_connectors(
    offset_curves,
    thicknesses,
    i_segment,
    start_index,
    end_index,
    inner_start=None,
    inner_end=None,
    closed=False,
):
    """Return start/end through-thickness connectors for one shell segment.

    Connectors include an intermediate point when the adjacent stack is thinner,
    preserving squared layer transitions in the FreeCAD face boundary.
    """

    thickness = thicknesses[i_segment]
    previous_thickness = thicknesses[i_segment - 1] if i_segment > 0 else (thicknesses[-1] if closed else thickness)
    next_thickness = thicknesses[i_segment + 1] if i_segment < len(thicknesses) - 1 else (thicknesses[0] if closed else thickness)

    outer_start = offset_curves[0.0][start_index]
    outer_end = offset_curves[0.0][end_index]
    inner_start = offset_curves[thickness][start_index] if inner_start is None else inner_start
    inner_end = offset_curves[thickness][end_index] if inner_end is None else inner_end

    start_connector = [inner_start]
    if thickness > previous_thickness > 0:
        start_connector.append(offset_curves[previous_thickness][start_index])
    start_connector.append(outer_start)

    end_connector = [outer_end]
    if thickness > next_thickness > 0:
        end_connector.append(offset_curves[next_thickness][end_index])
    end_connector.append(inner_end)

    return np.array(start_connector), np.array(end_connector)


def _merge_trailing_edge_region_pairs(regions):
    """Merge HP/LP shell regions that still share a trailing-edge endpoint.

    This legacy helper is kept for simple TE closure cases.  The current detailed
    TE path usually trims HP/LP apart and adds an adhesive face instead.
    """

    merged = []
    removed_ids = set()
    shell_regions_by_layer = {}
    for region in regions:
        if region.outer_points is None:
            continue
        layer = _layer_index_from_name(region.name)
        if layer is not None:
            shell_regions_by_layer.setdefault(layer, []).append(region)

    replacements = {}
    for layer_regions in shell_regions_by_layer.values():
        if len(layer_regions) < 2:
            continue
        hp_region = layer_regions[0]
        lp_region = layer_regions[-1]
        if not (
            np.allclose(hp_region.outer_points[0], lp_region.outer_points[-1])
            and np.allclose(hp_region.inner_points[0], lp_region.inner_points[-1])
        ):
            continue
        replacements[id(hp_region)] = _merged_trailing_edge_region(hp_region, lp_region)
        removed_ids.add(id(lp_region))

    for region in regions:
        if id(region) in removed_ids:
            continue
        merged.append(replacements.get(id(region), region))
    return merged


def _layer_index_from_name(name):
    """Extract the integer ``layerNN`` suffix from a generated region name."""

    marker = "_layer"
    if marker not in name:
        return None
    try:
        return int(name.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _merged_trailing_edge_region(hp_region, lp_region):
    """Build one explicit boundary from paired HP/LP trailing-edge regions."""

    station_prefix = hp_region.name.split("_", 1)[0]
    layer_suffix = hp_region.name.rsplit("_", 1)[-1]
    return FreeCADFaceRegion(
        name=f"{station_prefix}_TE_closure_{layer_suffix}",
        material_name=hp_region.material_name,
        ply_angle=hp_region.ply_angle,
        edge_points=[
            hp_region.outer_points,
            np.vstack((hp_region.outer_points[-1], lp_region.inner_points[0])),
            lp_region.inner_points,
            hp_region.inner_points,
            np.vstack((hp_region.inner_points[-1], lp_region.outer_points[0])),
            lp_region.outer_points,
        ],
        edge_kinds=["spline", "line", "spline", "spline", "line", "spline"],
    )


def _leading_edge_shell_regions(hp_stack, lp_stack, station, hp_outer_points, lp_outer_points, section, transformer):
    """Create paired HP/LP leading-edge shell regions when layer counts match.

    A shared LE offset avoids small overlaps at the nose by offsetting the HP
    and LP curves together as one combined curve.  If the layer counts differ,
    the function falls back to independent stack offsets.
    """

    if (
        len(hp_outer_points) < 2
        or len(lp_outer_points) < 2
        or _polyline_lengths(hp_outer_points)[-1] <= 1e-9
        or _polyline_lengths(lp_outer_points)[-1] <= 1e-9
    ):
        return []

    if len(hp_stack.plygroups) != len(lp_stack.plygroups):
        hp_regions = _shell_regions_from_stack(hp_stack, station, "HP", hp_outer_points, section, transformer)
        lp_regions = _shell_regions_from_stack(lp_stack, station, "LP", lp_outer_points, section, transformer)
        return hp_regions + lp_regions

    regions = []
    current_hp_outer = _clean_polyline(hp_outer_points)
    current_lp_outer = _clean_polyline(lp_outer_points)
    for i_layer, (hp_plygroup, lp_plygroup) in enumerate(zip(hp_stack.plygroups, lp_stack.plygroups)):
        hp_thickness = transformer.length_from_mm(hp_plygroup.nPlies * hp_plygroup.thickness)
        lp_thickness = transformer.length_from_mm(lp_plygroup.nPlies * lp_plygroup.thickness)
        thickness = 0.5 * (hp_thickness + lp_thickness)
        if thickness <= 0:
            continue

        combined_outer = np.vstack((current_hp_outer, current_lp_outer[1:]))
        combined_inner = _offset_open_polyline_inward(combined_outer, section.closed_points, thickness)
        current_hp_inner = _clean_polyline(combined_inner[: len(current_hp_outer)])
        current_lp_inner = _clean_polyline(combined_inner[len(current_hp_outer) - 1 :])

        regions.append(
            FreeCADFaceRegion(
                name=f"Station{station:03d}_HP_{hp_stack.name}_layer{i_layer:02d}",
                material_name=hp_plygroup.materialid,
                ply_angle=hp_plygroup.angle,
                laminate_name=_laminate_name(station, "HP", hp_stack.name, i_layer),
                plies=_plies_from_plygroup(hp_plygroup, transformer),
                outer_points=current_hp_outer,
                inner_points=current_hp_inner,
            )
        )
        regions.append(
            FreeCADFaceRegion(
                name=f"Station{station:03d}_LP_{lp_stack.name}_layer{i_layer:02d}",
                material_name=lp_plygroup.materialid,
                ply_angle=lp_plygroup.angle,
                laminate_name=_laminate_name(station, "LP", lp_stack.name, i_layer),
                plies=_plies_from_plygroup(lp_plygroup, transformer),
                outer_points=current_lp_outer,
                inner_points=current_lp_inner,
            )
        )
        current_hp_outer = current_hp_inner
        current_lp_outer = current_lp_inner

    return regions


def _web_regions(blade, station, transformer, cs_params, shell_regions):
    """Build shear-web laminate and adhesive regions for one station.

    Webs attach to the innermost HP/LP spar-cap shell regions.  The web layer
    widths are centered on prescribed spar-interface locations, optional
    adhesive thickness offsets the actual web edges inward, and the affected
    spar inner edges are split so FreeCAD face metadata still maps cleanly.
    """

    stackdb = blade.stackdb
    if stackdb.swstacks is None:
        return []

    regions = []
    hp_spar_region = _innermost_spar_region(shell_regions, blade, station, "HP", 3)
    lp_spar_region = _innermost_spar_region(shell_regions, blade, station, "LP", 8)
    if hp_spar_region is None or lp_spar_region is None:
        return regions

    hp_side_regions = _innermost_side_regions(shell_regions, "HP")
    lp_side_regions = _innermost_side_regions(shell_regions, "LP")
    hp_interface_edges = {}
    lp_interface_edges = {}
    if station >= stackdb.swstacks.shape[1]:
        # StackDB omits the final station when the web thickness tapers to
        # zero.  Do not reuse the previous station's web laminate there; that
        # creates duplicate/near-coincident webs at the blade tip.
        return regions
    stack_station = station
    for i_web in range(stackdb.swstacks.shape[0]):
        if i_web >= stackdb.swstacks.shape[0]:
            break
        web_stack = stackdb.swstacks[i_web, stack_station]
        if not web_stack.plygroups:
            continue

        web_thickness = transformer.length_from_mm(sum(web_stack.layer_thicknesses()))
        if web_thickness <= 0:
            continue

        web_points = (
            transformer.points(blade.keypoints.web_points[i_web][:, :, station])
            if _web_has_explicit_yaml_geometry(blade, i_web) and i_web < len(blade.keypoints.web_points)
            else None
        )
        hp_attach_region = hp_spar_region
        lp_attach_region = lp_spar_region
        if web_points is not None:
            # YAML-defined webs are not guaranteed to land on the spar-cap
            # stack.  Choose the shell segment from the outer-surface web
            # location, then attach to that segment's inner edge.  Selecting by
            # inner-edge distance alone can jump across a stack boundary after
            # laminate offsets are applied, which makes webs miss the thick
            # spar-cap faces even when the YAML arcs lie inside them.
            hp_attach_region = _nearest_outer_region(hp_side_regions, web_points[0]) or hp_spar_region
            lp_attach_region = _nearest_outer_region(lp_side_regions, web_points[1]) or lp_spar_region
            if _web_endpoint_is_inside_spar_arc(blade, i_web, station, "HP"):
                hp_attach_region = hp_spar_region
            if _web_endpoint_is_inside_spar_arc(blade, i_web, station, "LP"):
                lp_attach_region = lp_spar_region

        interfaces = _spar_web_interfaces(
            hp_attach_region.inner_points,
            lp_attach_region.inner_points,
            station,
            transformer,
            cs_params,
            i_web,
            web_stack,
            web_points,
        )
        if interfaces is None:
            continue

        # Keep HP/LP interval lists in plygroup order.  Reversing the LP list
        # pairs a thick core interval on one side with a thin skin interval on
        # the other, creating long triangular-looking web faces.
        hp_edges, lp_edges = interfaces
        hp_adhesive_edges, hp_web_edges = _web_connection_edges(hp_edges, lp_edges, transformer, cs_params, station, i_web)
        lp_adhesive_edges, lp_web_edges = _web_connection_edges(lp_edges, hp_edges, transformer, cs_params, station, i_web)

        hp_interface_edges.setdefault(id(hp_attach_region), (hp_attach_region, []) )[1].append(
            _join_connected_edges(_connected_edge_order(hp_adhesive_edges, hp_web_edges)[0])
        )
        lp_interface_edges.setdefault(id(lp_attach_region), (lp_attach_region, []) )[1].append(
            _join_connected_edges(_connected_edge_order(lp_adhesive_edges, lp_web_edges)[0])
        )

        regions.extend(_web_laminate_regions(station, i_web, hp_web_edges, lp_web_edges, web_stack, transformer))
        regions.extend(
            _web_adhesive_regions_from_edges(
                station,
                i_web,
                hp_adhesive_edges,
                hp_web_edges,
                lp_adhesive_edges,
                lp_web_edges,
                cs_params,
            )
        )

    for region, interface_edges in hp_interface_edges.values():
        _split_region_inner_edge_for_interfaces(region, interface_edges)
    for region, interface_edges in lp_interface_edges.values():
        _split_region_inner_edge_for_interfaces(region, interface_edges)
    return regions


def _innermost_spar_region(shell_regions, blade, station, side, stack_index):
    """Return the innermost generated shell layer for a spar-cap stack."""

    stackdb = blade.stackdb
    stack_station = min(station, stackdb.stacks.shape[1] - 1)
    stack_name = stackdb.stacks[stack_index, stack_station].name
    prefix = f"Station{station:03d}_{side}_{stack_name}_layer"
    candidates = [region for region in shell_regions if region.name.startswith(prefix) and region.inner_points is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda region: _layer_index_from_name(region.name) or 0)


def _innermost_side_regions(shell_regions, side):
    """Return the innermost generated shell layer for each stack on one side."""

    by_stack = {}
    for region in shell_regions:
        if f"_{side}_" not in region.name or region.inner_points is None:
            continue
        stack_name = region.name.rsplit("_layer", 1)[0]
        current = by_stack.get(stack_name)
        if current is None or (_layer_index_from_name(region.name) or 0) > (_layer_index_from_name(current.name) or 0):
            by_stack[stack_name] = region
    return list(by_stack.values())


def _nearest_outer_region(regions, point):
    """Find the shell region whose outer curve is closest to a point."""

    best_region = None
    best_distance = float("inf")
    for region in regions:
        distance = _distance_to_polyline(region.outer_points, point)
        if distance < best_distance:
            best_distance = distance
            best_region = region
    return best_region


def _web_has_explicit_yaml_geometry(blade, i_web):
    """Return whether this web group came from WindIO start/end_nd_arc data."""

    components = blade.definition.components.values()
    return any(
        component.group == i_web + 1
        and component.web_start_nd_arc is not None
        and component.web_end_nd_arc is not None
        for component in components
    )


def _web_endpoint_is_inside_spar_arc(blade, i_web, station, side, tolerance=1e-9):
    """Return whether a YAML web endpoint falls inside the spar-cap arc bounds."""

    if i_web >= len(blade.keypoints.web_arcs):
        return False
    if side == "HP":
        web_arc = blade.keypoints.web_arcs[i_web][0, station]
        spar_arcs = blade.keypoints.key_arcs[[3, 4], station]
    else:
        web_arc = blade.keypoints.web_arcs[i_web][1, station]
        spar_arcs = blade.keypoints.key_arcs[[8, 9], station]
    return min(spar_arcs) - tolerance <= web_arc <= max(spar_arcs) + tolerance


def _spar_web_interfaces(hp_inner_spar, lp_inner_spar, station, transformer, cs_params, i_web, web_stack, web_points=None):
    """Return HP/LP interface edge intervals for one shear web.

    Layer widths come from web ply thicknesses.  The interface center is inset
    from the spar end by web thickness plus adhesive width, then limited to
    ``45%`` of each spar length so very short spar regions cannot be consumed by
    the web connection.
    """

    layer_widths = [
        transformer.length_from_mm(plygroup.nPlies * plygroup.thickness)
        for plygroup in web_stack.plygroups
        if transformer.length_from_mm(plygroup.nPlies * plygroup.thickness) > 0
    ]
    web_thickness = sum(layer_widths)
    if web_thickness <= 0:
        return None

    adhesive_width = transformer.length_from_m(
        _station_value(cs_params.get("web_adhesive_width"), station, default=0.0)
    )
    inset = web_thickness + adhesive_width

    hp_length = _polyline_lengths(hp_inner_spar)[-1]
    lp_length = _polyline_lengths(lp_inner_spar)[-1]
    inset = min(inset, 0.45 * hp_length, 0.45 * lp_length)
    if inset <= 1e-9:
        return None

    if web_points is not None:
        # Prefer pyNuMAD/WindIO web keypoints when they are available.  The old
        # FreeCAD generator only had two locations: web 0 at one spar end and
        # every other web at the opposite end.  That hid additional YAML webs by
        # drawing them on top of each other.  Projecting each web endpoint onto
        # the innermost spar curves keeps any number of web stacks distinct.
        hp_center = _project_point_to_polyline(hp_inner_spar, web_points[0])
        lp_center = _project_point_to_polyline(lp_inner_spar, web_points[1])
    elif i_web == 0:
        hp_center = hp_length - inset
        lp_center = inset
    else:
        hp_center = inset
        lp_center = lp_length - inset

    boundary_clearance = _web_boundary_clearance(web_thickness, adhesive_width, hp_length, lp_length)
    hp_intervals = _centered_intervals(hp_length, hp_center, layer_widths, boundary_clearance)
    # The HP and LP spar curves run in opposite physical directions around the
    # section.  Build the LP intervals from reversed widths, then restore
    # plygroup order, so each web layer connects to a same-thickness interval
    # without crossing the outer web layers.
    lp_intervals = list(reversed(_centered_intervals(lp_length, lp_center, list(reversed(layer_widths)), boundary_clearance)))
    if not hp_intervals or not lp_intervals:
        return None

    hp_edges = [_polyline_between(hp_inner_spar, start, end) for start, end in hp_intervals]
    lp_edges = [_polyline_between(lp_inner_spar, start, end) for start, end in lp_intervals]
    return hp_edges, lp_edges


def _web_boundary_clearance(web_thickness, adhesive_width, hp_length, lp_length):
    """Return a small clearance from shell-region boundaries for web layers.

    Near the blade tip, a YAML web endpoint can lie just inside a spar-cap
    interval.  If the laminate-width interval is clipped flush to the spar/panel
    boundary, one side of the web core can cut across the neighboring panel.
    Use a modest clearance when the spar segment has enough room, but let very
    short segments fall back to zero clearance instead of deleting the web.
    """

    available_length = min(hp_length, lp_length)
    requested = max(adhesive_width, 0.25 * web_thickness, 0.002)
    if web_thickness + 2.0 * requested <= 0.95 * available_length:
        return requested
    return 0.0


def _centered_intervals(length, center, widths, boundary_clearance=0.0):
    """Place consecutive layer-width intervals around a center distance.

    If the requested total width exceeds the available curve length, widths are
    uniformly scaled to ``95%`` of the length.  Intervals shorter than ``1e-9``
    are discarded as degenerate.
    """

    widths = [width for width in widths if width > 0]
    total_width = sum(widths)
    if total_width <= 0 or length <= 1e-9:
        return []
    if total_width > length:
        scale = 0.95 * length / total_width
        widths = [width * scale for width in widths]
        total_width = sum(widths)

    start = center - 0.5 * total_width
    lower = min(boundary_clearance, max(length - total_width, 0.0))
    upper = max(length - total_width - boundary_clearance, lower)
    start = float(np.clip(start, lower, upper))
    intervals = []
    for width in widths:
        end = start + width
        if end - start > 1e-9:
            intervals.append((start, end))
        start = end
    return intervals


def _web_interface_centerline(hp_edges, lp_edges):
    """Return an approximate centerline joining HP and LP web interfaces."""

    hp_points = np.vstack((hp_edges[0][0], hp_edges[-1][-1]))
    lp_points = np.vstack((lp_edges[0][0], lp_edges[-1][-1]))
    return np.vstack((hp_points.mean(axis=0), lp_points.mean(axis=0)))


def _web_connection_edges(interface_edges, opposite_edges, transformer, cs_params, station, i_web):
    """Separate shell adhesive edges from web laminate edges.

    With zero adhesive thickness, both sets of edges are identical.  With a
    positive fore/aft adhesive thickness, web edges are offset toward the
    opposite interface while the original interface remains the adhesive outer
    boundary.
    """

    adhesive_m = _station_value(
        cs_params.get("web_fore_adhesive_thickness" if i_web == 0 else "web_aft_adhesive_thickness"),
        station,
        default=0.0,
    )
    adhesive_thickness = transformer.length_from_m(adhesive_m)
    if adhesive_thickness <= 0:
        return interface_edges, interface_edges

    target = np.vstack(opposite_edges).mean(axis=0)
    web_edges = [_offset_edge_toward(edge, target, adhesive_thickness) for edge in interface_edges]
    return interface_edges, web_edges


def _offset_edge_toward(edge, target, distance):
    """Offset each point of an edge toward a target point by a fixed distance."""

    offset_points = []
    for point in edge:
        direction = _unit(target - point)
        offset_points.append(point + direction * distance)
    return np.array(offset_points)


def _split_region_inner_edge_for_interfaces(region, interface_edges):
    """Replace a spar inner curve with pieces split around web interfaces.

    The shell region originally has one continuous inner spline.  After web
    insertion, portions covered by web adhesive should be exact interface edges.
    This function projects interface endpoints to the inner curve, splits the
    curve by arc length, and rewrites the region as an explicit edge boundary.
    """

    if region is None or not interface_edges:
        return

    inner_points = region.inner_points
    inner_length = _polyline_lengths(inner_points)[-1]
    split_distances = [0.0, inner_length]
    interface_intervals = []
    for edge in interface_edges:
        start = _project_point_to_polyline(inner_points, edge[0])
        end = _project_point_to_polyline(inner_points, edge[-1])
        if end < start:
            start, end = end, start
            edge = np.flip(edge, axis=0)
        split_distances.extend((start, end))
        interface_intervals.append((start, end, edge))

    split_distances = sorted(split_distances)
    unique_distances = []
    for distance in split_distances:
        if not unique_distances or abs(distance - unique_distances[-1]) > 1e-9:
            unique_distances.append(distance)

    inner_edges = []
    for start, end in zip(unique_distances[:-1], unique_distances[1:]):
        if end - start <= 1e-9:
            continue
        matching_edge = next(
            (
                edge
                for edge_start, edge_end, edge in interface_intervals
                if abs(edge_start - start) <= 1e-9 and abs(edge_end - end) <= 1e-9
            ),
            None,
        )
        inner_edges.append(matching_edge if matching_edge is not None else _polyline_between(inner_points, start, end))

    if not inner_edges:
        return

    region.edge_points = [region.outer_points]
    region.edge_kinds = ["spline"]
    region.edge_points.append(region.end_connector)
    region.edge_kinds.append("line")
    for edge in reversed(inner_edges):
        region.edge_points.append(np.flip(edge, axis=0))
        region.edge_kinds.append("spline")
    region.edge_points.append(region.start_connector)
    region.edge_kinds.append("line")


def _web_laminate_regions(station, i_web, hp_edges, lp_edges, web_stack, transformer):
    """Create one web laminate face per nonzero web plygroup."""

    regions = []
    i_valid_layer = 0
    for i_layer, plygroup in enumerate(web_stack.plygroups):
        if i_valid_layer >= len(hp_edges) or i_valid_layer >= len(lp_edges):
            break
        if plygroup.nPlies * plygroup.thickness <= 0:
            continue
        hp_edge = hp_edges[i_valid_layer]
        lp_edge = lp_edges[i_valid_layer]
        regions.append(
            FreeCADFaceRegion(
                name=f"Station{station:03d}_web{i_web}_layer{i_layer:02d}",
                material_name=plygroup.materialid,
                ply_angle=plygroup.angle,
                laminate_name=_laminate_name(station, f"web{i_web}", "SW", i_layer),
                plies=_plies_from_plygroup(plygroup, transformer),
                edge_points=_web_face_edges(hp_edge, lp_edge),
                edge_kinds=["spline", "line", "spline", "line"],
            )
        )
        i_valid_layer += 1
    return regions


def _web_adhesive_regions_from_edges(station, i_web, hp_outer_edges, hp_inner_edges, lp_outer_edges, lp_inner_edges, cs_params):
    """Create HP/LP web adhesive regions when adhesive thickness is nonzero."""

    regions = []
    adhesive_name = cs_params.get("adhesive_mat_name", "Adhesive")
    for side, outer_edges, inner_edges in (
        ("hp", hp_outer_edges, hp_inner_edges),
        ("lp", lp_outer_edges, lp_inner_edges),
    ):
        if all(np.allclose(outer_edge, inner_edge) for outer_edge, inner_edge in zip(outer_edges, inner_edges)):
            continue

        edge_points, edge_kinds = _web_adhesive_face_edges(outer_edges, inner_edges)
        regions.append(
            FreeCADFaceRegion(
                name=f"Station{station:03d}_web{i_web}_{side}_adhesive",
                material_name=adhesive_name,
                ply_angle=0,
                edge_points=edge_points,
                edge_kinds=edge_kinds,
            )
        )

    return regions


def _web_adhesive_face_edges(outer_edges, inner_edges):
    """Return the lower-gap, non-crossing boundary for a web adhesive face.

    Two plausible orientations are tested because HP/LP edge order can flip
    depending on station geometry.  The chosen orientation minimizes closure gap
    and penalizes self-intersection.
    """

    outer_edges, inner_edges = _connected_edge_order(outer_edges, inner_edges)
    outer_edge = _join_connected_edges(outer_edges)

    forward_edges = []
    forward_kinds = []
    forward_edges.append(outer_edge)
    forward_kinds.append("spline")
    forward_edges.append(np.vstack((outer_edge[-1], inner_edges[-1][-1])))
    forward_kinds.append("line")
    for edge in reversed(inner_edges):
        forward_edges.append(np.flip(edge, axis=0))
        forward_kinds.append("spline")
    forward_edges.append(np.vstack((inner_edges[0][0], outer_edge[0])))
    forward_kinds.append("line")

    alternate_edges = []
    alternate_kinds = []
    alternate_edges.append(outer_edge)
    alternate_kinds.append("spline")
    alternate_edges.append(np.vstack((outer_edge[-1], inner_edges[0][0])))
    alternate_kinds.append("line")
    for edge in inner_edges:
        alternate_edges.append(edge)
        alternate_kinds.append("spline")
    alternate_edges.append(np.vstack((inner_edges[-1][-1], outer_edge[0])))
    alternate_kinds.append("line")

    forward_score = _edge_boundary_gap(forward_edges)
    alternate_score = _edge_boundary_gap(alternate_edges)
    if _edge_boundary_self_intersects(forward_edges):
        forward_score += 1.0
    if _edge_boundary_self_intersects(alternate_edges):
        alternate_score += 1.0
    if alternate_score < forward_score:
        return alternate_edges, alternate_kinds
    return forward_edges, forward_kinds


def _join_connected_edges(edges):
    """Join ordered edge point arrays into one polyline without duplicates."""

    points = []
    for edge in edges:
        if not points:
            points.extend(edge)
        else:
            points.extend(edge[1:])
    return np.array(points)


def _connected_edge_order(outer_edges, inner_edges):
    """Choose the edge ordering/orientation with the smallest path gaps."""

    candidates = [
        (outer_edges, inner_edges),
        ([np.flip(edge, axis=0) for edge in outer_edges], [np.flip(edge, axis=0) for edge in inner_edges]),
        (list(reversed(outer_edges)), list(reversed(inner_edges))),
        (
            [np.flip(edge, axis=0) for edge in reversed(outer_edges)],
            [np.flip(edge, axis=0) for edge in reversed(inner_edges)],
        ),
    ]

    def path_gap(edges):
        if len(edges) < 2:
            return 0.0
        return max(np.linalg.norm(first[-1] - second[0]) for first, second in zip(edges[:-1], edges[1:]))

    return min(candidates, key=lambda candidate: path_gap(candidate[0]) + path_gap(candidate[1]))


def _web_face_edges(first_edge, second_edge):
    """Return a four-edge web face boundary, avoiding crossed connectors."""

    forward = [
        first_edge,
        np.vstack((first_edge[-1], second_edge[-1])),
        np.flip(second_edge, axis=0),
        np.vstack((second_edge[0], first_edge[0])),
    ]
    crossed = _edge_boundary_self_intersects(forward)

    alternate = [
        first_edge,
        np.vstack((first_edge[-1], second_edge[0])),
        second_edge,
        np.vstack((second_edge[-1], first_edge[0])),
    ]
    if crossed and not _edge_boundary_self_intersects(alternate):
        return alternate
    return forward


def _edge_boundary_gap(edge_points):
    """Return the largest endpoint gap between consecutive boundary edges."""

    if not edge_points:
        return 0.0
    return max(
        np.linalg.norm(first[-1] - second[0])
        for first, second in zip(edge_points, edge_points[1:] + edge_points[:1])
    )


def _edge_boundary_self_intersects(edge_points):
    """Return whether an ordered edge boundary crosses itself in 2D."""

    points = []
    for edge in edge_points:
        if not points:
            points.extend(edge)
        else:
            points.extend(edge[1:])
    points = np.array(points)
    for i_point in range(len(points)):
        first_start = points[i_point]
        first_end = points[(i_point + 1) % len(points)]
        for j_point in range(i_point + 1, len(points)):
            if abs(i_point - j_point) <= 1 or (i_point == 0 and j_point == len(points) - 1):
                continue
            second_start = points[j_point]
            second_end = points[(j_point + 1) % len(points)]
            if _segments_intersect_2d(first_start, first_end, second_start, second_end):
                return True
    return False


def _segments_intersect_2d(first_start, first_end, second_start, second_end):
    """Return true when two non-adjacent 2D segments properly intersect."""

    first_orientation = _orientation_2d(first_start, first_end, second_start)
    second_orientation = _orientation_2d(first_start, first_end, second_end)
    third_orientation = _orientation_2d(second_start, second_end, first_start)
    fourth_orientation = _orientation_2d(second_start, second_end, first_end)
    return first_orientation * second_orientation < -1e-12 and third_orientation * fourth_orientation < -1e-12


def _orientation_2d(start, end, point):
    """Return the signed 2D cross product for point orientation."""

    return np.cross(end[:2] - start[:2], point[:2] - start[:2])


def _split_polyline_at_points(points, boundary_points):
    """Split a polyline at projected boundary points.

    Boundary points are first projected to arc-length locations on ``points``.
    This avoids depending on exact coordinate equality between keypoints and
    sampled airfoil coordinates.
    """

    distances = [_project_point_to_polyline(points, point) for point in boundary_points]
    distances[0] = 0.0
    distances[-1] = _polyline_lengths(points)[-1]
    segments = []
    for start, end in zip(distances[:-1], distances[1:]):
        if end < start:
            start, end = end, start
        segments.append(_polyline_between(points, start, end))
    return segments


def _polyline_lengths(points):
    """Return cumulative arc lengths for a polyline."""

    distances = [0.0]
    for start, end in zip(points[:-1], points[1:]):
        distances.append(distances[-1] + np.linalg.norm(end - start))
    return np.array(distances)


def _clean_polyline(points, min_distance=1e-3):
    """Remove near-duplicate interior polyline points.

    The default ``1e-3`` output-unit spacing is deliberately larger than pure
    floating-point tolerance; it suppresses very short edges that can create tiny
    mesh elements while preserving endpoints exactly.
    """

    if len(points) <= 2:
        return points

    cleaned = [points[0]]
    for point in points[1:-1]:
        if np.linalg.norm(point - cleaned[-1]) >= min_distance:
            cleaned.append(point)

    if np.linalg.norm(points[-1] - cleaned[-1]) < min_distance and len(cleaned) > 1:
        cleaned[-1] = points[-1]
    else:
        cleaned.append(points[-1])

    return np.array(cleaned)


def _clean_polygon_points(points, min_distance=1e-9):
    """Remove duplicate polygon points and an optional duplicate closing point."""

    if len(points) <= 2:
        return points

    cleaned = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - cleaned[-1]) >= min_distance:
            cleaned.append(point)

    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) < min_distance:
        cleaned.pop()

    return np.array(cleaned)


def _valid_face_boundary(outer_points, inner_points, tolerance=1e-9):
    """Return whether two strip curves can form a non-degenerate face."""

    if len(outer_points) < 2 or len(inner_points) < 2:
        return False
    return _polyline_lengths(outer_points)[-1] > tolerance and _polyline_lengths(inner_points)[-1] > tolerance


def _project_point_to_polyline(points, point):
    """Project a point to the closest arc-length location on a polyline."""

    cumulative = _polyline_lengths(points)
    best_distance = float("inf")
    best_s = 0.0
    for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared == 0:
            continue
        t = np.clip(float(np.dot(point - start, segment) / length_squared), 0.0, 1.0)
        projected = start + t * segment
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance = distance
            best_s = cumulative[i] + t * np.sqrt(length_squared)
    return best_s


def _distance_to_polyline(points, point):
    """Return the shortest distance from a point to a polyline."""

    projected = _point_at_distance(points, _project_point_to_polyline(points, point))
    return float(np.linalg.norm(point - projected))


def _point_at_distance(points, distance):
    """Interpolate a point at a clipped arc-length distance on a polyline."""

    cumulative = _polyline_lengths(points)
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    index = np.searchsorted(cumulative, distance, side="right") - 1
    index = min(index, len(points) - 2)
    start = points[index]
    end = points[index + 1]
    length = cumulative[index + 1] - cumulative[index]
    if length == 0:
        return start.copy()
    fraction = (distance - cumulative[index]) / length
    return start + fraction * (end - start)


def _polyline_between(points, start_distance, end_distance):
    """Return the polyline sub-curve between two arc-length distances."""

    cumulative = _polyline_lengths(points)
    selected = [_point_at_distance(points, start_distance)]
    for i, distance in enumerate(cumulative[1:-1], start=1):
        if start_distance < distance < end_distance:
            selected.append(points[i])
    selected.append(_point_at_distance(points, end_distance))
    return np.array(selected)


def _offset_open_polyline_inward(points, closed_points, distance, miter_limit=4.0):
    """Offset an open polyline inward relative to the closed airfoil boundary.

    Segment normals are chosen from the orientation of ``closed_points``.  At
    interior vertices, adjacent offset lines are intersected to form a miter. If
    lines are parallel or the miter exceeds ``miter_limit * distance`` (default
    ``4.0``), an averaged-normal fallback prevents long spikes.
    """

    orientation = _polygon_orientation(closed_points)
    distances = np.full((len(points),), distance) if np.isscalar(distance) else np.array(distance)
    tangents = []
    normals = []
    for start, end in zip(points[:-1], points[1:]):
        tangent = _unit(end - start)
        normal = _perp(tangent)
        if orientation < 0:
            normal = -normal
        tangents.append(tangent)
        normals.append(normal)

    offset_points = [points[0] + distances[0] * normals[0]]
    for i in range(1, len(points) - 1):
        previous_start = points[i - 1] + distances[i - 1] * normals[i - 1]
        previous_end = points[i] + distances[i] * normals[i - 1]
        next_start = points[i] + distances[i] * normals[i]
        next_end = points[i + 1] + distances[i + 1] * normals[i]
        intersection = _line_intersection_2d(previous_start, previous_end, next_start, next_end)
        miter_length = np.linalg.norm(intersection - points[i]) if intersection is not None else float("inf")
        maximum_miter = miter_limit * abs(distances[i])
        if intersection is None or (maximum_miter > 0 and miter_length > maximum_miter):
            normal = _unit(normals[i - 1] + normals[i])
            intersection = points[i] + distances[i] * normal
        offset_points.append(intersection)

    offset_points.append(points[-1] + distances[-1] * normals[-1])
    return np.array(offset_points)


def _line_intersection_2d(first_start, first_end, second_start, second_end, tolerance=1e-12):
    """Return the 2D line intersection, or ``None`` for near-parallel lines."""

    first_direction = first_end[:2] - first_start[:2]
    second_direction = second_end[:2] - second_start[:2]
    matrix = np.column_stack((first_direction, -second_direction))
    determinant = np.linalg.det(matrix)
    if abs(determinant) <= tolerance:
        return None

    parameters = np.linalg.solve(matrix, second_start[:2] - first_start[:2])
    xy = first_start[:2] + parameters[0] * first_direction
    return np.array([xy[0], xy[1], first_start[2]])


def _polygon_orientation(points):
    """Return ``1`` for counterclockwise or ``-1`` for clockwise point order."""

    xy = np.asarray(points)[:, :2]
    x = xy[:, 0]
    y = xy[:, 1]
    return 1.0 if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) >= 0 else -1.0


def _rectangle_about_line(start, end, width):
    """Return four points for a rectangle centered on a line segment."""

    tangent = _unit(end - start)
    normal = _perp(tangent)
    half_width = 0.5 * width
    return np.vstack((start - normal * half_width, end - normal * half_width, end + normal * half_width, start + normal * half_width))


def _perp(vector):
    """Return the in-plane left normal of a vector."""

    return np.array([-vector[1], vector[0], 0.0])


def _unit(vector):
    """Return a unit vector, using x-direction for zero-length input."""

    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _station_value(values, station, default=0.0):
    """Return a scalar parameter or the station-specific value from a sequence."""

    if values is None:
        return default
    try:
        return float(values[station])
    except (TypeError, IndexError):
        return float(values)


def _json_value(value):
    """Return a JSON-safe scalar/list representation for NumPy/Python values."""

    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _abs_json_value(value):
    converted = _json_value(value)
    if isinstance(converted, list):
        return [_abs_json_value(item) for item in converted]
    if isinstance(converted, (int, float)):
        return abs(converted)
    return converted


def _is_nan(value):
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _laminate_name(station, side, stack_name, i_layer):
    """Return a stable laminate name for a generated plygroup face."""

    return f"Station{station:03d}_{side}_{stack_name}_layer{i_layer:02d}_laminate"


def _plies_from_plygroup(plygroup, transformer):
    """Expand a pyNuMAD plygroup into HomoGen-style ply entries.

    pyNuMAD consolidates consecutive plies with the same material and angle into
    one plygroup.  HomoGen expects the laminate definition as an ordered list of
    plies, so repeated plies are expanded here.  ``thickness`` is in the same
    units as the exported FreeCAD geometry.
    Core, coating, resin, and adhesive-like plygroups are returned as ``None`` so
    their faces remain direct material assignments instead of laminate entries.
    """

    if not _plygroup_is_laminate(plygroup):
        return None

    n_plies = int(plygroup.nPlies or 0)
    if n_plies <= 0:
        return []

    thickness = float(transformer.length_from_mm(plygroup.thickness))
    return [
        {
            "material": plygroup.materialid,
            "angle": float(plygroup.angle),
            "thickness": thickness,
        }
        for _ in range(n_plies)
    ]


def _plygroup_is_laminate(plygroup):
    """Return whether a plygroup should be exported as a laminate definition."""

    material_name = str(plygroup.materialid).lower()
    non_laminate_markers = ("foam", "core", "gelcoat", "adhesive", "resin")
    return not any(marker in material_name for marker in non_laminate_markers)


def _laminate_key(plies):
    """Return a deterministic key for comparing laminate ply stacks."""

    return json.dumps(plies, sort_keys=True, separators=(",", ":"))


def _require_freecad_modules():
    """Import FreeCAD modules or raise a targeted setup error."""

    try:
        import FreeCAD as App
        import Part
    except ImportError as exc:
        raise ImportError(
            "FreeCAD Python modules are required for make_freecad_section_part() "
            "and make_freecad_cross_section_parts(). Run this from FreeCAD, "
            "freecadcmd, or a Python environment that can import FreeCAD."
        ) from exc
    return App, Part


def _freecad_vector(point, App):
    """Convert a NumPy/list point to ``FreeCAD.Vector``."""

    return App.Vector(float(point[0]), float(point[1]), float(point[2]))


def _freecad_bspline_edge(points, App, Part):
    """Create a FreeCAD edge from points, using a line for two-point edges."""

    if len(points) == 2:
        return Part.LineSegment(_freecad_vector(points[0], App), _freecad_vector(points[1], App)).toShape()
    curve = Part.BSplineCurve()
    curve.interpolate([_freecad_vector(point, App) for point in points])
    return curve.toShape()


def _freecad_line_edges(points, App, Part):
    """Create straight FreeCAD edges between consecutive nonduplicate points."""

    edges = []
    for start, end in zip(points[:-1], points[1:]):
        if _freecad_vector(start, App).distanceToPoint(_freecad_vector(end, App)) > 1e-9:
            edges.append(Part.LineSegment(_freecad_vector(start, App), _freecad_vector(end, App)).toShape())
    return edges


def _freecad_face_from_points(points, App, Part):
    """Create a planar FreeCAD face from an ordered polygon boundary."""

    closed = list(points)
    closed.append(points[0])
    return Part.Face(Part.makePolygon([_freecad_vector(point, App) for point in closed]))


def _freecad_face_between_curves(outer_points, inner_points, App, Part, start_connector=None, end_connector=None):
    """Create a FreeCAD face bounded by outer/inner curves and connectors.

    For straight end connectors, a ruled surface is preferred because FreeCAD's
    generic wire face filling can occasionally create large spurious rectangular
    faces for long, thin spline strips.  Non-straight connectors, or ruled
    surface failures, fall back to a closed wire face.
    """

    outer_edge = _freecad_bspline_edge(outer_points, App, Part)
    if start_connector is None:
        start_connector = [inner_points[0], outer_points[0]]
    if end_connector is None:
        end_connector = [outer_points[-1], inner_points[-1]]
    if _connector_is_straight(start_connector) and _connector_is_straight(end_connector):
        # FreeCAD can occasionally fill long, thin spline wires with a spurious
        # rectangular face; a ruled surface keeps these strip regions bounded by
        # the intended inner and outer curves.
        inner_edge = _freecad_bspline_edge(inner_points, App, Part)
        try:
            return Part.makeRuledSurface(outer_edge, inner_edge)
        except Exception:
            pass

    inner_edge = _freecad_bspline_edge(list(reversed(inner_points)), App, Part)
    edges = [outer_edge]
    edges.extend(_freecad_line_edges(end_connector, App, Part))
    edges.append(inner_edge)
    edges.extend(_freecad_line_edges(start_connector, App, Part))
    return Part.Face(Part.Wire(edges))


def _connector_is_straight(points, tolerance=1e-7):
    """Return whether connector points lie on one 2D line within tolerance."""

    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        return True
    start = points[0]
    end = points[-1]
    segment = end - start
    length = np.linalg.norm(segment[:2])
    if length <= tolerance:
        return False
    for point in points[1:-1]:
        distance = abs(np.cross(segment[:2], (point - start)[:2])) / length
        if distance > tolerance:
            return False
    return True


def _freecad_face_from_edge_points(edge_points, edge_kinds, App, Part):
    """Create a FreeCAD face from an explicit edge list.

    ``edge_kinds`` selects either line edges or spline edges for each point
    group.  This path is used for adhesives and split boundaries whose topology
    is more specific than a simple outer/inner strip.
    """

    edges = []
    for points, kind in zip(edge_points, edge_kinds):
        if kind == "line":
            edges.extend(_freecad_line_edges(points, App, Part))
        else:
            edges.append(_freecad_bspline_edge(points, App, Part))
    return Part.Face(Part.Wire(edges))


def _freecad_face_from_region(region, App, Part):
    """Dispatch a serialized or dataclass region to the right face builder."""

    if _region_value(region, "points") is not None:
        return _freecad_face_from_points(_region_value(region, "points"), App, Part)
    if _region_value(region, "edge_points") is not None:
        return _freecad_face_from_edge_points(
            _region_value(region, "edge_points"),
            _region_value(region, "edge_kinds"),
            App,
            Part,
        )
    return _freecad_face_between_curves(
        _region_value(region, "outer_points"),
        _region_value(region, "inner_points"),
        App,
        Part,
        _region_value(region, "start_connector"),
        _region_value(region, "end_connector"),
    )


def _freecad_stitched_section_shape(faces, Part):
    """Return a sewn FreeCAD compound from individual face shapes."""

    shape = Part.makeCompound(faces)
    shape.sewShape()
    return shape


def _set_string_property(obj, name, value, group="", description=""):
    """Set a FreeCAD string property, creating it if needed."""

    if name not in getattr(obj, "PropertiesList", []):
        obj.addProperty("App::PropertyString", name, group, description)
    setattr(obj, name, value)


def _get_or_create_turbine_metadata_object(doc):
    metadata_obj = doc.getObject("TurbineMetadata") if hasattr(doc, "getObject") else None
    if metadata_obj is None:
        metadata_obj = doc.addObject("App::FeaturePython", "TurbineMetadata")
        metadata_obj.Label = "Turbine Metadata"
    return metadata_obj


def _initialize_turbine_metadata_defaults(metadata_obj, *, station_count=0):
    if "MaterialDefinitions" not in getattr(metadata_obj, "PropertiesList", []):
        _set_string_property(
            metadata_obj,
            "MaterialDefinitions",
            json.dumps([]),
            group="Turbine",
            description="JSON turbine-level material property table from pyNuMAD",
        )
    if "LaminateDefinitions" not in getattr(metadata_obj, "PropertiesList", []):
        _set_string_property(
            metadata_obj,
            "LaminateDefinitions",
            json.dumps([]),
            group="Turbine",
            description="JSON turbine-level laminate ply stack table",
        )
    if "StationCount" not in getattr(metadata_obj, "PropertiesList", []):
        metadata_obj.addProperty("App::PropertyInteger", "StationCount", "Turbine", "Number of blade stations in the YAML")
        metadata_obj.StationCount = int(station_count)
    _set_turbine_messages(metadata_obj, _existing_turbine_messages(metadata_obj))


def _make_turbine_metadata_object(doc, *, material_table, laminate_table, station_count):
    """Create/update the document-level HomoGen metadata object."""

    metadata_obj = _get_or_create_turbine_metadata_object(doc)
    _set_string_property(
        metadata_obj,
        "MaterialDefinitions",
        json.dumps(material_table),
        group="Turbine",
        description="JSON turbine-level material property table from pyNuMAD",
    )
    _set_string_property(
        metadata_obj,
        "LaminateDefinitions",
        json.dumps(laminate_table),
        group="Turbine",
        description="JSON turbine-level laminate ply stack table",
    )
    if "StationCount" not in getattr(metadata_obj, "PropertiesList", []):
        metadata_obj.addProperty("App::PropertyInteger", "StationCount", "Turbine", "Number of blade stations in the YAML")
    metadata_obj.StationCount = int(station_count)
    _set_turbine_messages(metadata_obj, [])
    return metadata_obj


def _set_view_color(obj, color):
    """Assign a view color when running in a FreeCAD GUI-capable context."""

    if hasattr(obj, "ViewObject") and obj.ViewObject is not None:
        obj.ViewObject.ShapeColor = color


def _color_for_material(material_name):
    """Return a simple display color based on material name keywords."""

    name = material_name.lower()
    if "adhesive" in name:
        return (1.0, 0.84, 0.0, 0.0)
    if "carbon" in name:
        return (0.18, 0.18, 0.18, 0.0)
    if "foam" in name:
        return (0.84, 0.78, 0.52, 0.0)
    if "web" in name:
        return (0.2, 0.5, 0.85, 0.0)
    if "spar" in name or "uni" in name:
        return (0.22, 0.55, 0.32, 0.0)
    if "triax" in name or "biax" in name:
        return (0.56, 0.8, 0.36, 0.0)
    return (0.68, 0.72, 0.78, 0.0)


def _region_value(region, key):
    """Read a field from either a serialized dict or a region dataclass."""

    if isinstance(region, dict):
        return region.get(key)
    return getattr(region, key)


def _parsed_region_name(name):
    """Parse generated region names into a fixed snake_case metadata schema."""

    parts = name.split("_")
    parsed = dict(
        station=None,
        layer=None,
        side=None,
        stack_index=None,
        stack_name=None,
        component_name=None,
        web_index=None,
    )
    if parts and parts[0].startswith("Station"):
        parsed["station"] = int(parts[0].replace("Station", ""))

    layer_index = next((index for index, part in enumerate(parts) if part.startswith("layer")), None)
    if layer_index is not None:
        parsed["layer"] = int(parts[layer_index].replace("layer", ""))

    if len(parts) > 1 and parts[1] in ("HP", "LP"):
        parsed["side"] = parts[1]
        if len(parts) > 2:
            parsed["stack_index"] = parts[2]
        if layer_index is not None:
            parsed["stack_name"] = "_".join(parts[2:layer_index])
            parsed["component_name"] = "_".join(parts[3:layer_index])
        return parsed

    if len(parts) > 1 and parts[1].startswith("web"):
        parsed["web_index"] = int(parts[1].replace("web", ""))
        if len(parts) > 2 and parts[2] in ("hp", "lp"):
            parsed["side"] = parts[2].upper()
        return parsed

    return parsed


def _serialize_regions(regions):
    """Convert region dataclasses into JSON-serializable dictionaries."""

    serialized = []
    for region in regions or []:
        item = {
            "name": region.name,
            "material_name": region.material_name,
            "ply_angle": region.ply_angle,
        }
        if region.laminate_name is not None:
            item["laminate_name"] = region.laminate_name
        if region.plies is not None:
            item["plies"] = region.plies
        if region.points is not None:
            item["points"] = region.points.tolist()
        if region.outer_points is not None:
            item["outer_points"] = region.outer_points.tolist()
        if region.inner_points is not None:
            item["inner_points"] = region.inner_points.tolist()
        if region.start_connector is not None:
            item["start_connector"] = region.start_connector.tolist()
        if region.end_connector is not None:
            item["end_connector"] = region.end_connector.tolist()
        if region.edge_points is not None:
            item["edge_points"] = [points.tolist() for points in region.edge_points]
        if region.edge_kinds is not None:
            item["edge_kinds"] = region.edge_kinds
        serialized.append(item)
    return serialized


def _freecad_script(payload):
    """Return a self-contained FreeCAD Python script for serialized sections."""

    data = json.dumps(payload, indent=2)
    return f"""# Generated by pynumad.analysis.freecad.make_cross_sections
import json

import FreeCAD as App
import Part


DATA = json.loads({data!r})


def vector(point):
    return App.Vector(float(point[0]), float(point[1]), float(point[2]))


def bspline_edge(points):
    if len(points) == 2:
        return Part.LineSegment(vector(points[0]), vector(points[1])).toShape()
    curve = Part.BSplineCurve()
    curve.interpolate([vector(point) for point in points])
    return curve.toShape()


def face_from_points(points):
    closed = list(points)
    closed.append(points[0])
    return Part.Face(Part.makePolygon([vector(point) for point in closed]))


def line_edges(points):
    edges = []
    for start, end in zip(points[:-1], points[1:]):
        if vector(start).distanceToPoint(vector(end)) > 1e-9:
            edges.append(Part.LineSegment(vector(start), vector(end)).toShape())
    return edges


def face_between_curves(outer_points, inner_points, start_connector=None, end_connector=None):
    outer_edge = bspline_edge(outer_points)
    if start_connector is None:
        start_connector = [inner_points[0], outer_points[0]]
    if end_connector is None:
        end_connector = [outer_points[-1], inner_points[-1]]
    if connector_is_straight(start_connector) and connector_is_straight(end_connector):
        # FreeCAD can occasionally fill long, thin spline wires with a spurious
        # rectangular face; a ruled surface keeps these strip regions bounded by
        # the intended inner and outer curves.
        inner_edge = bspline_edge(inner_points)
        try:
            return Part.makeRuledSurface(outer_edge, inner_edge)
        except Exception:
            pass

    inner_edge = bspline_edge(list(reversed(inner_points)))
    edges = [outer_edge]
    edges.extend(line_edges(end_connector))
    edges.append(inner_edge)
    edges.extend(line_edges(start_connector))
    return Part.Face(Part.Wire(edges))


def connector_is_straight(points, tolerance=1e-7):
    if len(points) <= 2:
        return True
    start = points[0]
    end = points[-1]
    segment = [end[0] - start[0], end[1] - start[1]]
    length = (segment[0] ** 2 + segment[1] ** 2) ** 0.5
    if length <= tolerance:
        return False
    for point in points[1:-1]:
        offset = [point[0] - start[0], point[1] - start[1]]
        distance = abs(segment[0] * offset[1] - segment[1] * offset[0]) / length
        if distance > tolerance:
            return False
    return True


def face_from_edge_points(edge_points, edge_kinds):
    edges = []
    for points, kind in zip(edge_points, edge_kinds):
        if kind == "line":
            edges.extend(line_edges(points))
        else:
            edges.append(bspline_edge(points))
    return Part.Face(Part.Wire(edges))


def face_from_region(region):
    if "points" in region:
        return face_from_points(region["points"])
    if "edge_points" in region:
        return face_from_edge_points(region["edge_points"], region["edge_kinds"])
    return face_between_curves(
        region["outer_points"],
        region["inner_points"],
        region.get("start_connector"),
        region.get("end_connector"),
    )


def stitched_section_shape(faces):
    shape = Part.makeCompound(faces)
    shape.sewShape()
    return shape


def regions_in_shape_face_order(regions, source_faces, stitched_shape):
    ordered_regions, messages = regions_in_shape_face_order_with_messages(regions, source_faces, stitched_shape)
    return ordered_regions


def regions_in_shape_face_order_with_messages(regions, source_faces, stitched_shape, station=None):
    shape_faces = list(getattr(stitched_shape, "Faces", []) or [])
    regions = list(regions or [])
    source_faces = list(source_faces or [])
    messages = []
    if len(shape_faces) != len(regions) or len(source_faces) != len(regions):
        messages.append(
            generation_message(
                "error",
                "face_count_mismatch",
                (
                    "FaceMaterialMap face-order verification failed because the generated region count, "
                    "source face count, and stitched FreeCAD face count do not match. The map was written "
                    "in region-generation order and may not match FreeCAD Face indices."
                ),
                station=station,
                source="freecad_cross_sections.face_material_map",
                region_count=len(regions),
                source_face_count=len(source_faces),
                stitched_face_count=len(shape_faces),
            )
        )
        return regions, messages

    source_signatures = [face_signature(face) for face in source_faces]
    shape_signatures = [face_signature(face) for face in shape_faces]
    if any(signature is None for signature in source_signatures + shape_signatures):
        messages.append(
            generation_message(
                "warning",
                "face_signature_unavailable",
                (
                    "FaceMaterialMap face-order verification was skipped because at least one FreeCAD "
                    "face did not expose area and center-of-mass data. The map was written in "
                    "region-generation order and should be checked before assigning materials."
                ),
                station=station,
                source="freecad_cross_sections.face_material_map",
                region_count=len(regions),
                source_face_count=len(source_faces),
                stitched_face_count=len(shape_faces),
            )
        )
        return regions, messages

    unused = set(range(len(source_signatures)))
    ordered = []
    for shape_signature in shape_signatures:
        best_index = min(
            unused,
            key=lambda index: face_signature_distance(shape_signature, source_signatures[index]),
        )
        unused.remove(best_index)
        ordered.append(regions[best_index])
    return ordered, messages


def generation_message(severity, code, message, station=None, source=None, **details):
    item = {{
        "severity": severity,
        "code": code,
        "message": message,
    }}
    if station is not None:
        item["station"] = station
    if source is not None:
        item["source"] = source
    if details:
        item["details"] = details
    return item


def shell_laminate_vertex_contact_messages(regions, tolerance=1e-8, thickness_ratio=5.0):
    messages = []
    indexed_regions = list(enumerate(regions or []))
    for first_pos, (first_index, first) in enumerate(indexed_regions):
        first_thickness = region_laminate_thickness(first)
        if first_thickness is None:
            continue
        first_parsed = parsed_region_name(first["name"])
        if first_parsed.get("side") not in ("HP", "LP") or first_parsed.get("web_index") is not None:
            continue
        if "_to_" in first["name"]:
            continue
        first_polygon = region_boundary_points(first)
        if first_polygon is None:
            continue
        for second_index, second in indexed_regions[first_pos + 1:]:
            second_thickness = region_laminate_thickness(second)
            if second_thickness is None:
                continue
            second_parsed = parsed_region_name(second["name"])
            if second_parsed.get("side") != first_parsed.get("side") or second_parsed.get("web_index") is not None:
                continue
            if "_to_" in second["name"]:
                continue
            if "SPAR" not in first["name"].upper() and "SPAR" not in second["name"].upper():
                continue
            second_polygon = region_boundary_points(second)
            if second_polygon is None:
                continue
            common_vertices = common_boundary_vertices(first_polygon, second_polygon, tolerance)
            if not common_vertices:
                continue
            if regions_share_boundary_edge(first_polygon, second_polygon, tolerance):
                continue
            if shell_component_adhesive_covers_vertices(indexed_regions, common_vertices, tolerance):
                continue
            thick = max(first_thickness, second_thickness)
            thin = min(first_thickness, second_thickness)
            if thin <= 0 or thick / thin < thickness_ratio:
                continue
            messages.append(
                generation_message(
                    "warning",
                    "shell_laminate_vertex_contact",
                    (
                        "Two shell laminate regions with a large thickness mismatch meet only at a vertex. "
                        "This can create an unmeshable interface; consider adding an adhesive or transition "
                        "region at this shell component boundary."
                    ),
                    station=first_parsed.get("station"),
                    source="freecad_cross_sections.shell_interfaces",
                    first_face_index=first_index,
                    first_region_name=first["name"],
                    first_material_name=first["material_name"],
                    first_thickness=first_thickness,
                    second_face_index=second_index,
                    second_region_name=second["name"],
                    second_material_name=second["material_name"],
                    second_thickness=second_thickness,
                    common_vertices=common_vertices,
                )
            )
    return messages


def shell_component_adhesive_covers_vertices(indexed_regions, vertices, tolerance):
    for _, region in indexed_regions:
        name = region["name"]
        if "_to_" not in name or "adhesive" not in name.lower():
            continue
        polygon = region_boundary_points(region)
        if polygon is None:
            continue
        if any(any(point_distance_2d(vertex, point) <= tolerance for point in polygon) for vertex in vertices):
            return True
    return False


def region_laminate_thickness(region):
    plies = region.get("plies") or []
    if not plies:
        return None
    return sum(float(ply.get("thickness", 0.0) or 0.0) for ply in plies)


def region_boundary_points(region):
    if region.get("edge_points") is not None:
        points = []
        for edge in region["edge_points"]:
            points.extend(edge)
        return clean_boundary_points(points)
    if region.get("outer_points") is None or region.get("inner_points") is None:
        return None
    points = list(region["outer_points"])
    if region.get("end_connector") is not None:
        points.extend(region["end_connector"])
    points.extend(list(reversed(region["inner_points"])))
    if region.get("start_connector") is not None:
        points.extend(region["start_connector"])
    return clean_boundary_points(points)


def clean_boundary_points(points, tolerance=1e-12):
    cleaned = []
    for point in points:
        point = [float(point[0]), float(point[1]), float(point[2])]
        if cleaned and point_distance(point, cleaned[-1]) <= tolerance:
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and point_distance(cleaned[0], cleaned[-1]) <= tolerance:
        cleaned.pop()
    return cleaned


def point_distance(first, second):
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 + (first[2] - second[2]) ** 2) ** 0.5


def point_distance_2d(first, second):
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def common_boundary_vertices(first_points, second_points, tolerance):
    common = []
    for first in first_points:
        if any(point_distance_2d(first, second) <= tolerance for second in second_points):
            if not any(point_distance_2d(first, existing) <= tolerance for existing in common):
                common.append(first)
    return common


def regions_share_boundary_edge(first_points, second_points, tolerance):
    for first_start, first_end in zip(first_points, first_points[1:] + first_points[:1]):
        if point_distance_2d(first_start, first_end) <= tolerance:
            continue
        for second_start, second_end in zip(second_points, second_points[1:] + second_points[:1]):
            if point_distance_2d(second_start, second_end) <= tolerance:
                continue
            if (
                point_distance_2d(first_start, second_end) <= tolerance
                and point_distance_2d(first_end, second_start) <= tolerance
            ) or (
                point_distance_2d(first_start, second_start) <= tolerance
                and point_distance_2d(first_end, second_end) <= tolerance
            ):
                return True
    return False


def set_turbine_messages(metadata_obj, messages):
    warnings = [message for message in messages if message.get("severity") == "warning"]
    errors = [message for message in messages if message.get("severity") == "error"]
    if "WarningMessages" not in getattr(metadata_obj, "PropertiesList", []):
        metadata_obj.addProperty("App::PropertyString", "WarningMessages", "Turbine", "JSON warning messages from pyNuMAD FreeCAD section generation")
    metadata_obj.WarningMessages = json.dumps(warnings)
    if "ErrorMessages" not in getattr(metadata_obj, "PropertiesList", []):
        metadata_obj.addProperty("App::PropertyString", "ErrorMessages", "Turbine", "JSON error messages from pyNuMAD FreeCAD section generation")
    metadata_obj.ErrorMessages = json.dumps(errors)


def face_signature(face):
    try:
        center = getattr(face, "CenterOfMass")
        return (
            float(getattr(face, "Area")),
            float(center.x),
            float(center.y),
            float(center.z),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def face_signature_distance(first, second):
    area_scale = max(abs(first[0]), abs(second[0]), 1.0)
    area_error = abs(first[0] - second[0]) / area_scale
    center_error = sum((first[index] - second[index]) ** 2 for index in range(1, 4)) ** 0.5
    return area_error + center_error


def parsed_region_name(name):
    parts = name.split("_")
    parsed = dict(
        station=None,
        layer=None,
        side=None,
        stack_index=None,
        stack_name=None,
        component_name=None,
        web_index=None,
    )
    if parts and parts[0].startswith("Station"):
        parsed["station"] = int(parts[0].replace("Station", ""))

    layer_index = next((index for index, part in enumerate(parts) if part.startswith("layer")), None)
    if layer_index is not None:
        parsed["layer"] = int(parts[layer_index].replace("layer", ""))

    if len(parts) > 1 and parts[1] in ("HP", "LP"):
        parsed["side"] = parts[1]
        if len(parts) > 2:
            parsed["stack_index"] = parts[2]
        if layer_index is not None:
            parsed["stack_name"] = "_".join(parts[2:layer_index])
            parsed["component_name"] = "_".join(parts[3:layer_index])
        return parsed

    if len(parts) > 1 and parts[1].startswith("web"):
        parsed["web_index"] = int(parts[1].replace("web", ""))
        if len(parts) > 2 and parts[2] in ("hp", "lp"):
            parsed["side"] = parts[2].upper()
        return parsed

    return parsed


def face_metadata(regions, laminate_table, material_table):
    metadata = []
    laminate_index_by_key = {{
        laminate_key(item["plies"]): item["laminate_index"]
        for item in laminate_table
    }}
    material_index_by_name = {{
        item["material_name"]: item.get("material_index")
        for item in material_table
    }}
    for index, region in enumerate(regions):
        plies = region.get("plies", [])
        laminate_index = laminate_index_by_key.get(laminate_key(plies)) if plies else None
        assignment_type = "laminate" if laminate_index is not None else "material"
        assignment_index = laminate_index if laminate_index is not None else material_index_by_name.get(region["material_name"])
        assignment_name = laminate_table[laminate_index]["laminate_name"] if laminate_index is not None else region["material_name"]
        item = parsed_region_name(region["name"])
        item.update(
            dict(
                face_index=index,
                region_name=region["name"],
                material_name=region["material_name"],
                assignment_type=assignment_type,
                assignment_index=assignment_index,
                assignment_name=assignment_name,
            )
        )
        metadata.append(item)
    return metadata


def laminate_definitions(regions):
    definitions = []
    index_by_key = {{}}
    for region in regions:
        plies = region.get("plies", [])
        if not plies:
            continue
        key = laminate_key(plies)
        if key in index_by_key:
            continue
        laminate_index = len(definitions)
        index_by_key[key] = laminate_index
        definitions.append(
            {{
                "laminate_index": laminate_index,
                "laminate_name": "Laminate{{:03d}}".format(laminate_index),
                "plies": plies,
            }}
        )
    return definitions


def laminate_key(plies):
    return json.dumps(plies, sort_keys=True, separators=(",", ":"))


def color_for_material(material_name):
    name = material_name.lower()
    if "adhesive" in name:
        return (1.0, 0.84, 0.0, 0.0)
    if "carbon" in name:
        return (0.18, 0.18, 0.18, 0.0)
    if "foam" in name:
        return (0.84, 0.78, 0.52, 0.0)
    if "web" in name:
        return (0.2, 0.5, 0.85, 0.0)
    if "spar" in name or "uni" in name:
        return (0.22, 0.55, 0.32, 0.0)
    if "triax" in name or "biax" in name:
        return (0.56, 0.8, 0.36, 0.0)
    return (0.68, 0.72, 0.78, 0.0)


doc = App.newDocument(DATA["wt_name"] + "_cross_sections")
created = []
generation_messages = []

if DATA["detailed"]:
    metadata_obj = doc.addObject("App::FeaturePython", "TurbineMetadata")
    metadata_obj.Label = "Turbine Metadata"
    metadata_obj.addProperty("App::PropertyString", "MaterialDefinitions", "Turbine", "JSON turbine-level material property table from pyNuMAD")
    metadata_obj.MaterialDefinitions = json.dumps(DATA["material_definitions"])
    metadata_obj.addProperty("App::PropertyString", "LaminateDefinitions", "Turbine", "JSON turbine-level laminate ply stack table")
    metadata_obj.LaminateDefinitions = json.dumps(DATA["laminate_definitions"])
    metadata_obj.addProperty("App::PropertyInteger", "StationCount", "Turbine", "Number of blade stations in the YAML")
    metadata_obj.StationCount = int(DATA["station_count"])
    set_turbine_messages(metadata_obj, generation_messages)

for section in DATA["sections"]:
    station = section["station"]
    hp_points = section["hp"]
    lp_points = section["lp"]

    if DATA["detailed"]:
        section_faces = []
        for region in section["regions"]:
            face_shape = face_from_region(region)
            section_faces.append(face_shape)
            if DATA["debug_faces"]:
                face_obj = doc.addObject("Part::Feature", region["name"])
                face_obj.Shape = face_shape
                face_obj.Label = region["name"] + " | " + region["material_name"] + " | angle " + str(region["ply_angle"])
                if hasattr(face_obj, "ViewObject") and face_obj.ViewObject is not None:
                    face_obj.ViewObject.ShapeColor = color_for_material(region["material_name"])

        stitched_obj = doc.addObject("Part::Feature", "Station{{:03d}}_section".format(station))
        stitched_obj.Shape = stitched_section_shape(section_faces)
        stitched_obj.Label = "Station {{:03d}}".format(station)
        face_ordered_regions, face_map_messages = regions_in_shape_face_order_with_messages(section["regions"], section_faces, stitched_obj.Shape, station=station)
        face_map_messages.extend(shell_laminate_vertex_contact_messages(face_ordered_regions))
        generation_messages.extend(face_map_messages)
        stitched_obj.addProperty("App::PropertyString", "FaceMaterialMap", "Turbine", "JSON map from face index to material metadata")
        stitched_obj.FaceMaterialMap = json.dumps(face_metadata(face_ordered_regions, DATA["laminate_definitions"], DATA["material_definitions"]))
        stitched_obj.addProperty("App::PropertyString", "StationFrame", "Turbine", "JSON station reference-axis origin, rotations, and local coordinate system")
        stitched_obj.StationFrame = json.dumps(section.get("station_frame", {{}}))
        if hasattr(stitched_obj, "ViewObject") and stitched_obj.ViewObject is not None:
            stitched_obj.ViewObject.ShapeColor = (0.78, 0.82, 0.86, 0.0)
        created.append(stitched_obj)

        if DATA["debug_faces"]:
            hp_obj = doc.addObject("Part::Feature", "Station{{:03d}}_outer_HP".format(station))
            hp_obj.Shape = bspline_edge(hp_points)
            lp_obj = doc.addObject("Part::Feature", "Station{{:03d}}_outer_LP".format(station))
            lp_obj.Shape = bspline_edge(lp_points)
    else:
        hp_edge = bspline_edge(hp_points)
        lp_edge = bspline_edge(list(reversed(lp_points)))
        te_edge = Part.LineSegment(vector(lp_points[0]), vector(hp_points[0])).toShape()
        wire = Part.Wire([hp_edge, lp_edge, te_edge])

        wire_obj = doc.addObject("Part::Feature", "Station{{:03d}}_wire".format(station))
        wire_obj.Shape = wire
        wire_obj.addProperty("App::PropertyString", "StationFrame", "Turbine", "JSON station reference-axis origin, rotations, and local coordinate system")
        wire_obj.StationFrame = json.dumps(section.get("station_frame", {{}}))
        created.append(wire_obj)

        hp_obj = doc.addObject("Part::Feature", "Station{{:03d}}_HP".format(station))
        hp_obj.Shape = hp_edge
        lp_obj = doc.addObject("Part::Feature", "Station{{:03d}}_LP".format(station))
        lp_obj.Shape = lp_edge
        te_obj = doc.addObject("Part::Feature", "Station{{:03d}}_TE".format(station))
        te_obj.Shape = te_edge

        if DATA["make_faces"]:
            face_obj = doc.addObject("Part::Feature", "Station{{:03d}}_face".format(station))
            face_obj.Shape = Part.Face(wire)
            created.append(face_obj)

if DATA["detailed"]:
    set_turbine_messages(metadata_obj, generation_messages)

doc.recompute()

if DATA["export_step"]:
    Part.export(created, DATA["step_path"])

doc.saveAs(DATA["fcstd_path"])
"""
