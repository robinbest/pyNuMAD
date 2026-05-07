"""FreeCAD export helpers for pyNuMAD blade cross sections.

This module intentionally covers only the outer 2D cross-section shape.  It is
not a drop-in replacement for the Cubit meshing workflow in
``pynumad.analysis.cubit``.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np


@dataclass
class FreeCADCrossSection:
    """Station-local blade section data arranged for FreeCAD curve creation."""

    station: int
    te_point: np.ndarray
    hp_points: np.ndarray
    lp_points: np.ndarray

    @property
    def closed_points(self):
        """Return points ordered around the airfoil perimeter."""

        return np.vstack((self.te_point, self.hp_points, np.flip(self.lp_points[:-1], axis=0)))


@dataclass
class FreeCADFaceRegion:
    """A named cross-section face region for the generated FreeCAD script."""

    name: str
    material_name: str
    ply_angle: float
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
    return FreeCADCrossSection(station=station, te_point=xyz[0, :], hp_points=hp_points, lp_points=lp_points)


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
        regions=regions,
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
        "sections": [
            {
                "station": section.station,
                "hp": section.hp_points.tolist(),
                "lp": section.lp_points.tolist(),
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

    section_builder = get_detailed_cross_section if detailed else get_cross_section
    created = []
    for station in station_list:
        kwargs = {
            "geometry_scaling": geometry_scaling,
            "normalize_chord": normalize_chord,
            "move_le_to_origin": move_le_to_origin,
        }
        if detailed:
            kwargs["cs_params"] = cs_params
        section = section_builder(blade, station, **kwargs)
        created.append(make_freecad_section_part(section, doc=doc, debug_faces=debug_faces))

    doc.recompute()
    return created


def make_freecad_section_part(section, *, doc=None, name=None, debug_faces=False):
    """Create one FreeCAD object from a pyNuMAD cross-section data object.

    Detailed sections become a sewn shell with one face per material region.
    The returned object gets a ``FaceMaterialMap`` JSON string property whose
    entries are ordered to match ``obj.Shape.Faces``.
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
        section_obj.Label = f"Station {section.station:03d} stitched section"
        _set_string_property(
            section_obj,
            "FaceMaterialMap",
            json.dumps(face_material_metadata(section.regions)),
            group="pyNuMAD",
            description="JSON map from face index to material metadata",
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
    return wire_obj


def face_material_metadata(regions):
    """Return face-index/material metadata for serialized or object regions."""

    metadata = []
    for index, region in enumerate(regions or []):
        region_name = _region_value(region, "name")
        item = _parsed_region_name(region_name)
        item.update(
            dict(
                face_index=index,
                region_name=region_name,
                material_name=_region_value(region, "material_name"),
                ply_angle=_region_value(region, "ply_angle"),
            )
        )
        metadata.append(item)
    return metadata


class _StationTransformer:
    def __init__(
        self,
        blade,
        station,
        *,
        geometry_scaling,
        normalize_chord,
        move_le_to_origin,
    ):
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
        xyz = np.array(points, dtype=float) * self.geometry_scaling
        if self.normalize_chord:
            xyz = xyz / (self.chord * self.geometry_scaling)
        if self.move_le_to_origin:
            xyz = xyz - (self.le / (self.chord * self.geometry_scaling) if self.normalize_chord else self.le)
        return xyz

    def length_from_m(self, length_m):
        length = length_m * self.geometry_scaling
        if self.normalize_chord:
            length = length / (self.chord * self.geometry_scaling)
        return length

    def length_from_mm(self, length_mm):
        return self.length_from_m(0.001 * length_mm)


def _shell_regions(blade, station, section, transformer, cs_params):
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
    stacks, sides, segments, trailing_edge = _trim_trailing_edge_segments(
        stacks, sides, segments, station, transformer, cs_params
    )
    regions = _perimeter_shell_regions(stacks, sides, segments, station, section, transformer)
    regions.extend(_trailing_edge_adhesive_regions(station, trailing_edge, regions, cs_params))
    return regions


def _clamp_le_surface_protrusion(hp_points, lp_points, te_point, le_point, tolerance=1e-9):
    if abs(le_point[0] - te_point[0]) <= tolerance:
        return

    le_is_x_maximum = le_point[0] > te_point[0]
    _clamp_trailing_points_to_le_x(hp_points, le_point[0], le_is_x_maximum, tolerance)
    _clamp_trailing_points_to_le_x(lp_points, le_point[0], le_is_x_maximum, tolerance)
    hp_points[-1] = le_point
    lp_points[-1] = le_point


def _clamp_trailing_points_to_le_x(points, le_x, le_is_x_maximum, tolerance):
    for i_point in reversed(range(len(points))):
        excess = points[i_point, 0] - le_x if le_is_x_maximum else le_x - points[i_point, 0]
        if excess <= tolerance:
            if i_point != len(points) - 1:
                break
            continue
        points[i_point, 0] = le_x


def _shell_segments(blade, station, section, transformer):
    keypoints = transformer.points(blade.keypoints.key_points[:, :, station])
    hp_boundaries = np.vstack((section.hp_points[0], keypoints[0:5], section.hp_points[-1]))
    lp_boundaries = np.vstack((section.lp_points[-1], keypoints[5:10], section.lp_points[0]))

    hp_segments = _split_polyline_at_points(section.hp_points, hp_boundaries)
    lp_segments = _split_polyline_at_points(np.flip(section.lp_points, axis=0), lp_boundaries)
    return hp_segments, lp_segments


def _remove_zero_length_shell_segments(stacks, sides, segments):
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


def _trailing_edge_split_width(hp_stacks, hp_segments, lp_stacks, lp_segments, transformer, cs_params, station):
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


def _point_at_path_distance(segments, distance):
    remaining = distance
    for segment in segments:
        length = _polyline_lengths(segment)[-1]
        if remaining <= length:
            return _point_at_distance(segment, remaining)
        remaining -= length
    return segments[-1][-1]


def _point_at_reversed_path_distance(segments, distance):
    remaining = distance
    for segment in reversed(segments):
        length = _polyline_lengths(segment)[-1]
        if remaining <= length:
            return _point_at_distance(segment, length - remaining)
        remaining -= length
    return segments[0][0]


def _trailing_edge_target_gap(hp_stack, lp_stack, transformer, cs_params, station, initial_gap):
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


def _perimeter_shell_regions(stacks, sides, segments, station, section, transformer):
    current_segments = [_clean_polyline(segment) for segment in segments]
    stack_name_counts = {
        (side, stack.name): sum(1 for other_side, other_stack in zip(sides, stacks) if other_side == side and other_stack.name == stack.name)
        for side, stack in zip(sides, stacks)
    }
    max_layers = max((len(stack.plygroups) for stack in stacks), default=0)
    regions = []

    for i_layer in range(max_layers):
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
            regions.append(
                FreeCADFaceRegion(
                    name=f"Station{station:03d}_{sides[i_segment]}_{_shell_region_stack_name(stack, sides, stack_name_counts, i_segment)}_layer{i_layer:02d}",
                    material_name=plygroup.materialid,
                    ply_angle=plygroup.angle,
                    outer_points=outer_segment,
                    inner_points=inner_segment,
                    start_connector=start_connector,
                    end_connector=end_connector,
                )
            )
            current_segments[i_segment] = inner_segment

    return regions


def _shell_region_stack_name(stack, sides, stack_name_counts, i_segment):
    name = stack.name
    if stack_name_counts[(sides[i_segment], stack.name)] <= 1:
        return name
    if i_segment == 0 or i_segment == len(sides) - 1:
        return name + "_TE"
    return name


def _trailing_edge_adhesive_regions(station, trailing_edge, shell_regions, cs_params):
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


def _shell_cut_connector(shell_regions, outer_point, connector_end, side=None, tolerance=1e-8):
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
        if np.linalg.norm(region_outer_point - points[-1]) > tolerance:
            continue
        connector = region.start_connector if connector_end == "start" else region.end_connector
        ordered = np.flip(connector, axis=0) if connector_end == "start" else connector
        if np.linalg.norm(ordered[0] - points[-1]) > tolerance:
            ordered = np.flip(ordered, axis=0)
        points.extend(ordered[1:])
    return np.array(points)


def _combine_connected_segments(segments):
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
    offset_curves = {0.0: points}
    for thickness in sorted(set(thicknesses)):
        if thickness > 0:
            offset_curves[thickness] = _offset_open_polyline_inward(points, closed_points, thickness)
    return offset_curves


def _close_matching_curve_endpoints(offset_curves):
    if not _is_closed_polyline(offset_curves[0.0]):
        return
    for thickness, points in offset_curves.items():
        if thickness <= 0:
            continue
        intersection = _curve_end_intersection(points, "start", points, "end")
        points[0] = intersection
        points[-1] = intersection


def _is_closed_polyline(points, tolerance=1e-9):
    return len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= tolerance


def _curve_end_intersection(first_points, first_end, second_points, second_end):
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
    if end == "start":
        return points[0], points[1]
    if end == "end":
        return points[-1], points[-2]
    raise ValueError(f"Unknown curve end: {end}")


def _square_stair_step_boundaries(offset_curves, thicknesses, segment_slices, current_segments, closed=False):
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
    for i_segment, (start, end) in enumerate(segment_slices):
        if i_segment > 0 and np.linalg.norm(current_segments[i_segment - 1][-1] - current_segments[i_segment][0]) > tolerance:
            _square_offset_curve_endpoint(offset_curves, current_segments[i_segment], start, "start")
        if i_segment < len(segment_slices) - 1 and np.linalg.norm(current_segments[i_segment][-1] - current_segments[i_segment + 1][0]) > tolerance:
            _square_offset_curve_endpoint(offset_curves, current_segments[i_segment], end - 1, "end")


def _square_offset_curve_endpoint(offset_curves, outer_segment, boundary_index, boundary_end):
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
    marker = "_layer"
    if marker not in name:
        return None
    try:
        return int(name.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _merged_trailing_edge_region(hp_region, lp_region):
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
                outer_points=current_hp_outer,
                inner_points=current_hp_inner,
            )
        )
        regions.append(
            FreeCADFaceRegion(
                name=f"Station{station:03d}_LP_{lp_stack.name}_layer{i_layer:02d}",
                material_name=lp_plygroup.materialid,
                ply_angle=lp_plygroup.angle,
                outer_points=current_lp_outer,
                inner_points=current_lp_inner,
            )
        )
        current_hp_outer = current_hp_inner
        current_lp_outer = current_lp_inner

    return regions


def _web_regions(blade, station, transformer, cs_params, shell_regions):
    stackdb = blade.stackdb
    if stackdb.swstacks is None:
        return []

    regions = []
    hp_spar_region = _innermost_spar_region(shell_regions, blade, station, "HP", 3)
    lp_spar_region = _innermost_spar_region(shell_regions, blade, station, "LP", 8)
    if hp_spar_region is None or lp_spar_region is None:
        return regions

    hp_interface_edges = []
    lp_interface_edges = []
    stack_station = min(station, stackdb.swstacks.shape[1] - 1)
    for i_web in range(stackdb.swstacks.shape[0]):
        if i_web >= stackdb.swstacks.shape[0]:
            break
        web_stack = stackdb.swstacks[i_web, stack_station]
        if not web_stack.plygroups:
            continue

        web_thickness = transformer.length_from_mm(sum(web_stack.layer_thicknesses()))
        if web_thickness <= 0:
            continue

        interfaces = _spar_web_interfaces(
            hp_spar_region.inner_points,
            lp_spar_region.inner_points,
            station,
            transformer,
            cs_params,
            i_web,
            web_stack,
        )
        if interfaces is None:
            continue

        hp_edges, lp_edges = interfaces
        lp_edges = list(reversed(lp_edges))
        hp_adhesive_edges, hp_web_edges = _web_connection_edges(hp_edges, lp_edges, transformer, cs_params, station, i_web)
        lp_adhesive_edges, lp_web_edges = _web_connection_edges(lp_edges, hp_edges, transformer, cs_params, station, i_web)

        hp_interface_edges.append(_join_connected_edges(_connected_edge_order(hp_adhesive_edges, hp_web_edges)[0]))
        lp_interface_edges.append(_join_connected_edges(_connected_edge_order(lp_adhesive_edges, lp_web_edges)[0]))

        regions.extend(_web_laminate_regions(station, i_web, hp_web_edges, lp_web_edges, web_stack))
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

    _split_region_inner_edge_for_interfaces(hp_spar_region, hp_interface_edges)
    _split_region_inner_edge_for_interfaces(lp_spar_region, lp_interface_edges)
    return regions


def _innermost_spar_region(shell_regions, blade, station, side, stack_index):
    stackdb = blade.stackdb
    stack_station = min(station, stackdb.stacks.shape[1] - 1)
    stack_name = stackdb.stacks[stack_index, stack_station].name
    prefix = f"Station{station:03d}_{side}_{stack_name}_layer"
    candidates = [region for region in shell_regions if region.name.startswith(prefix) and region.inner_points is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda region: _layer_index_from_name(region.name) or 0)


def _spar_web_interfaces(hp_inner_spar, lp_inner_spar, station, transformer, cs_params, i_web, web_stack):
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

    if i_web == 0:
        hp_center = hp_length - inset
        lp_center = inset
    else:
        hp_center = inset
        lp_center = lp_length - inset

    hp_intervals = _centered_intervals(hp_length, hp_center, layer_widths)
    lp_intervals = _centered_intervals(lp_length, lp_center, layer_widths)
    if not hp_intervals or not lp_intervals:
        return None

    hp_edges = [_polyline_between(hp_inner_spar, start, end) for start, end in hp_intervals]
    lp_edges = [_polyline_between(lp_inner_spar, start, end) for start, end in lp_intervals]
    return hp_edges, lp_edges


def _centered_intervals(length, center, widths):
    widths = [width for width in widths if width > 0]
    total_width = sum(widths)
    if total_width <= 0 or length <= 1e-9:
        return []
    if total_width > length:
        scale = 0.95 * length / total_width
        widths = [width * scale for width in widths]
        total_width = sum(widths)

    start = center - 0.5 * total_width
    start = float(np.clip(start, 0.0, max(length - total_width, 0.0)))
    intervals = []
    for width in widths:
        end = start + width
        if end - start > 1e-9:
            intervals.append((start, end))
        start = end
    return intervals


def _web_interface_centerline(hp_edges, lp_edges):
    hp_points = np.vstack((hp_edges[0][0], hp_edges[-1][-1]))
    lp_points = np.vstack((lp_edges[0][0], lp_edges[-1][-1]))
    return np.vstack((hp_points.mean(axis=0), lp_points.mean(axis=0)))


def _web_connection_edges(interface_edges, opposite_edges, transformer, cs_params, station, i_web):
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
    offset_points = []
    for point in edge:
        direction = _unit(target - point)
        offset_points.append(point + direction * distance)
    return np.array(offset_points)


def _split_region_inner_edge_for_interfaces(region, interface_edges):
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


def _web_laminate_regions(station, i_web, hp_edges, lp_edges, web_stack):
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
                edge_points=_web_face_edges(hp_edge, lp_edge),
                edge_kinds=["spline", "line", "spline", "line"],
            )
        )
        i_valid_layer += 1
    return regions


def _web_adhesive_regions_from_edges(station, i_web, hp_outer_edges, hp_inner_edges, lp_outer_edges, lp_inner_edges, cs_params):
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
    points = []
    for edge in edges:
        if not points:
            points.extend(edge)
        else:
            points.extend(edge[1:])
    return np.array(points)


def _connected_edge_order(outer_edges, inner_edges):
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
    if not edge_points:
        return 0.0
    return max(
        np.linalg.norm(first[-1] - second[0])
        for first, second in zip(edge_points, edge_points[1:] + edge_points[:1])
    )


def _edge_boundary_self_intersects(edge_points):
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
    first_orientation = _orientation_2d(first_start, first_end, second_start)
    second_orientation = _orientation_2d(first_start, first_end, second_end)
    third_orientation = _orientation_2d(second_start, second_end, first_start)
    fourth_orientation = _orientation_2d(second_start, second_end, first_end)
    return first_orientation * second_orientation < -1e-12 and third_orientation * fourth_orientation < -1e-12


def _orientation_2d(start, end, point):
    return np.cross(end[:2] - start[:2], point[:2] - start[:2])


def _split_polyline_at_points(points, boundary_points):
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
    distances = [0.0]
    for start, end in zip(points[:-1], points[1:]):
        distances.append(distances[-1] + np.linalg.norm(end - start))
    return np.array(distances)


def _clean_polyline(points, min_distance=1e-3):
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
    if len(outer_points) < 2 or len(inner_points) < 2:
        return False
    return _polyline_lengths(outer_points)[-1] > tolerance and _polyline_lengths(inner_points)[-1] > tolerance


def _project_point_to_polyline(points, point):
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


def _point_at_distance(points, distance):
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
    cumulative = _polyline_lengths(points)
    selected = [_point_at_distance(points, start_distance)]
    for i, distance in enumerate(cumulative[1:-1], start=1):
        if start_distance < distance < end_distance:
            selected.append(points[i])
    selected.append(_point_at_distance(points, end_distance))
    return np.array(selected)


def _offset_open_polyline_inward(points, closed_points, distance, miter_limit=4.0):
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
    xy = np.asarray(points)[:, :2]
    x = xy[:, 0]
    y = xy[:, 1]
    return 1.0 if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) >= 0 else -1.0


def _rectangle_about_line(start, end, width):
    tangent = _unit(end - start)
    normal = _perp(tangent)
    half_width = 0.5 * width
    return np.vstack((start - normal * half_width, end - normal * half_width, end + normal * half_width, start + normal * half_width))


def _perp(vector):
    return np.array([-vector[1], vector[0], 0.0])


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _station_value(values, station, default=0.0):
    if values is None:
        return default
    try:
        return float(values[station])
    except (TypeError, IndexError):
        return float(values)


def _require_freecad_modules():
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
    return App.Vector(float(point[0]), float(point[1]), float(point[2]))


def _freecad_bspline_edge(points, App, Part):
    if len(points) == 2:
        return Part.LineSegment(_freecad_vector(points[0], App), _freecad_vector(points[1], App)).toShape()
    curve = Part.BSplineCurve()
    curve.interpolate([_freecad_vector(point, App) for point in points])
    return curve.toShape()


def _freecad_line_edges(points, App, Part):
    edges = []
    for start, end in zip(points[:-1], points[1:]):
        if _freecad_vector(start, App).distanceToPoint(_freecad_vector(end, App)) > 1e-9:
            edges.append(Part.LineSegment(_freecad_vector(start, App), _freecad_vector(end, App)).toShape())
    return edges


def _freecad_face_from_points(points, App, Part):
    closed = list(points)
    closed.append(points[0])
    return Part.Face(Part.makePolygon([_freecad_vector(point, App) for point in closed]))


def _freecad_face_between_curves(outer_points, inner_points, App, Part, start_connector=None, end_connector=None):
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
    edges = []
    for points, kind in zip(edge_points, edge_kinds):
        if kind == "line":
            edges.extend(_freecad_line_edges(points, App, Part))
        else:
            edges.append(_freecad_bspline_edge(points, App, Part))
    return Part.Face(Part.Wire(edges))


def _freecad_face_from_region(region, App, Part):
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
    shape = Part.makeCompound(faces)
    shape.sewShape()
    return shape


def _set_string_property(obj, name, value, group="", description=""):
    if name not in getattr(obj, "PropertiesList", []):
        obj.addProperty("App::PropertyString", name, group, description)
    setattr(obj, name, value)


def _set_view_color(obj, color):
    if hasattr(obj, "ViewObject") and obj.ViewObject is not None:
        obj.ViewObject.ShapeColor = color


def _color_for_material(material_name):
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
    if isinstance(region, dict):
        return region.get(key)
    return getattr(region, key)


def _parsed_region_name(name):
    parts = name.split("_")
    parsed = dict(RegionName=name)
    if parts and parts[0].startswith("Station"):
        parsed["Station"] = int(parts[0].replace("Station", ""))

    layer_index = next((index for index, part in enumerate(parts) if part.startswith("layer")), None)
    if layer_index is not None:
        parsed["Layer"] = int(parts[layer_index].replace("layer", ""))

    if len(parts) > 1 and parts[1] in ("HP", "LP"):
        parsed["Side"] = parts[1]
        if len(parts) > 2:
            parsed["StackIndex"] = parts[2]
        if layer_index is not None:
            parsed["StackName"] = "_".join(parts[2:layer_index])
            parsed["ComponentName"] = "_".join(parts[3:layer_index])
        return parsed

    if len(parts) > 1 and parts[1].startswith("web"):
        parsed["Feature"] = "web"
        parsed["WebIndex"] = int(parts[1].replace("web", ""))
        if len(parts) > 2 and parts[2] in ("hp", "lp"):
            parsed["Side"] = parts[2].upper()
            parsed["IsAdhesive"] = "adhesive" in parts
        return parsed

    return parsed


def _serialize_regions(regions):
    serialized = []
    for region in regions or []:
        item = {
            "name": region.name,
            "material_name": region.material_name,
            "ply_angle": region.ply_angle,
        }
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


def parsed_region_name(name):
    parts = name.split("_")
    parsed = dict(RegionName=name)
    if parts and parts[0].startswith("Station"):
        parsed["Station"] = int(parts[0].replace("Station", ""))

    layer_index = next((index for index, part in enumerate(parts) if part.startswith("layer")), None)
    if layer_index is not None:
        parsed["Layer"] = int(parts[layer_index].replace("layer", ""))

    if len(parts) > 1 and parts[1] in ("HP", "LP"):
        parsed["Side"] = parts[1]
        if len(parts) > 2:
            parsed["StackIndex"] = parts[2]
        if layer_index is not None:
            parsed["StackName"] = "_".join(parts[2:layer_index])
            parsed["ComponentName"] = "_".join(parts[3:layer_index])
        return parsed

    if len(parts) > 1 and parts[1].startswith("web"):
        parsed["Feature"] = "web"
        parsed["WebIndex"] = int(parts[1].replace("web", ""))
        if len(parts) > 2 and parts[2] in ("hp", "lp"):
            parsed["Side"] = parts[2].upper()
            parsed["IsAdhesive"] = "adhesive" in parts
        return parsed

    return parsed


def face_metadata(regions):
    metadata = []
    for index, region in enumerate(regions):
        item = parsed_region_name(region["name"])
        item.update(
            dict(
                face_index=index,
                region_name=region["name"],
                material_name=region["material_name"],
                ply_angle=region["ply_angle"],
            )
        )
        metadata.append(item)
    return metadata


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
        stitched_obj.Label = "Station {{:03d}} stitched section".format(station)
        stitched_obj.addProperty("App::PropertyString", "FaceMaterialMap", "pyNuMAD", "JSON map from face index to material metadata")
        stitched_obj.FaceMaterialMap = json.dumps(face_metadata(section["regions"]))
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

doc.recompute()

if DATA["export_step"]:
    Part.export(created, DATA["step_path"])

doc.saveAs(DATA["fcstd_path"])
"""
