import numpy as np
import subprocess
import tempfile
import trimesh
from pathlib import Path
from typing import Optional
import typer

from vis_util import (
    extract_faces,
    apply_transformations,
    offset_edges_from_surface,
    _read_step_file,
    _discretize_all_edges,
    _create_video_with_ffmpeg,
)

app = typer.Typer(pretty_exceptions_enable=False)


# =============================================================================
# Default Parameters
# =============================================================================

DEFAULT_EDGE_DEFLECTION = 0.0005
DEFAULT_MESH_LINEAR_DEFLECTION = 0.01
DEFAULT_MESH_ANGULAR_DEFLECTION = 0.1

FAST_EDGE_DEFLECTION = 0.005
FAST_MESH_LINEAR_DEFLECTION = 0.1
FAST_MESH_ANGULAR_DEFLECTION = 1.0

DEFAULT_RESOLUTION = 256
DEFAULT_COLOR_A = "blue"
DEFAULT_COLOR_B = "silver"
DEFAULT_COLOR_MODE = "rgb"
DEFAULT_ROTATION_ANGLE = 0.0

DEFAULT_CAMERA_DISTANCE = 3.0
DEFAULT_CAMERA_HEIGHT = 2.0
DEFAULT_CAMERA_HEIGHT_LOW = 0.5
DEFAULT_CAMERA_BASE_ANGLE = -45.0

DEFAULT_VIDEO_FPS = 24
DEFAULT_VIDEO_DURATION = 3.0
DEFAULT_VIDEO_FORMAT = "mp4"

GROUND_PLANE_Z = -0.5

BLENDER_EXECUTABLE = "./blender"
BLENDER_PAIRED_RENDER_SCRIPT = "render_paired_step.py"


# =============================================================================
# Processing Functions
# =============================================================================


def _transform_and_normalize(
    vertices, edges, stand_upright, flip_z, no_normalize,
    ground_plane_z=GROUND_PLANE_Z, normalization_params=None
):
    """
    Apply transformations and normalization to vertices and edges.

    Returns (vertices, edges_rescaled, norm_params).
    If normalization_params is provided, those are used instead of computing new ones.
    """
    vertices = apply_transformations(vertices, stand_upright, flip_z)
    edges_transformed = [apply_transformations(e, stand_upright, flip_z) for e in edges]

    if not no_normalize:
        if normalization_params is not None:
            center = normalization_params["center"]
            scale = normalization_params["scale"]
            z_shift = normalization_params["z_shift"]
        else:
            v_min = vertices.min(axis=0, keepdims=True)
            v_max = vertices.max(axis=0, keepdims=True)
            center = ((v_max + v_min) / 2).flatten()
            scale = float(np.max(v_max - v_min))
            vertices_norm = (vertices - center) / scale
            z_shift = vertices_norm[:, 2].min() - ground_plane_z

        if len(vertices) > 0:
            vertices = (vertices - center) / scale
            vertices[:, 2] -= z_shift
    else:
        center = np.zeros(3)
        scale = 1.0
        z_shift = 0.0

    edges_rescaled = []
    for edge in edges_transformed:
        if not no_normalize:
            edge = (edge - center) / scale
            edge[:, 2] -= z_shift
        edges_rescaled.append(edge)

    return vertices, edges_rescaled, {"center": center, "scale": scale, "z_shift": z_shift}


def _compute_normalization_params_from_a(
    shape_a,
    stand_upright, flip_z,
    mesh_linear_deflection, mesh_angular_deflection,
    ground_plane_z,
):
    """
    Compute normalization parameters from shape A only.

    Shape B (if any) is placed in the same coordinate space so that the viewpoint
    is always anchored to shape A regardless of what shape B looks like.
    """
    vertices_a, _ = extract_faces(shape_a, mesh_linear_deflection, mesh_angular_deflection)
    vertices_a = apply_transformations(vertices_a, stand_upright, flip_z)

    v_min = vertices_a.min(axis=0, keepdims=True)
    v_max = vertices_a.max(axis=0, keepdims=True)
    center = ((v_min + v_max) / 2).flatten()
    scale = float(np.max(v_max - v_min))

    vertices_norm = (vertices_a - center) / scale
    z_shift = vertices_norm[:, 2].min() - ground_plane_z

    return {"center": center, "scale": scale, "z_shift": z_shift}


def process_step_file(
    step_file: Path,
    output_dir: Path,
    edge_deflection: float,
    mesh_linear_deflection: float,
    mesh_angular_deflection: float,
    stand_upright: bool,
    flip_z: bool,
    no_normalize: bool,
    output_stem: str = None,
    ground_plane_z: float = GROUND_PLANE_Z,
    normalization_params: dict = None,
) -> Path:
    """
    Process a STEP file to NPZ format.

    If normalization_params is given, those parameters are used (shared normalization
    for paired rendering). Otherwise, parameters are computed from this file's geometry.
    Returns path to the NPZ file, or None if processing failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or step_file.stem
    brep_npz_path = output_dir / f"{stem}.npz"

    shape = _read_step_file(step_file)
    if shape is None:
        return None

    all_edges = _discretize_all_edges(shape, edge_deflection)
    vertices, triangles = extract_faces(shape, mesh_linear_deflection, mesh_angular_deflection)

    vertices, all_edges_rescaled, norm_params = _transform_and_normalize(
        vertices, all_edges, stand_upright, flip_z, no_normalize,
        ground_plane_z, normalization_params
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    all_edges_rescaled = offset_edges_from_surface(all_edges_rescaled, mesh)

    edges_flat = np.concatenate(all_edges_rescaled, axis=0) if all_edges_rescaled else np.empty((0, 3))
    edges_counts = np.array([len(e) for e in all_edges_rescaled], dtype=np.int64)
    np.savez(
        brep_npz_path,
        edges_flat=edges_flat,
        edges_counts=edges_counts,
        vertices=vertices,
        triangles=triangles,
        scale=norm_params["scale"],
        center=norm_params["center"],
    )

    return brep_npz_path


def render_blender_paired(
    npz_file_a: Path,
    npz_file_b: Optional[Path],
    output_dir: Path,
    blender_script: Path,
    color_a: str = DEFAULT_COLOR_A,
    color_b: str = DEFAULT_COLOR_B,
    rotation_angle: float = DEFAULT_ROTATION_ANGLE,
    flip_z: bool = False,
    stand_upright: bool = False,
    color_mode: str = DEFAULT_COLOR_MODE,
    resolution: int = DEFAULT_RESOLUTION,
    camera_distance: float = None,
    camera_height: float = None,
    camera_base_angle: float = None,
    output_name: str = None,
) -> Path:
    """
    Render two NPZ files together using Blender.

    Returns path to the rendered image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name:
        output_image = output_dir / output_name
    elif npz_file_b is not None:
        output_image = output_dir / f"{npz_file_a.stem}_{npz_file_b.stem}.png"
    else:
        output_image = output_dir / f"{npz_file_a.stem}.png"

    cmd = [
        BLENDER_EXECUTABLE,
        "--background",
        "--python",
        str(blender_script),
        "--",
        str(npz_file_a),
        str(npz_file_b) if npz_file_b is not None else "none",
        str(output_image),
        color_a,
        color_b,
        str(rotation_angle),
        str(resolution),
    ]

    if flip_z:
        cmd.append("flip")
    if stand_upright:
        cmd.append("stand_upright")
    cmd.append(color_mode)

    if camera_distance is not None:
        cmd.append(f"camdist={camera_distance}")
    if camera_height is not None:
        cmd.append(f"camheight={camera_height}")
    if camera_base_angle is not None:
        cmd.append(f"camangle={camera_base_angle}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    log = result.stdout + "\n" + result.stderr

    if result.returncode != 0 or "Traceback (most recent call last)" in log:
        print(f"\n[BLENDER LOG]\n{log}")
        label = f"{npz_file_a} + {npz_file_b}" if npz_file_b is not None else str(npz_file_a)
        raise RuntimeError(f"Blender failed while rendering {label}")

    return output_image


def _render_paired_video_frames(
    npz_path_a: Path,
    npz_path_b: Optional[Path],
    frames_dir: Path,
    n_frames: int,
    blender_script: Path,
    color_a: str,
    color_b: str,
    resolution: int,
    flip_z: bool,
    stand_upright: bool,
    color_mode: str,
    blender_executable: str = BLENDER_EXECUTABLE,
    camera_distance: float = None,
    camera_height: float = None,
    camera_height_low: float = None,
    camera_base_angle: float = None,
):
    """Render all frames for a paired video animation in a single Blender session.

    Passes nframes=N to render_paired_step.py, which inserts per-frame camera
    keyframes and calls bpy.ops.render.render(animation=True). This avoids
    repeated GPU/shader initialisation and is significantly faster than one
    Blender subprocess per frame.
    """
    frame_pattern = frames_dir / "frame_"

    cmd = [
        blender_executable,
        "--background",
        "--python",
        str(blender_script),
        "--",
        str(npz_path_a),
        str(npz_path_b) if npz_path_b is not None else "none",
        str(frame_pattern),
        color_a,
        color_b,
        "0",           # rotation_angle unused in animation mode
        str(resolution),
    ]

    if flip_z:
        cmd.append("flip")
    if stand_upright:
        cmd.append("stand_upright")
    cmd.append(color_mode)

    if camera_distance is not None:
        cmd.append(f"camdist={camera_distance}")
    if camera_height is not None:
        cmd.append(f"camheight={camera_height}")
    if camera_height_low is not None:
        cmd.append(f"camheightlow={camera_height_low}")
    if camera_base_angle is not None:
        cmd.append(f"camangle={camera_base_angle}")

    cmd.append(f"nframes={n_frames}")

    typer.echo("Launching Blender (single session for all frames)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    log = result.stdout + "\n" + result.stderr

    if result.returncode != 0 or "Traceback (most recent call last)" in log:
        print(f"\n[BLENDER LOG]\n{log}")
        raise RuntimeError("Blender animation render failed")

    # Blender names frames frame_0000.png … frame_NNNN.png
    missing = [i for i in range(n_frames) if not (frames_dir / f"frame_{i:04d}.png").exists()]
    if missing:
        typer.echo(f"Error: {len(missing)} frame(s) missing after render (e.g. frame_{missing[0]:04d}.png)")
        raise typer.Exit(1)


# =============================================================================
# CLI Commands
# =============================================================================


@app.command()
def render(
    step_file_a: Path = typer.Argument(..., help="First STEP file (shape A)"),
    step_file_b: Optional[Path] = typer.Argument(None, help="Second STEP file (shape B); omit to render shape A alone"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for the rendered image"),
    output_name: Optional[str] = typer.Option(None, help="Output filename (default: <stem_a>_<stem_b>.png)"),
    color_a: str = typer.Option(DEFAULT_COLOR_A, help="Color for shape A"),
    color_b: str = typer.Option(DEFAULT_COLOR_B, help="Color for shape B"),
    edge_deflection: float = typer.Option(DEFAULT_EDGE_DEFLECTION, help="Edge discretization deflection"),
    mesh_linear_deflection: float = typer.Option(DEFAULT_MESH_LINEAR_DEFLECTION, help="Mesh linear deflection"),
    mesh_angular_deflection: float = typer.Option(DEFAULT_MESH_ANGULAR_DEFLECTION, help="Mesh angular deflection"),
    stand_upright: bool = typer.Option(False, help="Rotate 90° around X axis"),
    flip_z: bool = typer.Option(False, help="Flip along Z axis"),
    no_normalize: bool = typer.Option(False, help="Skip normalization"),
    rotation_angle: float = typer.Option(DEFAULT_ROTATION_ANGLE, help="Camera rotation angle in degrees"),
    color_mode: str = typer.Option(DEFAULT_COLOR_MODE, help="Color mode (rgb/rgba)"),
    resolution: int = typer.Option(DEFAULT_RESOLUTION, help="Rendering resolution in pixels"),
    fast: bool = typer.Option(False, help="Fast mode: coarser mesh for speed"),
    camera_distance: float = typer.Option(None, help="Camera distance from model"),
    camera_height: float = typer.Option(None, help="Camera height (Z position)"),
    camera_base_angle: float = typer.Option(None, help="Camera base angle in degrees"),
    ground_plane_z: float = typer.Option(GROUND_PLANE_Z, help="Ground plane Z position after normalization"),
):
    """
    Render two STEP files together as a single static image.

    Both shapes are placed in the same coordinate space using joint normalization,
    preserving their relative positions as in an assembly.
    """
    files_to_check = [step_file_a] + ([step_file_b] if step_file_b is not None else [])
    for f in files_to_check:
        if not f.exists():
            typer.echo(f"Error: {f} does not exist")
            raise typer.Exit(1)

    blender_script = Path(BLENDER_PAIRED_RENDER_SCRIPT)
    if not blender_script.exists():
        typer.echo(f"Error: Blender script not found at {blender_script}")
        raise typer.Exit(1)

    if fast:
        edge_deflection = FAST_EDGE_DEFLECTION
        mesh_linear_deflection = FAST_MESH_LINEAR_DEFLECTION
        mesh_angular_deflection = FAST_MESH_ANGULAR_DEFLECTION

    if output_dir is None:
        output_dir = step_file_a.parent

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        typer.echo("Computing normalization from shape A...")
        shape_a = _read_step_file(step_file_a)
        if shape_a is None:
            raise typer.Exit(1)

        norm_params = None
        if not no_normalize:
            norm_params = _compute_normalization_params_from_a(
                shape_a, stand_upright, flip_z,
                mesh_linear_deflection, mesh_angular_deflection, ground_plane_z
            )

        typer.echo(f"Processing {step_file_a.name}...")
        npz_a = process_step_file(
            step_file_a, tmpdir_path, edge_deflection,
            mesh_linear_deflection, mesh_angular_deflection,
            stand_upright, flip_z, no_normalize,
            output_stem=step_file_a.stem,
            ground_plane_z=ground_plane_z,
            normalization_params=norm_params,
        )
        if not npz_a:
            typer.echo("Error: Failed to process shape A")
            raise typer.Exit(1)

        npz_b = None
        if step_file_b is not None:
            typer.echo(f"Processing {step_file_b.name}...")
            npz_b = process_step_file(
                step_file_b, tmpdir_path, edge_deflection,
                mesh_linear_deflection, mesh_angular_deflection,
                stand_upright, flip_z, no_normalize,
                output_stem=step_file_b.stem,
                ground_plane_z=ground_plane_z,
                normalization_params=norm_params,
            )
            if not npz_b:
                typer.echo("Error: Failed to process shape B")
                raise typer.Exit(1)

        typer.echo("Rendering...")
        render_blender_paired(
            npz_a, npz_b, output_dir, blender_script,
            color_a, color_b, rotation_angle,
            flip_z, stand_upright, color_mode, resolution,
            camera_distance, camera_height, camera_base_angle,
            output_name=output_name,
        )

    typer.echo(f"✓ Render saved to: {output_dir}")


@app.command()
def render_video(
    step_file_a: Path = typer.Argument(..., help="First STEP file (shape A)"),
    step_file_b: Optional[Path] = typer.Argument(None, help="Second STEP file (shape B); omit to render shape A alone"),
    output_path: Optional[Path] = typer.Option(None, help="Output video path (default: next to step_file_a)"),
    output_folder: Optional[Path] = typer.Option(None, help="Output folder; filename is auto-generated as <stem_a>_<stem_b>.<format>"),
    fps: int = typer.Option(DEFAULT_VIDEO_FPS, help="Frames per second"),
    duration: float = typer.Option(DEFAULT_VIDEO_DURATION, help="Duration in seconds"),
    video_format: str = typer.Option(DEFAULT_VIDEO_FORMAT, help="Output format (mp4 or gif)"),
    color_a: str = typer.Option(DEFAULT_COLOR_A, help="Color for shape A"),
    color_b: str = typer.Option(DEFAULT_COLOR_B, help="Color for shape B"),
    resolution: int = typer.Option(DEFAULT_RESOLUTION, help="Rendering resolution in pixels"),
    edge_deflection: float = typer.Option(DEFAULT_EDGE_DEFLECTION, help="Edge discretization deflection"),
    mesh_linear_deflection: float = typer.Option(DEFAULT_MESH_LINEAR_DEFLECTION, help="Mesh linear deflection"),
    mesh_angular_deflection: float = typer.Option(DEFAULT_MESH_ANGULAR_DEFLECTION, help="Mesh angular deflection"),
    stand_upright: bool = typer.Option(False, help="Rotate 90° around X axis"),
    flip_z: bool = typer.Option(False, help="Flip along Z axis"),
    no_normalize: bool = typer.Option(False, help="Skip normalization"),
    color_mode: str = typer.Option(DEFAULT_COLOR_MODE, help="Color mode (rgb or rgba)"),
    fast: bool = typer.Option(False, help="Fast mode: coarser mesh for speed"),
    camera_distance: float = typer.Option(None, help="Camera distance from model"),
    camera_height: float = typer.Option(None, help="Camera height for the first phase (upper position)"),
    camera_height_low: float = typer.Option(None, help="Camera height for the second phase (lower position)"),
    camera_base_angle: float = typer.Option(None, help="Camera base angle in degrees"),
    ground_plane_z: float = typer.Option(GROUND_PLANE_Z, help="Ground plane Z position after normalization"),
):
    """
    Create a rotating video animation of two STEP files rendered together.

    Both shapes share the same coordinate space via joint normalization,
    preserving their relative positions as in an assembly.

    The camera completes two full orbits. Height follows a cosine cycle: upper
    for the first orbit, lower for the second, returning to upper at the end so
    the video loops cleanly.
    """
    import shutil

    files_to_check = [step_file_a] + ([step_file_b] if step_file_b is not None else [])
    for f in files_to_check:
        if not f.exists():
            typer.echo(f"Error: {f} does not exist")
            raise typer.Exit(1)

    if output_path is None:
        folder = output_folder if output_folder is not None else step_file_a.parent
        stem = f"{step_file_a.stem}_{step_file_b.stem}" if step_file_b is not None else step_file_a.stem
        output_path = folder / f"{stem}.{video_format}"
    else:
        ext = output_path.suffix.lower().lstrip(".")
        if ext in ("mp4", "gif"):
            video_format = ext

    if video_format not in ("mp4", "gif"):
        typer.echo("Error: video_format must be 'mp4' or 'gif'")
        raise typer.Exit(1)

    if not shutil.which("ffmpeg"):
        typer.echo("Error: ffmpeg is not installed or not in PATH")
        raise typer.Exit(1)

    blender_script = Path(BLENDER_PAIRED_RENDER_SCRIPT)
    if not blender_script.exists():
        typer.echo(f"Error: Blender script not found at {blender_script}")
        raise typer.Exit(1)

    if fast:
        edge_deflection = FAST_EDGE_DEFLECTION
        mesh_linear_deflection = FAST_MESH_LINEAR_DEFLECTION
        mesh_angular_deflection = FAST_MESH_ANGULAR_DEFLECTION

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        npz_dir = tmpdir_path / "npz"
        frames_dir = tmpdir_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        label = f"{step_file_a.name} + {step_file_b.name}" if step_file_b is not None else step_file_a.name
        typer.echo(f"Creating video: {label}")
        typer.echo(f"Duration: {duration}s, FPS: {fps}, Format: {video_format}")

        typer.echo("Computing normalization from shape A...")
        shape_a = _read_step_file(step_file_a)
        if shape_a is None:
            raise typer.Exit(1)

        norm_params = None
        if not no_normalize:
            norm_params = _compute_normalization_params_from_a(
                shape_a, stand_upright, flip_z,
                mesh_linear_deflection, mesh_angular_deflection, ground_plane_z
            )

        typer.echo(f"Converting {step_file_a.name} to NPZ...")
        npz_a = process_step_file(
            step_file_a, npz_dir, edge_deflection,
            mesh_linear_deflection, mesh_angular_deflection,
            stand_upright, flip_z, no_normalize,
            output_stem=step_file_a.stem,
            ground_plane_z=ground_plane_z,
            normalization_params=norm_params,
        )
        if not npz_a:
            typer.echo("Error: Failed to process shape A")
            raise typer.Exit(1)

        npz_b = None
        if step_file_b is not None:
            typer.echo(f"Converting {step_file_b.name} to NPZ...")
            npz_b = process_step_file(
                step_file_b, npz_dir, edge_deflection,
                mesh_linear_deflection, mesh_angular_deflection,
                stand_upright, flip_z, no_normalize,
                output_stem=step_file_b.stem,
                ground_plane_z=ground_plane_z,
                normalization_params=norm_params,
            )
            if not npz_b:
                typer.echo("Error: Failed to process shape B")
                raise typer.Exit(1)

        # Render frames
        n_frames = int(fps * duration)
        typer.echo(f"Rendering {n_frames} frames...")
        _render_paired_video_frames(
            npz_a, npz_b, frames_dir, n_frames,
            blender_script, color_a, color_b, resolution,
            flip_z, stand_upright, color_mode,
            blender_executable=BLENDER_EXECUTABLE,
            camera_distance=camera_distance,
            camera_height=camera_height,
            camera_height_low=camera_height_low,
            camera_base_angle=camera_base_angle,
        )

        # Assemble video
        typer.echo(f"Creating {video_format.upper()} video...")
        _create_video_with_ffmpeg(frames_dir, output_path, fps, video_format)

    typer.echo(f"✓ Video saved to: {output_path}")
    typer.echo(f"  Frames: {n_frames}, Duration: {duration}s, FPS: {fps}")


if __name__ == "__main__":
    app()
