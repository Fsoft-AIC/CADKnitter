import bpy
import numpy as np
import sys
import math
from pathlib import Path
from dataclasses import dataclass

COLOR_PRESETS = {
    "blue": [x / 255.0 for x in [135, 206, 235, 255]],
    "pink": [x / 255.0 for x in [255, 155, 188, 255]],
    "orange": [x / 255.0 for x in [255, 178, 111, 255]],
    "green": [x / 255.0 for x in [147, 197, 114, 255]],
    "silver": [x / 255.0 for x in [220, 220, 220, 255]],
    "lightorange": [x / 255.0 for x in [247, 127, 62, 255]],
    "lightblue": [x / 255.0 for x in [6, 125, 209, 255]],
    "transparent": [x / 255.0 for x in [6, 125, 209, 0]],
}


@dataclass(frozen=True)
class Arguments:
    input_file_a: Path
    input_file_b: Path
    output_file: Path
    color_a: str
    color_b: str
    rotation_angle: float
    resolution: int
    flip_z: bool
    stand_upright: bool
    color_mode: str
    camera_distance: float
    camera_height: float
    camera_height_low: float
    camera_base_angle: float
    n_frames: int  # 0 = single-frame mode, >0 = animation mode


def arg_parser() -> Arguments:
    """
    Parse command-line arguments for paired rendering.

    Single-frame mode (default):
        blender --background --python render_paired_step.py --
            npz_a npz_b output.png color_a color_b rotation_angle resolution [flags...]

    Animation mode (pass nframes=N flag):
        blender --background --python render_paired_step.py --
            npz_a npz_b /frames/dir/frame_ color_a color_b 0 resolution [flags...] nframes=N

    Flags: flip, stand_upright, rgb/rgba, camdist=X, camheight=X, camheightlow=X,
           camangle=X, nframes=N
    """
    parameters = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    if len(parameters) < 7:
        raise ValueError(
            "Usage: blender --background --python render_paired_step.py -- "
            "npz_a npz_b output.png color_a color_b rotation_angle resolution [flags...]"
        )

    input_file_a = Path(parameters[0])
    input_file_b = Path(parameters[1])
    output_file = Path(parameters[2])
    color_a = parameters[3]
    color_b = parameters[4]
    rotation_angle = float(parameters[5])
    resolution = int(parameters[6])

    flags = parameters[7:] if len(parameters) > 7 else []
    flip_z = "flip" in flags
    stand_upright = "stand_upright" in flags
    color_mode = "rgba" if "rgba" in flags else "rgb"

    camera_distance = 3.0
    camera_height = 2.0
    camera_height_low = 0.5
    camera_base_angle = -45.0
    n_frames = 0

    for flag in flags:
        if flag.startswith("camdist="):
            try:
                camera_distance = float(flag.split("=")[1])
            except ValueError:
                pass
        elif flag.startswith("camheightlow="):
            try:
                camera_height_low = float(flag.split("=")[1])
            except ValueError:
                pass
        elif flag.startswith("camheight="):
            try:
                camera_height = float(flag.split("=")[1])
            except ValueError:
                pass
        elif flag.startswith("camangle="):
            try:
                camera_base_angle = float(flag.split("=")[1])
            except ValueError:
                pass
        elif flag.startswith("nframes="):
            try:
                n_frames = int(flag.split("=")[1])
            except ValueError:
                pass

    if color_a not in COLOR_PRESETS:
        print(f"Warning: Unknown color '{color_a}', using 'blue'")
        color_a = "blue"
    if color_b not in COLOR_PRESETS:
        print(f"Warning: Unknown color '{color_b}', using 'silver'")
        color_b = "silver"

    return Arguments(
        input_file_a=input_file_a,
        input_file_b=input_file_b,
        output_file=output_file,
        color_a=color_a,
        color_b=color_b,
        rotation_angle=rotation_angle,
        resolution=resolution,
        flip_z=flip_z,
        stand_upright=stand_upright,
        color_mode=color_mode,
        camera_distance=camera_distance,
        camera_height=camera_height,
        camera_height_low=camera_height_low,
        camera_base_angle=camera_base_angle,
        n_frames=n_frames,
    )


def load_npz(path):
    """Load NPZ file and return (vertices, triangles, edges). Returns (None, None, []) for 'none'."""
    if str(path) == "none":
        return None, None, []
    data = np.load(path, allow_pickle=False)
    flat = data["edges_flat"]
    counts = data["edges_counts"]
    edges = []
    idx = 0
    for c in counts:
        edges.append(flat[idx:idx + int(c)])
        idx += int(c)
    vertices = data["vertices"] if "vertices" in data else None
    triangles = data["triangles"] if "triangles" in data else None
    return vertices, triangles, edges


def create_edge_object(edges, name="STEP_Edges"):
    curve_data = bpy.data.curves.new(name="EdgeCurve_" + name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 24
    curve_data.bevel_depth = 0.000
    curve_data.bevel_resolution = 1

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)

    for edge_points in edges:
        spline = curve_data.splines.new("NURBS")
        spline.points.add(len(edge_points) - 1)
        for i, point in enumerate(edge_points):
            spline.points[i].co = (*point, 1)
        spline.use_endpoint_u = True
        spline.order_u = 3

    return obj


def create_mesh_object(vertices, triangles, name="STEP_Mesh"):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], triangles.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_edge_material():
    mat = bpy.data.materials.new(name="Edge_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")

    principled.inputs["Base Color"].default_value = [x / 255 for x in [25, 25, 25, 255]]
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Specular"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 1.0

    mat.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return mat


def create_mesh_material(rgba: list, alpha: float = 1.0):
    mat = bpy.data.materials.new(name="Mesh_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes["Principled BSDF"]

    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Metallic"].default_value = 0.1
    principled.inputs["Roughness"].default_value = 0.8
    principled.inputs["Specular"].default_value = 0.1
    principled.inputs["Sheen"].default_value = 0.1

    if alpha < 1.0:
        mat.blend_method = "BLEND"
        principled.inputs["Alpha"].default_value = alpha
        mat.use_backface_culling = False

    return mat


def create_ground_plane():
    bpy.ops.mesh.primitive_plane_add(size=20)
    plane = bpy.context.active_object

    mat = bpy.data.materials.new(name="Ground_Material")
    mat.use_nodes = True
    principled = mat.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (1, 1, 1, 1)
    principled.inputs["Roughness"].default_value = 0.3

    plane.data.materials.append(mat)
    plane.is_shadow_catcher = True
    return plane


def set_white_background():
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.use_nodes = True

    compositor = bpy.context.scene.node_tree
    compositor.nodes.clear()

    render_layers = compositor.nodes.new(type="CompositorNodeRLayers")
    white_background = compositor.nodes.new(type="CompositorNodeRGB")
    white_background.outputs["RGBA"].default_value = (1, 1, 1, 1)
    alpha_over = compositor.nodes.new(type="CompositorNodeAlphaOver")
    composite_output = compositor.nodes.new(type="CompositorNodeComposite")

    compositor.links.new(render_layers.outputs["Image"], alpha_over.inputs[2])
    compositor.links.new(white_background.outputs["RGBA"], alpha_over.inputs[1])
    compositor.links.new(alpha_over.outputs["Image"], composite_output.inputs["Image"])

    bpy.context.scene.view_settings.view_transform = "Standard"


def create_lights():
    bpy.ops.object.light_add(type="AREA", location=(-1.5, -1.5, 2.5))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 35.0
    fill_light.scale = (3, 3, 3)

    bpy.ops.object.light_add(type="AREA", location=(2, -1, 2))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 35.0
    rim_light.scale = (2, 2, 2)

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 20))
    top_light = bpy.context.active_object
    top_light.data.energy = 0.7

    bpy.ops.object.light_add(type="POINT", location=(1, -3, 4))
    point_light = bpy.context.active_object
    point_light.data.energy = 350

    for light in bpy.data.lights:
        light.cycles.cast_shadow = False


def main():
    args = arg_parser()

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        gpu_devices = [d for d in prefs.devices if d.type in ("CUDA", "OPTIX")]
        for device in gpu_devices:
            device.use = True
        if gpu_devices:
            bpy.context.scene.cycles.device = "GPU"
        else:
            bpy.context.scene.cycles.device = "CPU"
            print("No CUDA/OptiX GPU found, falling back to CPU rendering.")
    except Exception as e:
        print(f"GPU setup failed ({e}), falling back to CPU rendering.")
        bpy.context.scene.cycles.device = "CPU"

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    vertices_a, triangles_a, edges_a = load_npz(args.input_file_a)
    vertices_b, triangles_b, edges_b = load_npz(args.input_file_b)

    # Camera orbits around shape A's bounding box center
    if vertices_a is not None and len(vertices_a) > 0:
        center_a = ((vertices_a.min(axis=0) + vertices_a.max(axis=0)) / 2).tolist()
    else:
        center_a = [0.0, 0.0, 0.0]

    # Create objects for shape A
    mesh_obj_a = None
    if vertices_a is not None and triangles_a is not None and len(vertices_a) > 0:
        mesh_obj_a = create_mesh_object(vertices_a, triangles_a, name="STEP_Mesh_A")
    edge_obj_a = create_edge_object(edges_a, name="STEP_Edges_A")

    # Create objects for shape B
    mesh_obj_b = None
    if vertices_b is not None and triangles_b is not None and len(vertices_b) > 0:
        mesh_obj_b = create_mesh_object(vertices_b, triangles_b, name="STEP_Mesh_B")
    edge_obj_b = create_edge_object(edges_b, name="STEP_Edges_B")

    create_lights()

    if mesh_obj_a:
        edge_obj_a.location.z += 0.0002
    if mesh_obj_b:
        edge_obj_b.location.z += 0.0002

    ground = create_ground_plane()
    # Keep the ground plane below the lowest camera position so it never
    # clips the view when camera_height_low is negative.
    min_cam_z = min(args.camera_height, args.camera_height_low) if args.n_frames > 0 else args.camera_height
    ground.location.z = min(-0.5, min_cam_z - 0.5)

    if mesh_obj_a:
        mesh_obj_a.data.materials.append(create_mesh_material(COLOR_PRESETS[args.color_a]))
    edge_obj_a.data.materials.append(create_edge_material())

    if mesh_obj_b:
        mesh_obj_b.data.materials.append(create_mesh_material(COLOR_PRESETS[args.color_b]))
    edge_obj_b.data.materials.append(create_edge_material())

    radius = args.camera_distance
    base_angle = math.radians(args.camera_base_angle)

    # Initial camera position (used for single-frame mode or as keyframe seed)
    theta0 = math.radians(args.rotation_angle)
    cam_x = radius * math.cos(base_angle + theta0)
    cam_y = radius * math.sin(base_angle + theta0)
    cam_z = args.camera_height

    bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
    camera = bpy.context.active_object

    empty = bpy.data.objects.new("CameraTarget", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = center_a

    track = camera.constraints.new(type="TRACK_TO")
    track.target = empty
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = True
    scene.cycles.samples = 256

    world = bpy.data.worlds["World"]
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    bg = nodes.new("ShaderNodeBackground")
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    mix = nodes.new("ShaderNodeMixShader")
    output = nodes.new("ShaderNodeOutputWorld")

    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1)
    bg.inputs["Strength"].default_value = 0.5
    ao.inputs["Distance"].default_value = 0.5
    ao.inputs["Color"].default_value = (0.1, 0.1, 0.1, 1)

    world.node_tree.links.new(ao.outputs["Color"], mix.inputs[1])
    world.node_tree.links.new(bg.outputs["Background"], mix.inputs[2])
    world.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])

    if args.color_mode == "rgb":
        set_white_background()
    else:
        bpy.context.scene.view_settings.view_transform = "Standard"

    if args.n_frames > 0:
        # Animation mode: insert per-frame camera keyframes, render all at once.
        # Two full orbits (720°). Height follows a cosine cycle: upper → lower → upper.
        n = args.n_frames
        h_upper = args.camera_height
        h_lower = args.camera_height_low

        scene.frame_start = 0
        scene.frame_end = n - 1
        scene.render.filepath = str(args.output_file)  # pattern, e.g. /tmp/frames/frame_

        for i in range(n):
            progress = i / n
            rotation_angle = i * (720.0 / n)
            total_angle = base_angle + math.radians(rotation_angle)
            frame_height = (
                (h_upper + h_lower) / 2.0
                + (h_upper - h_lower) / 2.0 * math.cos(2.0 * math.pi * progress)
            )
            camera.location = (
                radius * math.cos(total_angle),
                radius * math.sin(total_angle),
                frame_height,
            )
            camera.keyframe_insert(data_path="location", frame=i)

        bpy.ops.render.render(animation=True)
    else:
        scene.render.filepath = str(args.output_file)
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
