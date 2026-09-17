# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import requests
import trimesh
from matplotlib import colormaps
from scipy.spatial.transform import Rotation

try:
    import cv2
except ImportError:  # Sky filtering is optional; GLB export itself does not need OpenCV.
    cv2 = None


def predictions_to_glb(
    predictions: dict,
    conf_thres: float = 20.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    target_dir: str | None = None,
    max_points: int = 300000,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
) -> trimesh.Scene:
    """Convert VGGT predictions into a browser-viewable GLB scene.

    Both the original VGGT family and VGGT-Omega can use this function.  The
    original model directly predicts ``world_points`` and
    ``world_points_conf``; Omega's demo historically supplied the equivalent
    ``world_points_from_depth`` and ``depth_conf`` fields instead.

    The returned :class:`trimesh.Scene` contains a coloured POINTS primitive
    plus optional camera frusta.  This is the representation consumed by the
    ``<model-viewer>`` component used on the official VGGT project page.
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    conf_thres = float(conf_thres)
    if not 0.0 <= conf_thres <= 100.0:
        raise ValueError("conf_thres must be a percentile between 0 and 100")
    if max_points < 0:
        raise ValueError("max_points must be non-negative")

    points = _prediction_array(predictions, "world_points", "world_points_from_depth")
    conf = _prediction_array(predictions, "world_points_conf", "depth_conf")
    images = _prediction_array(predictions, "images")
    camera_matrices = _prediction_array(predictions, "extrinsic")

    points = _remove_batch_dimension(points, expected_ndim=5, name="world points")
    conf = _remove_batch_dimension(conf, expected_ndim=4, name="point confidence")
    images = _remove_batch_dimension(images, expected_ndim=5, name="images")
    camera_matrices = _remove_batch_dimension(camera_matrices, expected_ndim=4, name="extrinsic")
    if points.shape[:-1] != conf.shape:
        raise ValueError(f"Point/confidence shapes do not match: {points.shape} vs {conf.shape}")
    image_height_width = _image_height_width(images)
    if images.shape[0] != points.shape[0] or image_height_width != points.shape[1:3]:
        raise ValueError(
            "Images must have the same frame count and spatial size as the point map: "
            f"images={images.shape}, points={points.shape}"
        )
    if camera_matrices.shape != (points.shape[0], 3, 4):
        raise ValueError(
            "extrinsic must have shape [frames, 3, 4] matching the point map, got "
            f"{camera_matrices.shape}"
        )

    if filter_depth_edges and "depth" in predictions:
        depth = _remove_batch_dimension(_prediction_array(predictions, "depth"), expected_ndim=5, name="depth")
        if depth.shape[:-1] != conf.shape:
            raise ValueError(f"Depth/confidence shapes do not match: {depth.shape} vs {conf.shape}")
        conf = conf.copy()
        conf[depth_edge(depth[..., 0], rtol=depth_edge_rtol)] = 0.0

    if mask_sky and target_dir is not None:
        conf = apply_sky_mask(conf, target_dir)

    vertices = points.reshape(-1, 3)
    colors = _images_to_rgb(images).reshape(-1, 3)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf)
    if conf_thres > 0 and np.any(mask):
        conf_threshold = np.percentile(conf[mask], conf_thres)
        mask &= conf >= conf_threshold
    mask &= conf > 1e-5

    if mask_black_bg:
        mask &= colors.sum(axis=1) >= 16
    if mask_white_bg:
        mask &= ~((colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240))

    vertices = vertices[mask]
    colors = colors[mask]
    vertices, colors = _limit_points(vertices, colors, max_points)

    if vertices.size == 0:
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.array([[255, 255, 255]], dtype=np.uint8)
        scene_scale = 1.0
    else:
        lower = np.percentile(vertices, 5, axis=0)
        upper = np.percentile(vertices, 95, axis=0)
        scene_scale = float(np.linalg.norm(upper - lower))
        if scene_scale <= 0:
            scene_scale = 1.0

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors))

    extrinsics = np.zeros((len(camera_matrices), 4, 4), dtype=np.float64)
    extrinsics[:, :3, :4] = camera_matrices
    extrinsics[:, 3, 3] = 1.0

    if show_cam:
        colormap = colormaps.get_cmap("gist_rainbow")
        for i, world_to_camera in enumerate(extrinsics):
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba = colormap(i / max(len(extrinsics), 1))
            color = tuple(int(255 * x) for x in rgba[:3])
            integrate_camera_into_scene(scene, camera_to_world, color, scene_scale)

    return apply_scene_alignment(scene, extrinsics)


def point_cloud_to_glb(
    vertices: np.ndarray,
    colors: np.ndarray,
    camera_to_world: np.ndarray | None = None,
    max_points: int = 1_000_000,
) -> trimesh.Scene:
    """Create a GLB-ready coloured point cloud from already reconstructed points.

    This is the streaming-evaluation counterpart to :func:`predictions_to_glb`:
    callers can provide a bounded sample rather than materialising every pixel
    from a long sequence. ``vertices`` and ``camera_to_world`` use OpenCV
    coordinates and are converted to the OpenGL convention expected by GLB.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    colors = np.asarray(colors)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape [N, 3], got {vertices.shape}")
    if colors.ndim != 2 or colors.shape[0] != len(vertices) or colors.shape[1] not in (3, 4):
        raise ValueError(f"colors must have shape [N, 3] or [N, 4], got {colors.shape}")
    if max_points < 0:
        raise ValueError("max_points must be non-negative")
    if np.issubdtype(colors.dtype, np.floating):
        colors = (colors * 255 if colors.size and colors.max() <= 1.0 else colors).clip(0, 255).astype(np.uint8)
    else:
        colors = colors.clip(0, 255).astype(np.uint8)
    colors = colors[:, :3]

    mask = np.isfinite(vertices).all(axis=1)
    vertices, colors = vertices[mask], colors[mask]
    vertices, colors = _limit_points(vertices, colors, max_points)
    if not len(vertices):
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.array([[255, 255, 255]], dtype=np.uint8)
        scene_scale = 1.0
    else:
        lower, upper = np.percentile(vertices, 5, axis=0), np.percentile(vertices, 95, axis=0)
        scene_scale = max(float(np.linalg.norm(upper - lower)), 1.0)

    open_gl = get_opengl_conversion_matrix()
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=transform_points(open_gl, vertices), colors=colors))
    if camera_to_world is not None:
        camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
        if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (4, 4):
            raise ValueError(f"camera_to_world must have shape [frames, 4, 4], got {camera_to_world.shape}")
        colormap = colormaps.get_cmap("gist_rainbow")
        for index, transform in enumerate(camera_to_world):
            color = tuple(int(255 * value) for value in colormap(index / max(len(camera_to_world), 1))[:3])
            # Convert both world and camera axes from OpenCV to OpenGL.
            integrate_camera_into_scene(scene, open_gl @ transform, color, scene_scale)
    return scene


def _prediction_array(predictions: dict, *names: str) -> np.ndarray:
    """Return the first available prediction as a NumPy array."""
    for name in names:
        if name in predictions:
            value = predictions[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)
    choices = " or ".join(repr(name) for name in names)
    raise KeyError(f"Predictions must include {choices}")


def _remove_batch_dimension(array: np.ndarray, expected_ndim: int, name: str) -> np.ndarray:
    """Accept either a batched [1, ...] prediction or its per-scene form."""
    if array.ndim == expected_ndim:
        if array.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported for GLB export, got {name} shape {array.shape}")
        array = array[0]
    if array.ndim != expected_ndim - 1:
        raise ValueError(f"Unexpected {name} shape {array.shape}")
    return array


def _image_height_width(images: np.ndarray) -> tuple[int, int]:
    """Return image spatial dimensions for either [F, C, H, W] or [F, H, W, C]."""
    if images.shape[1] == 3:
        return tuple(images.shape[2:4])
    if images.shape[-1] == 3:
        return tuple(images.shape[1:3])
    raise ValueError(f"Images must be RGB [F, C, H, W] or [F, H, W, C], got {images.shape}")


def _images_to_rgb(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def _limit_points(vertices: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors
    indices = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    return vertices[indices], colors[indices]


def depth_edge(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])

    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)

    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)

    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    complete_transform = transform @ get_opengl_conversion_matrix() @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices = transform_points(complete_transform, vertices)

    camera_mesh = trimesh.Trimesh(vertices=vertices, faces=compute_camera_faces(camera_cone_shape))
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene: trimesh.Scene, extrinsics: np.ndarray) -> trimesh.Scene:
    opengl_conversion_matrix = get_opengl_conversion_matrix()
    scene.apply_transform(np.linalg.inv(extrinsics[0]) @ opengl_conversion_matrix)
    return scene


def get_opengl_conversion_matrix() -> np.ndarray:
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int | None = None) -> np.ndarray:
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    faces = []
    num_vertices = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices

        faces.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces += [(v3, v2, v1) for v1, v2, v3 in faces]
    return np.array(faces)


def apply_sky_mask(conf: np.ndarray, target_dir: str) -> np.ndarray:
    if cv2 is None:
        raise ImportError("Sky masking requires opencv-python; install the project's visualization dependencies.")
    image_dir = os.path.join(target_dir, "images")
    image_names = sorted(os.listdir(image_dir))
    height, width = conf.shape[-2:]
    masks = []
    skyseg_session = None

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(target_dir, "sky_masks", image_name)
        if os.path.exists(mask_path):
            sky_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            if not os.path.exists("skyseg.onnx"):
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                    "skyseg.onnx",
                )
            if skyseg_session is None:
                import onnxruntime

                skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
            sky_mask = segment_sky(image_path, skyseg_session, mask_path)

        if sky_mask.shape != (height, width):
            sky_mask = cv2.resize(sky_mask, (width, height))
        masks.append(sky_mask)

    return conf * (np.array(masks) > 0.1).astype(np.float32)


def segment_sky(image_path: str, onnx_session, mask_filename: str) -> np.ndarray:
    image = cv2.imread(image_path)
    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map)
    output_mask[result_map < 32] = 255

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size: list[int], image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.float32)
    image = (image / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)
    image = image.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: image})
    result = np.array(result).squeeze()
    result_min = np.min(result)
    result_max = np.max(result)
    if result_max > result_min:
        result = (result - result_min) / (result_max - result_min)
    else:
        result = np.zeros_like(result)
    return (result * 255).astype("uint8")


def download_file_from_url(url: str, filename: str) -> None:
    tmp_filename = f"{filename}.tmp"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(tmp_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    os.replace(tmp_filename, filename)
