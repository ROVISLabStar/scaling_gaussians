import numpy as np
import trimesh
import pyrender
import imageio


def load_scene(glb_path, rotate_y_neg90=False):
    """
    Loads a GLB scene and returns a populated PyRender scene.
    
    Args:
        glb_path (str): Path to the GLB or glTF file.
        rotate_y_neg90 (bool): If True, rotates the entire scene -90° around the Y-axis.

    Returns:
        pyrender.Scene: The scene with geometry nodes added and optionally rotated.
    """
    import numpy as np
    import trimesh
    import pyrender

    # Load as Trimesh scene
    scene = trimesh.load(glb_path, force='scene')
    pyrender_scene = pyrender.Scene()

    # Define optional -90° Y-axis rotation
    if rotate_y_neg90:
        angle = np.radians(90)
        rot_y = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        scene_rotation = np.eye(4)
        scene_rotation[:3, :3] = rot_y
    else:
        scene_rotation = np.eye(4)

    # Load and optionally rotate each mesh
    for node_name in scene.graph.nodes_geometry:
        mesh = scene.geometry[scene.graph[node_name][1]]
        transform, _ = scene.graph[node_name]
        pose = scene_rotation @ transform  # Apply optional rotation
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        pyrender_scene.add(pyrender_mesh, pose=pose)

    return pyrender_scene


def update_camera(scene, cam_node, camera, pose):
    """
    Updates the camera node and attaches a directional light at the same pose.

    Args:
        scene (pyrender.Scene): The scene to update.
        cam_node (pyrender.Node or None): Previous camera node (or None).
        camera (pyrender.Camera): The camera object.
        pose (np.ndarray): 4x4 pose matrix.

    Returns:
        pyrender.Node: The new camera node (light is handled internally).
    """
    if cam_node is not None:
        scene.remove_node(cam_node)
    
    # Remove any existing light nodes
    for node in list(scene.nodes):
        if isinstance(node.light, pyrender.Light):
            scene.remove_node(node)
    
    # Add camera
    cam_node = scene.add(camera, pose=pose)

    # Add directional light at same pose
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=pose)

    return cam_node


