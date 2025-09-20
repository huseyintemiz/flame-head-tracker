import torch
import numpy as np
import trimesh
import open3d as o3d
from submodules.flame_lib.FLAME import FLAME
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def visualize_flame_and_camera(ret_dict, tracker, device='cuda:0'):
    """
    Visualize FLAME mesh and camera in 3D using Open3D

    Parameters:
    -----------
    ret_dict : dict
        Dictionary containing FLAME parameters and camera data
        Expected keys: shape, exp, head_pose, jaw_pose, neck_pose, eye_pose, cam, K, fov
    tracker : Tracker
        Tracker object with FLAME configuration
    device : str
        Compute device ('cuda:0' or 'cpu')
    """
    # Initialize FLAME model
    flame = FLAME(config=tracker.flame_cfg).to(device)
    print('tane tane')
    #flame = tracker.flame 
# 
    print(f"Available parameters in ret_dict: {ret_dict.keys()}")

    # Extract parameters from ret_dict
    shape = torch.tensor(ret_dict['shape'], device=device, dtype=torch.float32)
    exp = torch.tensor(ret_dict['exp'], device=device, dtype=torch.float32)
    head_pose = torch.tensor(ret_dict['head_pose'], device=device, dtype=torch.float32)
    jaw_pose = torch.tensor(ret_dict['jaw_pose'], device=device, dtype=torch.float32)
    neck_pose = torch.tensor(ret_dict['neck_pose'], device=device, dtype=torch.float32) if 'neck_pose' in ret_dict else None
    eye_pose = torch.tensor(ret_dict['eye_pose'], device=device, dtype=torch.float32) if 'eye_pose' in ret_dict else None

    # Forward pass through FLAME to get vertices
    with torch.no_grad():
        vertices, landmarks2d, landmarks3d = flame(
            shape_params=shape,
            expression_params=exp,
            head_pose_params=head_pose,
            jaw_pose_params=jaw_pose,
            neck_pose_params=neck_pose,
            eye_pose_params=eye_pose
        )

    vertices = vertices.cpu().numpy()[0]  # [N_verts, 3]
    faces = flame.faces_tensor.cpu().numpy()  # [N_faces, 3]

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Paint mesh with neutral color
    mesh.paint_uniform_color([0.8, 0.7, 0.6])

    # Create camera visualization
    camera_params = ret_dict['cam'][0]  # [tx, ty, tz]
    K = ret_dict['K'][0] if 'K' in ret_dict else None

    # Camera position (translation parameters)
    cam_position = camera_params[:3]

    # Create camera frustum
    camera_lines = create_camera_frustum(cam_position, K, ret_dict.get('fov', [30])[0])

    # Create coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)

    # Visualize
    o3d.visualization.draw_geometries(
        [mesh, camera_lines, coord_frame],
        window_name="FLAME Model and Camera",
        width=1024,
        height=768,
        mesh_show_wireframe=False,
        mesh_show_back_face=True
    )

    return mesh, camera_lines

def create_camera_frustum(cam_position, K, fov, scale=100):
    """
    Create a camera frustum for visualization
    """
    # Camera center
    center = np.array([cam_position[0], cam_position[1], cam_position[2]]) * scale

    # Create frustum corners based on FOV
    fov_rad = np.radians(fov)
    aspect_ratio = 1.0  # Assuming square image

    # Near and far planes
    near = 10
    far = 200

    # Calculate frustum corners
    h_near = 2 * near * np.tan(fov_rad / 2)
    w_near = h_near * aspect_ratio
    h_far = 2 * far * np.tan(fov_rad / 2)
    w_far = h_far * aspect_ratio

    # Near plane corners
    near_corners = np.array([
        [-w_near/2, -h_near/2, near],
        [w_near/2, -h_near/2, near],
        [w_near/2, h_near/2, near],
        [-w_near/2, h_near/2, near]
    ])

    # Far plane corners
    far_corners = np.array([
        [-w_far/2, -h_far/2, far],
        [w_far/2, -h_far/2, far],
        [w_far/2, h_far/2, far],
        [-w_far/2, h_far/2, far]
    ])

    # Add camera position offset
    near_corners += center
    far_corners += center

    # Create lines for frustum
    lines = []
    colors = []

    # Connect near plane corners
    for i in range(4):
        lines.append([i, (i+1)%4])
        colors.append([1, 0, 0])  # Red for near plane

    # Connect far plane corners
    for i in range(4):
        lines.append([i+4, ((i+1)%4)+4])
        colors.append([0, 0, 1])  # Blue for far plane

    # Connect near to far
    for i in range(4):
        lines.append([i, i+4])
        colors.append([0, 1, 0])  # Green for sides

    # Create line set
    all_points = np.vstack([near_corners, far_corners])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def visualize_with_matplotlib(ret_dict, tracker, device='cuda:0'):
    """
    Alternative visualization using matplotlib for simpler view
    """
    # Initialize FLAME model
    flame = tracker.flame  # Use the tracker's FLAME model directly

    # Extract parameters
    shape = torch.tensor(ret_dict['shape'], device=device, dtype=torch.float32)
    exp = torch.tensor(ret_dict['exp'], device=device, dtype=torch.float32)
    head_pose = torch.tensor(ret_dict['head_pose'], device=device, dtype=torch.float32)
    jaw_pose = torch.tensor(ret_dict['jaw_pose'], device=device, dtype=torch.float32)
    neck_pose = torch.tensor(ret_dict['neck_pose'], device=device, dtype=torch.float32) if 'neck_pose' in ret_dict else None
    eye_pose = torch.tensor(ret_dict['eye_pose'], device=device, dtype=torch.float32) if 'eye_pose' in ret_dict else None

    # Forward pass
    with torch.no_grad():
        vertices, _, _ = flame(
            shape_params=shape,
            expression_params=exp,
            head_pose_params=head_pose,
            jaw_pose_params=jaw_pose,
            neck_pose_params=neck_pose,
            eye_pose_params=eye_pose
        )

    vertices = vertices.cpu().numpy()[0]

    # Camera parameters
    cam_position = ret_dict['cam'][0] * 100  # Scale for visualization

    # Create 3D plot
    fig = plt.figure(figsize=(12, 6))

    # Plot 1: Side view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.5, alpha=0.5, c='lightblue')
    ax1.scatter(*cam_position, s=100, c='red', marker='^', label='Camera')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Side View')
    ax1.legend()

    # Plot 2: Front view
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.5, alpha=0.5, c='lightblue')
    ax2.scatter(*cam_position, s=100, c='red', marker='^', label='Camera')
    ax2.view_init(elev=0, azim=0)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Front View')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return fig

def visualize_flame_and_camera_plotly(ret_dict, tracker, device='cuda:0'):
    """
    Visualize FLAME mesh and camera in 3D using Plotly

    Parameters:
    -----------
    ret_dict : dict
        Dictionary containing FLAME parameters and camera data
        Expected keys: shape, exp, head_pose, jaw_pose, neck_pose, eye_pose, cam, K, fov
    tracker : Tracker
        Tracker object with FLAME configuration
    device : str
        Compute device ('cuda:0' or 'cpu')

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object with the 3D visualization
    """
    # Initialize FLAME model
    flame = FLAME(config=tracker.flame_cfg).to(device)

    print(f"Available parameters in ret_dict: {ret_dict.keys()}")

    # Extract parameters from ret_dict
    shape = torch.tensor(ret_dict['shape'], device=device, dtype=torch.float32)
    exp = torch.tensor(ret_dict['exp'], device=device, dtype=torch.float32)
    head_pose = torch.tensor(ret_dict['head_pose'], device=device, dtype=torch.float32)
    jaw_pose = torch.tensor(ret_dict['jaw_pose'], device=device, dtype=torch.float32)
    neck_pose = torch.tensor(ret_dict['neck_pose'], device=device, dtype=torch.float32) if 'neck_pose' in ret_dict else None
    eye_pose = torch.tensor(ret_dict['eye_pose'], device=device, dtype=torch.float32) if 'eye_pose' in ret_dict else None

    # Forward pass through FLAME to get vertices
    with torch.no_grad():
        vertices, landmarks2d, landmarks3d = flame(
            shape_params=shape,
            expression_params=exp,
            head_pose_params=head_pose,
            jaw_pose_params=jaw_pose,
            neck_pose_params=neck_pose,
            eye_pose_params=eye_pose
        )

    vertices = vertices.cpu().numpy()[0]  # [N_verts, 3]
    faces = flame.faces_tensor.cpu().numpy()  # [N_faces, 3]

    # FLAME head is at origin in world space
    # Apply head pose rotation to orient the head correctly
    from scipy.spatial.transform import Rotation as R

    # Head pose contains rotation parameters
    if len(head_pose.shape) == 1:
        head_pose = head_pose.unsqueeze(0)

    # Convert rotation vector to rotation matrix
    head_rotation = head_pose[:, :3].cpu().numpy()[0]  # First 3 params are rotation
    rot = R.from_rotvec(head_rotation)
    rotation_matrix = rot.as_matrix()

    # Apply rotation to vertices (head remains at origin but rotates)
    # Note: We may need to flip Y axis to match computer vision coordinate system
    vertices_rotated = vertices @ rotation_matrix.T

    # Apply coordinate system correction for computer vision (Y down, Z forward)
    # FLAME uses Y up, but rendered images typically use Y down
    coord_correction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])  # Flip Y axis
    vertices_corrected = vertices_rotated @ coord_correction.T

    # Scale for better visualization (keep head at origin)
    scale_factor = 1000
    vertices_world = vertices_corrected * scale_factor

    # Extract camera parameters
    camera_params = ret_dict['cam'][0]  # Camera parameters: [euler_x, euler_y, euler_z, tx, ty, tz]

    # Camera rotation (Euler angles) and translation
    cam_euler_angles = camera_params[:3].numpy() if torch.is_tensor(camera_params) else camera_params[:3]
    cam_translation = camera_params[3:6].numpy() if torch.is_tensor(camera_params) else camera_params[3:6]

    # Apply camera rotation (Euler angles to rotation matrix)
    from scipy.spatial.transform import Rotation as R
    cam_rotation = R.from_euler('xyz', cam_euler_angles)
    cam_rotation_matrix = cam_rotation.as_matrix()

    # Camera position in world space (scaled to match mesh)
    cam_position = cam_translation * scale_factor

    # Create FLAME mesh trace with world-space transformed vertices
    flame_mesh = go.Mesh3d(
        x=vertices_world[:, 0],  # Use world-space transformed vertices
        y=vertices_world[:, 1],
        z=vertices_world[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightcoral',  # More visible color
        opacity=1.0,  # Full opacity
        name='FLAME Head',
        flatshading=False,
        lighting=dict(
            ambient=0.6,
            diffuse=0.9,
            specular=0.3,
            roughness=0.3,
            fresnel=0.2
        ),
        lightposition=dict(
            x=1000,
            y=1000,
            z=1000
        ),
        showscale=False,
        hovertemplate='FLAME Mesh<extra></extra>'
    )

    # Also create wireframe overlay for better edge visibility
    wireframe_edges = []
    for face in faces:
        # Add edges for each triangle face
        for i in range(3):
            edge_x = [vertices_world[face[i], 0], vertices_world[face[(i+1)%3], 0], None]
            edge_y = [vertices_world[face[i], 1], vertices_world[face[(i+1)%3], 1], None]
            edge_z = [vertices_world[face[i], 2], vertices_world[face[(i+1)%3], 2], None]
            wireframe_edges.append(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='darkblue', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Limit wireframe edges to reduce rendering overhead (take every 10th edge)
    wireframe_edges = wireframe_edges[::10]

    # Create image plane with rendered image texture
    # Extract rendered image from ret_dict
    img_rendered = ret_dict.get('img_rendered', [None])[0]

    if img_rendered is not None:
        # Convert image to format suitable for Plotly

        # Convert tensor to numpy if needed
        if hasattr(img_rendered, 'cpu'):
            img_array = img_rendered.cpu().numpy()
        else:
            img_array = img_rendered

        # Ensure image is in correct format (H, W, C) and range [0, 255]
        if img_array.ndim == 3:
            if img_array.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                img_array = np.transpose(img_array, (1, 2, 0))

        # Normalize to [0, 255] if needed
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

        # Get image dimensions
        img_height, img_width = img_array.shape[:2]

        # Scale image plane to match image aspect ratio
        aspect_ratio = img_width / img_height
        image_plane_height = 300
        image_plane_width = image_plane_height * aspect_ratio

        # Position image plane based on camera position
        # Place it between camera and head, facing the camera
        plane_distance = np.linalg.norm(cam_position) * 0.6  # 60% of camera distance

        # Direction from camera to head (normalized)
        if np.linalg.norm(cam_position) > 0:
            cam_to_head = -cam_position / np.linalg.norm(cam_position)
        else:
            cam_to_head = np.array([0, 0, 1])

        plane_center = cam_position + cam_to_head * plane_distance

        # Create plane oriented perpendicular to camera view direction
        # Use camera rotation matrix for proper orientation
        # In computer vision coords: X=right, Y=down, Z=forward
        right = cam_rotation_matrix[:, 0] * image_plane_width/2
        up = cam_rotation_matrix[:, 1] * image_plane_height/2  # Y is down in CV coords

        # Image plane corners
        corners = np.array([
            plane_center - right - up,  # bottom-left
            plane_center + right - up,  # bottom-right
            plane_center + right + up,  # top-right
            plane_center - right + up   # top-left
        ])

        # Create image plane mesh with texture
        image_plane = go.Mesh3d(
            x=corners[:, 0],
            y=corners[:, 1],
            z=corners[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            intensity=np.linspace(0, 1, 4),  # For texture mapping
            colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']],
            opacity=0.9,
            name='Rendered Image',
            showscale=False,
            hovertemplate='Rendered Image<extra></extra>'
        )

        # Add image as a separate surface plot for better texture display
        # Create a grid for the image (reduce resolution for performance)
        downsample = max(1, max(img_width, img_height) // 200)  # Downsample large images
        img_display = img_array[::downsample, ::downsample]
        display_height, display_width = img_display.shape[:2]

        x_grid = np.linspace(corners[0, 0], corners[1, 0], display_width)
        y_grid = np.linspace(corners[0, 1], corners[3, 1], display_height)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.full_like(X, plane_center[2])

        # Handle different image formats
        if len(img_display.shape) == 3 and img_display.shape[2] == 3:
            # RGB image - convert to grayscale for surface plot
            img_gray = np.mean(img_display, axis=2)
            image_surface = go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=img_gray,  # Don't flip - image is already correct
                colorscale='gray',
                showscale=False,
                name='Rendered Image (Grayscale)',
                hovertemplate='Rendered Image<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
            )
        else:
            # Grayscale image
            img_display_2d = img_display if len(img_display.shape) == 2 else img_display[:, :, 0]
            image_surface = go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=img_display_2d,  # Don't flip - image is already correct
                colorscale='gray',
                showscale=False,
                name='Rendered Image',
                hovertemplate='Rendered Image<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
            )

    else:
        # Fallback: simple gray image plane
        image_plane_width = 400
        image_plane_height = 300
        plane_center = cam_position * 0.5

        image_plane = go.Mesh3d(
            x=[plane_center[0] - image_plane_width/2, plane_center[0] + image_plane_width/2,
               plane_center[0] + image_plane_width/2, plane_center[0] - image_plane_width/2],
            y=[plane_center[1] - image_plane_height/2, plane_center[1] - image_plane_height/2,
               plane_center[1] + image_plane_height/2, plane_center[1] + image_plane_height/2],
            z=[plane_center[2], plane_center[2], plane_center[2], plane_center[2]],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color='lightgray',
            opacity=0.2,
            name='Image Plane'
        )
        image_surface = None

    # Camera frustum from actual camera position with rotation
    fov = ret_dict.get('fov', [30])[0]
    frustum_traces = create_camera_frustum_plotly(cam_position, fov, scale=5, cam_rotation_matrix=cam_rotation_matrix)

    # Create camera marker at actual camera position
    camera_marker = go.Scatter3d(
        x=[cam_position[0]],
        y=[cam_position[1]],
        z=[cam_position[2]],
        mode='markers+text',
        marker=dict(size=20, color='red', symbol='diamond'),
        text=['Camera'],
        textposition='top center',
        textfont=dict(size=14, color='red'),
        name='Camera Position'
    )

    # Create camera coordinate axes to show orientation
    axes_length = 100
    cam_axes_traces = [
        # Camera X-axis (right) - red
        go.Scatter3d(
            x=[cam_position[0], cam_position[0] + cam_rotation_matrix[0, 0] * axes_length],
            y=[cam_position[1], cam_position[1] + cam_rotation_matrix[0, 1] * axes_length],
            z=[cam_position[2], cam_position[2] + cam_rotation_matrix[0, 2] * axes_length],
            mode='lines',
            line=dict(color='red', width=4),
            name='Camera X (Right)',
            showlegend=False
        ),
        # Camera Y-axis (down) - green
        go.Scatter3d(
            x=[cam_position[0], cam_position[0] + cam_rotation_matrix[1, 0] * axes_length],
            y=[cam_position[1], cam_position[1] + cam_rotation_matrix[1, 1] * axes_length],
            z=[cam_position[2], cam_position[2] + cam_rotation_matrix[1, 2] * axes_length],
            mode='lines',
            line=dict(color='green', width=4),
            name='Camera Y (Down)',
            showlegend=False
        ),
        # Camera Z-axis (forward) - blue
        go.Scatter3d(
            x=[cam_position[0], cam_position[0] + cam_rotation_matrix[2, 0] * axes_length],
            y=[cam_position[1], cam_position[1] + cam_rotation_matrix[2, 1] * axes_length],
            z=[cam_position[2], cam_position[2] + cam_rotation_matrix[2, 2] * axes_length],
            mode='lines',
            line=dict(color='blue', width=4),
            name='Camera Z (Forward)',
            showlegend=False
        )
    ]

    # Add landmarks if available (transformed to world space)
    landmarks_trace = None
    if landmarks3d is not None:
        landmarks = landmarks3d.cpu().numpy()[0]
        # Apply same transformation as mesh (rotation + coordinate correction)
        landmarks_rotated = landmarks @ rotation_matrix.T
        landmarks_corrected = landmarks_rotated @ coord_correction.T  # Apply same Y-flip
        landmarks_world = landmarks_corrected * scale_factor  # Scale but keep at origin
        landmarks_trace = go.Scatter3d(
            x=landmarks_world[:, 0],
            y=landmarks_world[:, 1],
            z=landmarks_world[:, 2],
            mode='markers',
            marker=dict(size=2, color='limegreen'),  # Larger, more visible markers
            name='Landmarks'
        )

    # Create coordinate axes (scaled up for visibility)
    axes_length = 200  # Increased for better visibility with scaled mesh
    axes_traces = [
        # X-axis (red)
        go.Scatter3d(
            x=[0, axes_length], y=[0, 0], z=[0, 0],
            mode='lines+text',
            line=dict(color='red', width=5),
            text=['', 'X'],
            textposition='top center',
            showlegend=False
        ),
        # Y-axis (green)
        go.Scatter3d(
            x=[0, 0], y=[0, axes_length], z=[0, 0],
            mode='lines+text',
            line=dict(color='green', width=5),
            text=['', 'Y'],
            textposition='top center',
            showlegend=False
        ),
        # Z-axis (blue)
        go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axes_length],
            mode='lines+text',
            line=dict(color='blue', width=5),
            text=['', 'Z'],
            textposition='top center',
            showlegend=False
        )
    ]

    # Combine all traces (including wireframe edges, image plane, and camera axes)
    traces = [flame_mesh] + wireframe_edges + [image_plane] + frustum_traces + [camera_marker] + cam_axes_traces + axes_traces

    # Add image surface if it exists
    if 'image_surface' in locals() and image_surface is not None:
        traces.append(image_surface)

    if landmarks_trace is not None:
        traces.append(landmarks_trace)

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout for better 3D visualization
    fig.update_layout(
        title="FLAME Head at Origin with Camera Position (Computer Vision Coords)",
        scene=dict(
            xaxis=dict(title='X (Right)', gridcolor='gray', showbackground=True),
            yaxis=dict(title='Y (Down)', gridcolor='gray', showbackground=True),
            zaxis=dict(title='Z (Forward)', gridcolor='gray', showbackground=True),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # View from above (Y is down now)
                center=dict(x=0, y=0, z=0),  # Look at origin where head is
                up=dict(x=0, y=0, z=1)  # Z points up in view
            )
        ),
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        annotations=[
            dict(
                text="Head at Origin (0,0,0) - Computer Vision Coordinates",
                x=0.5,
                y=0.95,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="black")
            )
        ]
    )

    return fig


def create_camera_frustum_plotly(cam_position, fov, scale=1, cam_rotation_matrix=None):
    """
    Create camera frustum traces for Plotly visualization

    Parameters:
    -----------
    cam_position : array
        Camera position [x, y, z]
    fov : float
        Field of view in degrees
    scale : float
        Scale factor for frustum size
    cam_rotation_matrix : array, optional
        3x3 rotation matrix for camera orientation

    Returns:
    --------
    traces : list
        List of Plotly trace objects for the frustum
    """
    # Camera center
    center = np.array(cam_position)

    # Frustum parameters
    fov_rad = np.radians(fov)
    aspect_ratio = 1.0

    # Near and far planes (pointing toward origin where head is)
    near = 10 * scale
    far = 100 * scale

    # Calculate frustum corners
    h_near = 2 * near * np.tan(fov_rad / 2)
    w_near = h_near * aspect_ratio
    h_far = 2 * far * np.tan(fov_rad / 2)
    w_far = h_far * aspect_ratio

    # Use camera rotation matrix if provided, otherwise point toward origin
    if cam_rotation_matrix is not None:
        # Camera coordinate system from rotation matrix
        # Camera looks down -Z axis in its local coordinate system
        view_dir = -cam_rotation_matrix[:, 2]  # -Z axis of camera
        right = cam_rotation_matrix[:, 0]      # X axis of camera (right)
        up = -cam_rotation_matrix[:, 1]        # -Y axis of camera (up in image)
    else:
        # Fallback: point toward origin
        view_dir = -center / np.linalg.norm(center) if np.linalg.norm(center) > 0 else np.array([0, 0, -1])

        # Create orthogonal basis for frustum
        if abs(view_dir[2]) < 0.99:
            up = np.array([0, 0, 1])
        else:
            up = np.array([0, 1, 0])

        right = np.cross(view_dir, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, view_dir)
        up = up / np.linalg.norm(up)

    # Near plane corners (in camera's view direction)
    near_center = center + view_dir * near
    near_corners = np.array([
        near_center - right * w_near/2 - up * h_near/2,
        near_center + right * w_near/2 - up * h_near/2,
        near_center + right * w_near/2 + up * h_near/2,
        near_center - right * w_near/2 + up * h_near/2
    ])

    # Far plane corners (in camera's view direction)
    far_center = center + view_dir * far
    far_corners = np.array([
        far_center - right * w_far/2 - up * h_far/2,
        far_center + right * w_far/2 - up * h_far/2,
        far_center + right * w_far/2 + up * h_far/2,
        far_center - right * w_far/2 + up * h_far/2
    ])

    traces = []

    # Near plane rectangle
    near_x = list(near_corners[:, 0]) + [near_corners[0, 0]]
    near_y = list(near_corners[:, 1]) + [near_corners[0, 1]]
    near_z = list(near_corners[:, 2]) + [near_corners[0, 2]]
    traces.append(go.Scatter3d(
        x=near_x, y=near_y, z=near_z,
        mode='lines',
        line=dict(color='red', width=2),
        name='Near Plane',
        showlegend=False
    ))

    # Far plane rectangle
    far_x = list(far_corners[:, 0]) + [far_corners[0, 0]]
    far_y = list(far_corners[:, 1]) + [far_corners[0, 1]]
    far_z = list(far_corners[:, 2]) + [far_corners[0, 2]]
    traces.append(go.Scatter3d(
        x=far_x, y=far_y, z=far_z,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Far Plane',
        showlegend=False
    ))

    # Connect corners (frustum edges)
    for i in range(4):
        traces.append(go.Scatter3d(
            x=[near_corners[i, 0], far_corners[i, 0]],
            y=[near_corners[i, 1], far_corners[i, 1]],
            z=[near_corners[i, 2], far_corners[i, 2]],
            mode='lines',
            line=dict(color='green', width=1),
            showlegend=False
        ))

    # Add lines from camera center to near plane corners
    for i in range(4):
        traces.append(go.Scatter3d(
            x=[center[0], near_corners[i, 0]],
            y=[center[1], near_corners[i, 1]],
            z=[center[2], near_corners[i, 2]],
            mode='lines',
            line=dict(color='orange', width=1, dash='dash'),
            showlegend=False
        ))

    return traces


def visualize_flame_mesh_only(ret_dict, tracker, device='cuda:0', show_wireframe=True, show_landmarks=True):
    """
    Visualize only the FLAME mesh in 3D using Plotly (clean view without camera/frustum)

    Parameters:
    -----------
    ret_dict : dict
        Dictionary containing FLAME parameters
        Expected keys: shape, exp, head_pose, jaw_pose, neck_pose, eye_pose, cam
    tracker : Tracker
        Tracker object with FLAME configuration
    device : str
        Compute device ('cuda:0' or 'cpu')
    show_wireframe : bool
        Whether to show wireframe overlay on the mesh
    show_landmarks : bool
        Whether to show facial landmarks

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object with the 3D mesh visualization
    """
    # Initialize FLAME model
    flame = FLAME(config=tracker.flame_cfg).to(device)

    print(f"Rendering FLAME mesh with parameters from ret_dict")

    # Extract parameters from ret_dict
    shape = torch.tensor(ret_dict['shape'], device=device, dtype=torch.float32)
    exp = torch.tensor(ret_dict['exp'], device=device, dtype=torch.float32)
    head_pose = torch.tensor(ret_dict['head_pose'], device=device, dtype=torch.float32)
    jaw_pose = torch.tensor(ret_dict['jaw_pose'], device=device, dtype=torch.float32)
    neck_pose = torch.tensor(ret_dict['neck_pose'], device=device, dtype=torch.float32) if 'neck_pose' in ret_dict else None
    eye_pose = torch.tensor(ret_dict['eye_pose'], device=device, dtype=torch.float32) if 'eye_pose' in ret_dict else None

    # Forward pass through FLAME to get vertices
    with torch.no_grad():
        vertices, _, landmarks3d = flame(
            shape_params=shape,
            expression_params=exp,
            head_pose_params=head_pose,
            jaw_pose_params=jaw_pose,
            neck_pose_params=neck_pose,
            eye_pose_params=eye_pose
        )

    vertices = vertices.cpu().numpy()[0]  # [N_verts, 3]
    faces = flame.faces_tensor.cpu().numpy()  # [N_faces, 3]

    # Transform FLAME mesh (keep at origin, only apply rotation)
    from scipy.spatial.transform import Rotation as R

    # Apply head pose rotation
    if len(head_pose.shape) == 1:
        head_pose = head_pose.unsqueeze(0)

    # Convert rotation vector to rotation matrix
    head_rotation = head_pose[:, :3].cpu().numpy()[0]  # First 3 params are rotation
    rot = R.from_rotvec(head_rotation)
    rotation_matrix = rot.as_matrix()

    # Apply rotation to vertices (head stays at origin)
    vertices_rotated = vertices @ rotation_matrix.T

    # Scale for better visualization
    scale_factor = 1000
    vertices_world = vertices_rotated * scale_factor

    # Create FLAME mesh trace
    flame_mesh = go.Mesh3d(
        x=vertices_world[:, 0],
        y=vertices_world[:, 1],
        z=vertices_world[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightcoral',
        opacity=1.0,
        name='FLAME Mesh',
        flatshading=False,
        lighting=dict(
            ambient=0.7,
            diffuse=0.9,
            specular=0.4,
            roughness=0.3,
            fresnel=0.3
        ),
        lightposition=dict(
            x=1000,
            y=1000,
            z=1000
        ),
        showscale=False,
        hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
    )

    traces = [flame_mesh]

    # Optionally add wireframe overlay
    if show_wireframe:
        wireframe_edges = []
        # Sample edges to reduce overhead
        edge_sampling = 20  # Show every 20th edge
        for idx, face in enumerate(faces):
            if idx % edge_sampling == 0:  # Sample edges
                for i in range(3):
                    edge_x = [vertices_world[face[i], 0], vertices_world[face[(i+1)%3], 0], None]
                    edge_y = [vertices_world[face[i], 1], vertices_world[face[(i+1)%3], 1], None]
                    edge_z = [vertices_world[face[i], 2], vertices_world[face[(i+1)%3], 2], None]
                    wireframe_edges.append(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color='rgba(0,0,100,0.3)', width=0.5),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        traces.extend(wireframe_edges)

    # Optionally add landmarks
    if show_landmarks and landmarks3d is not None:
        landmarks = landmarks3d.cpu().numpy()[0]
        # Apply same transformation as mesh (rotation only, keep at origin)
        landmarks_rotated = landmarks @ rotation_matrix.T
        landmarks_world = landmarks_rotated * scale_factor

        landmarks_trace = go.Scatter3d(
            x=landmarks_world[:, 0],
            y=landmarks_world[:, 1],
            z=landmarks_world[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='limegreen',
                symbol='circle'
            ),
            name='Landmarks',
            hovertemplate='Landmark<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        )
        traces.append(landmarks_trace)

    # Create subtle coordinate axes
    axes_length = 150
    axes_traces = [
        go.Scatter3d(
            x=[0, axes_length], y=[0, 0], z=[0, 0],
            mode='lines',
            line=dict(color='rgba(255,0,0,0.3)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ),
        go.Scatter3d(
            x=[0, 0], y=[0, axes_length], z=[0, 0],
            mode='lines',
            line=dict(color='rgba(0,255,0,0.3)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ),
        go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axes_length],
            mode='lines',
            line=dict(color='rgba(0,0,255,0.3)', width=2),
            showlegend=False,
            hoverinfo='skip'
        )
    ]
    traces.extend(axes_traces)

    # Create figure
    fig = go.Figure(data=traces)

    # Calculate mesh bounds for better camera positioning
    mesh_center = vertices_world.mean(axis=0)
    mesh_range = vertices_world.max(axis=0) - vertices_world.min(axis=0)
    max_range = mesh_range.max()

    # Update layout for clean mesh visualization
    fig.update_layout(
        title="FLAME Mesh Visualization",
        scene=dict(
            xaxis=dict(
                title='',
                showgrid=True,
                gridcolor='rgba(200,200,200,0.2)',
                showbackground=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                title='',
                showgrid=True,
                gridcolor='rgba(200,200,200,0.2)',
                showbackground=False,
                showticklabels=False,
                zeroline=False
            ),
            zaxis=dict(
                title='',
                showgrid=True,
                gridcolor='rgba(200,200,200,0.2)',
                showbackground=False,
                showticklabels=False,
                zeroline=False
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8),
                center=dict(
                    x=mesh_center[0]/max_range,
                    y=mesh_center[1]/max_range,
                    z=mesh_center[2]/max_range
                ),
                up=dict(x=0, y=1, z=0)
            ),
            bgcolor='rgba(240,240,240,1)'
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig


if __name__ == "__main__":
    # This will be called from the notebook
    pass