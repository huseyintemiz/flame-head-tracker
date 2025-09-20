https://deepwiki.com/search/give-me-more-detail-about-land_f0f26e05-fd21-43be-98e8-7550566f28b2

You're asking about the landmark fitting camera pose optimization in the FLAME Head Tracker system, specifically the rigid camera pose fitting stage that estimates 6DoF camera parameters using detected facial landmarks.

## Camera Pose Optimization Overview

The landmark fitting camera pose optimization is implemented in the `run_rigid_camera_pose_fitting` method in `tracker_base.py` [1](#0-0) . This is Stage 1 of the two-stage optimization pipeline that estimates the 6DoF camera pose (yaw, pitch, roll, x, y, z) and field of view using detected facial landmarks.

## Optimization Parameters

The system optimizes three main camera-related parameters:

- **Camera rotation offsets** (`d_camera_rotation`): 3D rotation deltas [2](#0-1) 
- **Camera translation offsets** (`d_camera_translation`): 3D translation deltas [3](#0-2) 
- **FOV offsets** (`d_fov`): Field of view adjustments (optional) [4](#0-3) 

These parameters are initialized as learnable PyTorch parameters with learning rates of 0.05 initially, reduced to 0.005 after 1000 iterations [5](#0-4) .

## Loss Functions

The optimization uses multiple weighted loss terms:

### Facial Landmarks Loss
The primary loss focuses on face landmarks (excluding jawline) with higher weight: [6](#0-5) 

### Jawline Loss  
Separate loss for jawline landmarks with lower weight due to detection uncertainty: [7](#0-6) 

### Ear Landmarks Loss
When ear landmarks are available, the system applies angle-weighted ear fitting with threshold-based masking: [8](#0-7) 

The ear loss uses yaw angle weighting - left ear landmarks are used when yaw > 0, right ear when yaw < 0, with loss masking when landmark errors exceed 0.25 threshold.

### Temporal Smoothing
For video sequences, temporal smoothing regularizes camera motion between consecutive frames: [9](#0-8) 

## 3D-to-2D Projection Pipeline

The optimization projects 3D FLAME vertices to 2D screen coordinates through:

1. **FLAME forward pass** generates 3D vertices from shape/expression parameters [10](#0-9) 

2. **Camera intrinsics computation** from optimized FOV [11](#0-10) 

3. **Perspective projection** to clip coordinates [12](#0-11) 

4. **NDC conversion** for landmark comparison [13](#0-12) 

## Optimization Loop

The rigid fitting runs for 1500 iterations with Adam optimizer, using learning rate scheduling and NaN handling: [14](#0-13) 

The method returns the optimized camera pose, FOV, and parameter offsets that are then used in Stage 2 fine-tuning for expression and other FLAME coefficients.

## Notes

This rigid camera pose optimization is called from both landmark-based fitting (`run_fitting`) and photometric fitting (`run_fitting_photometric`) methods, serving as the foundation for accurate 3D face reconstruction by establishing proper camera viewpoint before optimizing facial expressions and other parameters.

Wiki pages you might want to explore:
- [Video Tracking (huseyintemiz/flame-head-tracker)](/wiki/huseyintemiz/flame-head-tracker#3.2)
- [Core Architecture (huseyintemiz/flame-head-tracker)](/wiki/huseyintemiz/flame-head-tracker#4)

### Citations

**File:** tracker_base.py (L587-589)
```python
    def run_rigid_camera_pose_fitting(self, shape, exp, head_pose, jaw_pose, 
                                      gt_landmarks, gt_ear_landmarks=None, temporal_smoothing=False):
        batch_size = shape.shape[0]
```

**File:** tracker_base.py (L592-598)
```python
        with torch.no_grad():
            vertices, _, _ = self.flame(shape_params=shape, expression_params=exp, 
                                        head_pose_params=head_pose, jaw_pose_params=jaw_pose) # [N, V, 3]
            face_68_vertices = self.flame.seletec_3d68(vertices)    # [N, 68, 3]
            left_ear_vertices = vertices[:, self.L_EAR_INDICES, :]  # [N, 20, 3]
            right_ear_vertices = vertices[:, self.R_EAR_INDICES, :] # [N, 20, 3]
            concat_vertices = torch.cat([face_68_vertices, left_ear_vertices, right_ear_vertices], dim=1) # [N, 108, 3]
```

**File:** tracker_base.py (L608-608)
```python
        d_camera_rotation = nn.Parameter(torch.zeros([batch_size, 3], dtype=torch.float32, device=self.device))
```

**File:** tracker_base.py (L609-609)
```python
        d_camera_translation = nn.Parameter(torch.zeros([batch_size, 3], dtype=torch.float32, device=self.device))
```

**File:** tracker_base.py (L610-610)
```python
        d_fov = nn.Parameter(torch.zeros([batch_size], dtype=torch.float32, device=self.device))
```

**File:** tracker_base.py (L611-617)
```python
        camera_params = [
            {'params': [d_camera_translation], 'lr': 0.05}, 
            {'params': [d_camera_rotation], 'lr': 0.05},
        ]
        if self.optimize_fov:
            camera_params.append({'params': [d_fov], 'lr': 0.05})

```

**File:** tracker_base.py (L636-718)
```python
        total_iterations = 1500
        for iter in range(total_iterations):
            # update learning rate
            if iter == 1000:
                e_opt_rigid.param_groups[0]['lr'] = 0.005    # translation
                e_opt_rigid.param_groups[1]['lr'] = 0.005    # rotation
                if self.optimize_fov:
                    e_opt_rigid.param_groups[2]['lr'] = 0.005     # fov

            # compute camera intrinsics
            optimized_fov = torch.clamp(fov + d_fov, min=10.0, max=50.0)                    # [N]
            optimized_focal_length = fov_to_focal(fov=optimized_fov, sensor_size=self.H)    # [N]
            Ks = build_intrinsics(focal_length=optimized_focal_length, image_size=self.H)   # [N,3,3]

            # project the vertices to 2D
            optimized_camera_pose = camera_pose + torch.cat([d_camera_rotation, d_camera_translation], dim=-1) # [N, 6]
            concat_verts_clip = batch_perspective_projection(verts=concat_vertices, camera_pose=optimized_camera_pose, 
                                                             K=Ks, image_size=self.H, near=self.znear, far=self.zfar) # [N, 108, 3]
            concat_verts_ndc_3d = batch_verts_clip_to_ndc(concat_verts_clip) # output [N, 108, 3] normalized to -1.0 ~ 1.0
            concat_verts_ndc_2d = concat_verts_ndc_3d[:,:,:2]
            
            # confidence weights (TODO)
            # with torch.no_grad():
            #     conf_weights = torch.ones([batch_size,68], dtype=torch.float32).to(self.device)

            # face 68 landmarks loss
            landmarks2d = concat_verts_ndc_2d[:,:68,:] # [N, 68, 3] normalized to -1.0 ~ 1.0
            loss_facial = compute_l2_distance_per_sample(landmarks2d[:, 17:, :2], gt_landmarks[:, 17:, :2]).sum() * 300   # face 51 landmarks
            loss_jawline = compute_l2_distance_per_sample(landmarks2d[:, :17, :2], gt_landmarks[:, :17, :2]).sum() * 200 # jawline loss
            loss_lmk = loss_facial + loss_jawline # [N]
            # loss_lmk = compute_l2_distance_per_sample(landmarks2d[:, :, :2], gt_landmarks[:, :, :2], lmk_weights).sum() * 500

            # ear landmarks loss
            EAR_LOSS_THRESHOLD = 0.25 # sometimes the detected ear landmarks are not accurate
            loss_ear = 0
            if self.use_ear_landmarks:
                left_ear_landmarks2d = concat_verts_ndc_2d[:,68:88,:2]    # [N, 20, 2]
                right_ear_landmarks2d = concat_verts_ndc_2d[:,88:108,:2]  # [N, 20, 2]

                with torch.no_grad():
                    euler_angles = rotation_vector_to_euler_angles(rot_vec = optimized_camera_pose[:, :3]) # [N, 3]
                    yaw_angles = euler_angles[:,0].detach()   # [N]
                    pitch_angles = euler_angles[:,1].detach() # [N]
                    mask_use_l = (yaw_angles > 0).float()   # [N]
                    mask_use_r = (yaw_angles < 0).float()   # [N]
                    abs_yaw_angles = torch.abs(yaw_angles)  # [N]
                    abs_yaw_angles[abs_yaw_angles < 0.1] = 0.0  # zero out small yaw angles
                    abs_pitch_angles = torch.abs(pitch_angles)  # [N]
                    abs_pitch_angles[abs_pitch_angles < 0.15] = 0.0  # zero out small pitch angles
                    angle_weights = abs_yaw_angles + abs_pitch_angles

                loss_l_ear = compute_l2_distance_per_sample(left_ear_landmarks2d, gt_ear_landmarks)
                mask_l_ear = (loss_l_ear < EAR_LOSS_THRESHOLD).float()

                loss_r_ear = compute_l2_distance_per_sample(right_ear_landmarks2d, gt_ear_landmarks)
                mask_r_ear = (loss_r_ear < EAR_LOSS_THRESHOLD).float()

                loss_l_ear = loss_l_ear * mask_l_ear * mask_use_l # values above threshold become 0
                loss_r_ear = loss_r_ear * mask_r_ear * mask_use_r # values above threshold become 0

                loss_ear = loss_l_ear + loss_r_ear # [N]
                loss_ear = loss_ear * 150 * angle_weights
                loss_ear = loss_ear.sum()

            if temporal_smoothing and batch_size >= 2 and iter > 600:
                reg_rot = torch.sum((optimized_camera_pose[1:,:3] - optimized_camera_pose[:-1,:3]) ** 2)
                reg_trans = torch.sum((optimized_camera_pose[1:,3:] - optimized_camera_pose[:-1,3:]) ** 2)
                loss_reg_cam_smooth = 150 * reg_rot + 150 * reg_trans
                # if iter % 100 == 0:
                #     print(loss_facial.item(), reg_rot.item(), reg_trans.item())
            else:
                loss_reg_cam_smooth = 0

            # loss = loss_facial + loss_jawline + loss_ear + loss_reg_cam_smooth
            loss = loss_lmk + loss_ear + loss_reg_cam_smooth
            # if iter % 100 == 0:
            #     print(f"Iter {iter}: loss_lmk={loss_lmk.item():.4f}, loss_ear={loss_ear.item() if isinstance(loss_ear, torch.Tensor) else loss_ear:.4f}, loss_reg_cam_smooth={loss_reg_cam_smooth.item() if isinstance(loss_reg_cam_smooth, torch.Tensor) else loss_reg_cam_smooth:.4f}, total_loss={loss.item():.4f}")

            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e5)
            e_opt_rigid.zero_grad()
            loss.backward()
            e_opt_rigid.step()

```
