#
# FLAME Tracker Reconstruction Base for Multi-View Input Images
# Adapted to current codebase (v4.1+)
# Original Author: Peizhi Yan
# Copyright (C) Peizhi Yan. 2024-2025
#

# Installed Packages
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Mediapipe  (version 0.10.15)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# FAN (1.4.1)
import face_alignment

# FLAME
from submodules.flame_lib.FLAME import FLAME, FLAMETex

# FLAME photometric fitting utilities
import submodules.flame_fitting.fitting_util as fitting_util
from submodules.flame_fitting.renderer import Renderer

# Face parsing model
from submodules.face_parsing.FaceParsingUtil import FaceParsing

# DECA
from utils.deca_inference_utils import create_deca_model, get_flame_code_from_deca

# MICA
from utils.mica_inference_utils import create_mica_model, get_shape_code_from_mica

# Utility
from utils.mp2dlib import convert_landmarks_mediapipe_to_dlib
from utils.loss_utils import *
from utils.general_utils import check_nan_in_dict, prepare_batch_visualized_results
import utils.o3d_utils as o3d_utils
from utils.image_utils import read_img, image_align, get_face_mask
from utils.graphics_utils import fov_to_focal, build_intrinsics, batch_perspective_projection, \
                                 batch_verts_clip_to_ndc, batch_verts_ndc_to_screen, \
                                 rotation_vector_to_euler_angles
from utils.matting_utils import load_matting_model, matting_single_image


class Tracker():

    def __init__(self, tracker_cfg):
        self.VERSION = '4.1-multiview'
        flame_cfg = {
            'mediapipe_face_landmarker_v2_path': tracker_cfg['mediapipe_face_landmarker_v2_path'],
            'flame_model_path': tracker_cfg['flame_model_path'],
            'flame_lmk_embedding_path': tracker_cfg['flame_lmk_embedding_path'],
            'tex_space_path': tracker_cfg['tex_space_path'],
            'tex_type': 'BFM',
            'camera_params': 3,          # do not change it
            'shape_params': 300,
            'expression_params': 100,    # by default, we use 100 FLAME expression coefficients
            'pose_params': 6,
            'tex_params': 50,            # we use the first 50 FLAME texture model coefficients
            'use_face_contour': False,   # we don't use the face countour landmarks
            'cropped_size': 256,         # the render size for rendering the mesh
            'batch_size': 1,             # do not change it
            'image_size': 224,           # used in DECA, do not change it
            'e_lr': 0.01,
            'e_wd': 0.0001,
            'savefolder': './test_results/',
            # weights of losses and reg terms
            'w_pho': 8,
            'w_lmks': 1,
            'w_shape_reg': 1e-4,
            'w_expr_reg': 1e-4,
            'w_pose_reg': 0,
        }
        self.device = tracker_cfg['device']
        self.flame_cfg = fitting_util.dict2obj(flame_cfg)
        self.IMG_SIZE = tracker_cfg['result_img_size']
        self.NUM_SHAPE_COEFFICIENTS = flame_cfg['shape_params']
        self.NUM_EXPR_COEFFICIENTS = flame_cfg['expression_params']
        self.NUM_TEX_COEFFICIENTS = flame_cfg['tex_params']

        self.set_landmark_detector('mediapipe')

        if 'ear_landmarker_path' in tracker_cfg:
            self.use_ear_landmarks = True
            self.ear_landmarker = torch.load(tracker_cfg['ear_landmarker_path'], weights_only=False).eval()
            self.ear_landmarker = self.ear_landmarker.to(self.device)
        else:
            self.use_ear_landmarks = False

        # Mediapipe face landmark detector
        base_options = python.BaseOptions(model_asset_path=tracker_cfg['mediapipe_face_landmarker_v2_path'])
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=False,
                                               num_faces=1)
        self.mediapipe_detector = vision.FaceLandmarker.create_from_options(options)

        # FAN face alignment predictor (68 landmarks)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_HALF_D, flip_input=True, face_detector='sfd')

        # Face parsing model
        self.face_parser = FaceParsing(model_path=tracker_cfg['face_parsing_model_path'])

        # FLAME model and FLAME texture model
        self.flame = FLAME(self.flame_cfg).to(self.device)
        self.flametex = FLAMETex(self.flame_cfg).to(self.device)

        # Eye Landmarks (mediapipe) and indices (FLAME mesh)
        self.R_EYE_MP_LMKS = [468, 469, 470, 471, 472]
        self.L_EYE_MP_LMKS = [473, 474, 475, 476, 477]
        self.R_EYE_INDICES = [4597, 4543, 4511, 4479, 4575]
        self.L_EYE_INDICES = [4051, 3997, 3965, 3933, 4020]

        # Ear Indices (FLAME mesh)
        self.L_EAR_INDICES = [342, 341, 166, 514, 476, 185, 369, 29, 204, 641, 179, 178, 71, 68, 138, 141, 91, 40, 96, 184]
        self.R_EAR_INDICES = [1263, 844, 845, 2655, 870, 872, 1207, 523, 901, 1859, 860, 621, 618, 890, 981, 556, 554, 676, 1209, 868]

        # Camera settings
        self.RENDER_SIZE = self.H = self.W = self.flame_cfg.cropped_size
        self.DEFAULT_FOV = 20.0
        self.DEFAULT_DISTANCE = 1.0
        self.DEFAULT_FOCAL = fov_to_focal(fov = self.DEFAULT_FOV, sensor_size = self.H)
        self.update_fov(fov = self.DEFAULT_FOV)
        self.bg_color = (1.0,1.0,1.0)
        self.znear = 0.01
        self.zfar  = 100.0

        # FLAME render (from DECA)
        self.flame_texture_render = Renderer(self.flame_cfg.cropped_size,
                                     obj_filename=tracker_cfg['template_mesh_file_path']).to(self.device)

        # Load the template FLAME triangle faces
        _, self.faces, self.uv_coords, _ = o3d_utils._read_obj_file(tracker_cfg['template_mesh_file_path'], uv=True)
        self.uv_coords = np.array(self.uv_coords, dtype=np.float32)
        self.mesh_faces = torch.from_numpy(self.faces).to(self.device).detach() # [F, 3]

        # Load DECA model
        self.deca = create_deca_model(self.device)

        # Load MICA model
        self.mica = create_mica_model(self.device)

        # Matting model
        if 'use_matting' in tracker_cfg:
            self.use_matting = tracker_cfg['use_matting']
            self.video_matting_model = load_matting_model(device=self.device)
        else:
            self.use_matting = False

        print(f'\n>>> Flame Head Tracker v{self.VERSION} ready.')


    def set_landmark_detector(self, landmark_detector='mediapipe'):
        assert landmark_detector in ['mediapipe', 'FAN'], "landmark_detector need to be either mediapipe or FAN"
        self.landmark_detector = landmark_detector


    def update_fov(self, fov : float):
        assert 10 <= fov <= 60, f"FOV must be between 10 and 60. Provided: {fov}"
        self.fov = fov
        self.focal = fov_to_focal(fov = fov, sensor_size = self.H)
        self.distance = self.DEFAULT_DISTANCE * (self.focal / self.DEFAULT_FOCAL)


    def mediapipe_face_detection(self, img):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = self.mediapipe_detector.detect(image)
        if len(detection_result.face_blendshapes) == 0:
            return None, None
        blend_scores = detection_result.face_blendshapes[0]
        blend_scores = np.array(list(map(lambda l: l.score, blend_scores)), dtype=np.float32)
        lmks_dense = detection_result.face_landmarks[0]
        lmks_dense = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks_dense)))
        lmks_dense[:, 0] = lmks_dense[:, 0] * img.shape[1]
        lmks_dense[:, 1] = lmks_dense[:, 1] * img.shape[0]
        return lmks_dense, blend_scores


    def fan_face_landmarks(self, img):
        face_landmarks = self.fa.get_landmarks(img)
        if face_landmarks is None:
            return None
        else:
            return face_landmarks[0][:,:2] # [68, 2]


    @torch.no_grad()
    def detect_ear_landmarks(self, img):
        EAR_LMK_DETECTOR_INPUT_SIZE = 368
        input_size = (EAR_LMK_DETECTOR_INPUT_SIZE, EAR_LMK_DETECTOR_INPUT_SIZE)
        input_image = cv2.resize(img, input_size)
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image[None]
        input_image_tensor = torch.from_numpy(input_image).to(self.device)
        heatmaps = self.ear_landmarker(input_image_tensor)
        heatmaps = heatmaps.detach().cpu().numpy()[0]
        heatmap = cv2.resize(heatmaps, input_size)
        blurred_heatmap = gaussian_filter(heatmap, sigma=2.5)
        temp = np.argmax(blurred_heatmap.reshape(-1, 55), axis=0)
        ear_landmarks = np.zeros([20, 2], dtype=np.float32)
        for i in range(20):
            idx = temp[i]
            x, y = idx % EAR_LMK_DETECTOR_INPUT_SIZE, idx // EAR_LMK_DETECTOR_INPUT_SIZE
            ear_landmarks[i, 0] = x
            ear_landmarks[i, 1] = y
        ear_landmarks = ear_landmarks / float(EAR_LMK_DETECTOR_INPUT_SIZE) * 2 - 1
        return ear_landmarks


    @torch.no_grad()
    def run_reconstruction_models(self, img, lmks_68):
        """Run DECA and MICA to get the initial FLAME coefficients for a single image."""
        deca_dict = get_flame_code_from_deca(self.deca, img, self.device)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        shape_code = get_shape_code_from_mica(self.mica, img_bgr, lmks_68, self.device) # [1, 300]
        recon_dict = {}
        recon_dict['shape'] = shape_code[:, :self.NUM_SHAPE_COEFFICIENTS]
        recon_dict['exp'] = np.zeros([1, self.NUM_EXPR_COEFFICIENTS], dtype=np.float32)
        exp_code = deca_dict['exp'].detach().cpu().numpy()[:,:min(50, self.NUM_EXPR_COEFFICIENTS)]
        recon_dict['exp'][:, :min(50, self.NUM_EXPR_COEFFICIENTS)] = exp_code
        pose = deca_dict['pose'].detach().cpu().numpy() # [1, 6] head + jaw pose
        recon_dict['head_pose'] = pose[:,:3]
        recon_dict['jaw_pose'] = pose[:,3:]
        recon_dict['tex'] = np.zeros([1, self.NUM_TEX_COEFFICIENTS], dtype=np.float32)
        tex_code = deca_dict['tex'].detach().cpu().numpy()[:,:min(50, self.NUM_TEX_COEFFICIENTS)]
        recon_dict['tex'][:, :min(50, self.NUM_TEX_COEFFICIENTS)] = tex_code
        recon_dict['light'] = deca_dict['light'].detach().cpu().numpy()
        return recon_dict


    def load_images_and_run(self, img_paths, realign=True, photometric_fitting=False, shape_code=None):
        imgs = []
        for img_path in img_paths:
            img = read_img(img_path)
            if self.use_matting:
                img = matting_single_image(self.video_matting_model, img)
            imgs.append(img)
        return self.run(imgs, realign, photometric_fitting, shape_code)


    def run(self, imgs, realign=True, photometric_fitting=False, shape_code=None):
        """
        Run multi-view FLAME tracking on the given images.
        input:
            -imgs: list of image data [numpy]
            -realign: for FFHQ, use False. for in-the-wild images, use True
            -photometric_fitting: whether to use photometric fitting or landmarks only
            -shape_code: the pre-estimated global shape code
        output:
            -ret_dict: results dictionary
        """
        N = len(imgs)

        # run Mediapipe face detector and get 68 landmarks for alignment
        lmks_dense_list = []
        face_landmarks_list = []
        for img in imgs:
            lmks_dense, blend_scores = self.mediapipe_face_detection(img)
            lmks_dense_list.append(lmks_dense)
            if lmks_dense is not None:
                face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
                face_landmarks_list.append(face_landmarks)
            else:
                face_landmarks_list.append(None)

        # re-align images
        imgs_aligned = []
        for i, img in enumerate(imgs):
            fl = face_landmarks_list[i]
            if fl is not None:
                img_aligned = image_align(img, fl, output_size=self.IMG_SIZE, standard='tracking',
                                        padding_mode='constant')
            else:
                img_aligned = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            imgs_aligned.append(img_aligned)

        if realign:
            imgs = [img_al.copy() for img_al in imgs_aligned]

        # run DECA+MICA reconstruction for each view
        recon_dict_list = []
        for i in range(N):
            if lmks_dense_list[i] is not None:
                lmks_68 = face_landmarks_list[i]
                recon_dict = self.run_reconstruction_models(img=np.copy(imgs[i]), lmks_68=np.copy(lmks_68))
                recon_dict_list.append(recon_dict)
            else:
                recon_dict_list.append(None)

        # run face parsing
        parsing_mask_list = []
        parsing_mask_aligned_list = []
        for i in range(N):
            img = imgs[i]
            img_aligned = imgs_aligned[i]
            parsing_mask = self.face_parser.run(img)
            parsing_mask_list.append(parsing_mask)
            if realign: parsing_mask_aligned = parsing_mask
            else: parsing_mask_aligned = self.face_parser.run(img_aligned)
            parsing_mask_aligned_list.append(parsing_mask_aligned)

        # average the DECA/MICA parameters across valid views
        params = {}
        for key in ['shape', 'exp', 'head_pose', 'jaw_pose', 'tex', 'light']:
            if key == 'shape' and shape_code is not None:
                params[key] = shape_code
            else:
                count = 0
                temp = None
                for rd in recon_dict_list:
                    if rd is not None:
                        val = rd[key]
                        if temp is None:
                            temp = val.copy()
                        else:
                            temp += val
                        count += 1
                if count == 0:
                    return None
                params[key] = temp / count

        if photometric_fitting:
            face_mask_list = []
            for i in range(N):
                parsing_mask = parsing_mask_list[i]
                face_mask = get_face_mask(parsing=parsing_mask, keep_ears=False)
                face_mask_list.append(face_mask)
            ret_dict = self.run_fitting_multiview_photometric(imgs, face_mask_list, recon_dict_list, params, shape_code)
        else:
            ret_dict = self.run_fitting_multiview(imgs, recon_dict_list, params, shape_code)

        if ret_dict is None:
            return None

        # check for NaNs (only on plain numpy arrays, skip lists and plotting keys)
        skip_keys = {'img_rendered', 'mesh_rendered', 'cam', 'img', 'img_aligned', 'parsing', 'parsing_aligned'}
        nan_check_dict = {k: v for k, v in ret_dict.items()
                          if isinstance(v, np.ndarray) and k not in skip_keys}
        _, nan_status = check_nan_in_dict(nan_check_dict)
        if nan_status:
            return None

        # add more data
        ret_dict['img'] = imgs
        ret_dict['img_aligned'] = imgs_aligned
        ret_dict['parsing'] = parsing_mask_list
        ret_dict['parsing_aligned'] = parsing_mask_aligned_list

        return ret_dict


    def _detect_landmarks_for_fitting(self, imgs_resized):
        """Detect landmarks on resized images for multi-view fitting.
        Returns normalized landmarks in -1..1 range, plus ear landmarks if enabled.
        """
        N = len(imgs_resized)
        lmks_dense_list = []
        face_landmarks_list = []
        gt_ear_landmarks_list = []
        valid_mask = []

        for i, img_resized in enumerate(imgs_resized):
            lmks_dense, _ = self.mediapipe_face_detection(img_resized)
            if lmks_dense is None:
                lmks_dense_list.append(None)
                face_landmarks_list.append(None)
                gt_ear_landmarks_list.append(None)
                valid_mask.append(False)
                continue

            # normalize landmarks to -1..1
            lmks_dense[:, :2] = lmks_dense[:, :2] / float(self.RENDER_SIZE) * 2 - 1
            if self.landmark_detector == 'mediapipe':
                face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
            elif self.landmark_detector == 'FAN':
                face_landmarks = self.fan_face_landmarks(img_resized)
                if face_landmarks is None:
                    lmks_dense_list.append(None)
                    face_landmarks_list.append(None)
                    gt_ear_landmarks_list.append(None)
                    valid_mask.append(False)
                    continue
                face_landmarks = face_landmarks[:, :2] / float(self.RENDER_SIZE) * 2 - 1

            lmks_dense_list.append(lmks_dense)
            face_landmarks_list.append(face_landmarks)
            valid_mask.append(True)

            if self.use_ear_landmarks:
                ear_landmarks = self.detect_ear_landmarks(img_resized)
                gt_ear_landmark = torch.from_numpy(ear_landmarks[None]).float().to(self.device)
                gt_ear_landmarks_list.append(gt_ear_landmark)

        # prepare ground-truth landmark tensors (only for valid views)
        gt_landmark_list = []
        gt_eyes_landmark_list = []
        for i in range(N):
            if face_landmarks_list[i] is not None:
                gt_landmark_list.append(np.array(face_landmarks_list[i]).astype(np.float32))
                gt_eyes_landmark_list.append(
                    np.array(lmks_dense_list[i][self.R_EYE_MP_LMKS + self.L_EYE_MP_LMKS]).astype(np.float32))
            else:
                gt_landmark_list.append(np.zeros([68, 2], dtype=np.float32))
                gt_eyes_landmark_list.append(np.zeros([10, 2], dtype=np.float32))

        gt_landmark_tensor = torch.from_numpy(np.array(gt_landmark_list)).float().to(self.device)       # [N,68,2]
        gt_eyes_landmark_tensor = torch.from_numpy(np.array(gt_eyes_landmark_list)).float().to(self.device) # [N,10,2]

        return {
            'lmks_dense_list': lmks_dense_list,
            'face_landmarks_list': face_landmarks_list,
            'gt_ear_landmarks_list': gt_ear_landmarks_list,
            'gt_landmark_tensor': gt_landmark_tensor,
            'gt_eyes_landmark_tensor': gt_eyes_landmark_tensor,
            'valid_mask': valid_mask,
        }


    def _run_rigid_fitting_multiview(self, imgs, params, recon_dict_list, lmk_data):
        """Stage 1: rigid fitting on the camera pose (6DoF) for each view."""
        N = len(imgs)
        valid_mask = lmk_data['valid_mask']
        gt_landmark_tensor = lmk_data['gt_landmark_tensor']
        gt_ear_landmarks_list = lmk_data['gt_ear_landmarks_list']

        # prepare FLAME coefficients
        shape = torch.from_numpy(params['shape']).to(self.device).detach()
        exp = torch.from_numpy(params['exp']).to(self.device).detach()
        head_pose = torch.from_numpy(params['head_pose']).to(self.device).detach()
        jaw_pose = torch.from_numpy(params['jaw_pose']).to(self.device).detach()
        head_pose *= 0  # clear FLAME's head pose (we use camera pose instead)

        # prepare per-view 6DoF camera poses
        camera_pose = torch.zeros([1, 6], dtype=torch.float32, device=self.device)
        camera_pose[0, -1] = self.distance

        fov_val = torch.tensor([self.fov], dtype=torch.float32, device=self.device)

        d_camera_rotation_list = []
        d_camera_translation_list = []
        all_params = []
        for _ in range(N):
            d_rot = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32, device=self.device))
            d_trans = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32, device=self.device))
            d_camera_rotation_list.append(d_rot)
            d_camera_translation_list.append(d_trans)
            all_params.extend([d_rot, d_trans])

        e_opt_rigid = torch.optim.Adam([
            {'params': [p for p in d_camera_rotation_list], 'lr': 0.01},
            {'params': [p for p in d_camera_translation_list], 'lr': 0.05},
        ], weight_decay=0.00001)

        # FLAME reconstruction (only once for rigid optimization)
        with torch.no_grad():
            vertices, _, _ = self.flame(shape_params=shape, expression_params=exp,
                                         head_pose_params=head_pose, jaw_pose_params=jaw_pose) # [1, V, 3]
            face_68_vertices = self.flame.seletec_3d68(vertices)           # [1, 68, 3]
            left_ear_vertices = vertices[:, self.L_EAR_INDICES, :]         # [1, 20, 3]
            right_ear_vertices = vertices[:, self.R_EAR_INDICES, :]        # [1, 20, 3]
            concat_vertices = torch.cat([face_68_vertices, left_ear_vertices, right_ear_vertices], dim=1) # [1, 108, 3]

        # compute intrinsics
        K = build_intrinsics(focal_length=fov_to_focal(fov=fov_val, sensor_size=self.H), image_size=self.H) # [1,3,3]

        total_iterations = 1000
        for iter in range(total_iterations):
            e_opt_rigid.zero_grad()

            if iter == 700:
                e_opt_rigid.param_groups[0]['lr'] = 0.005
                e_opt_rigid.param_groups[1]['lr'] = 0.01
            if iter <= 700: l_f = 100; l_c = 500
            else: l_f = 500; l_c = 100

            count = 0
            loss = 0
            for i in range(N):
                if recon_dict_list[i] is None or not valid_mask[i]:
                    continue
                count += 1

                optimized_camera_pose = camera_pose + torch.cat([d_camera_rotation_list[i], d_camera_translation_list[i]], dim=-1) # [1,6]
                concat_verts_clip = batch_perspective_projection(
                    verts=concat_vertices, camera_pose=optimized_camera_pose,
                    K=K, image_size=self.H, near=self.znear, far=self.zfar) # [1, 108, 3]
                concat_verts_ndc = batch_verts_clip_to_ndc(concat_verts_clip) # [1, 108, 3]

                landmarks2d = concat_verts_ndc[:, :68, :2]  # [1, 68, 2]

                # ear landmarks loss
                EAR_LOSS_THRESHOLD = 0.2
                loss_ear = 0
                if self.use_ear_landmarks and gt_ear_landmarks_list[i] is not None:
                    left_ear_2d = concat_verts_ndc[:, 68:88, :2]   # [1, 20, 2]
                    right_ear_2d = concat_verts_ndc[:, 88:108, :2] # [1, 20, 2]
                    gt_ear_landmark = gt_ear_landmarks_list[i]
                    loss_l_ear = compute_l2_distance_per_sample(left_ear_2d, gt_ear_landmark)
                    loss_r_ear = compute_l2_distance_per_sample(right_ear_2d, gt_ear_landmark)
                    if loss_l_ear < EAR_LOSS_THRESHOLD:
                        loss_ear = loss_ear + loss_l_ear
                    if loss_r_ear < EAR_LOSS_THRESHOLD:
                        loss_ear = loss_ear + loss_r_ear
                loss_ear = loss_ear * 100

                gt_landmark = gt_landmark_tensor[i][None]
                loss_facial = compute_l2_distance_per_sample(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]).sum() * l_f
                loss_contour = compute_l2_distance_per_sample(landmarks2d[:, :17, :2], gt_landmark[:, :17, :2]).sum() * l_c
                loss = loss + loss_facial + loss_contour + loss_ear

            if count == 0:
                return None
            loss = loss / count
            loss.backward()
            e_opt_rigid.step()

        # collect optimized camera poses
        optimized_camera_pose_list = []
        for i in range(N):
            ocp = camera_pose + torch.cat([d_camera_rotation_list[i], d_camera_translation_list[i]], dim=-1)
            optimized_camera_pose_list.append(ocp.detach())

        return {
            'camera_pose': camera_pose,
            'fov_val': fov_val,
            'K': K,
            'optimized_camera_pose_list': optimized_camera_pose_list,
            'd_camera_rotation_list': d_camera_rotation_list,
            'd_camera_translation_list': d_camera_translation_list,
            'shape': shape,
            'exp': exp,
            'head_pose': head_pose,
            'jaw_pose': jaw_pose,
        }


    def run_fitting_multiview(self, imgs, recon_dict_list, params, shape_code):
        """Landmark-based multi-view fitting.
            - Stage 1: rigid fitting on the camera pose (6DoF) based on detected landmarks
            - Stage 2: fine-tune expression, jaw pose, and eye pose
        """
        N = len(imgs)

        # resize for FLAME fitting
        imgs_resized = [cv2.resize(img, (self.RENDER_SIZE, self.RENDER_SIZE)) for img in imgs]

        # detect landmarks
        lmk_data = self._detect_landmarks_for_fitting(imgs_resized)
        gt_landmark_tensor = lmk_data['gt_landmark_tensor']
        gt_eyes_landmark_tensor = lmk_data['gt_eyes_landmark_tensor']
        valid_mask = lmk_data['valid_mask']

        ############################################################
        ## Stage 1: rigid fitting (estimate the 6DoF camera pose)  #
        ############################################################
        rigid_result = self._run_rigid_fitting_multiview(imgs, params, recon_dict_list, lmk_data)
        if rigid_result is None:
            return None

        shape = rigid_result['shape']
        exp = rigid_result['exp']
        head_pose = rigid_result['head_pose']
        jaw_pose = rigid_result['jaw_pose']
        K = rigid_result['K']
        optimized_camera_pose_list = rigid_result['optimized_camera_pose_list']

        ############################
        ## Stage 2: fine fitting   #
        ############################
        d_exp = nn.Parameter(torch.zeros(params['exp'].shape, dtype=torch.float32, device=self.device))
        d_jaw = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32, device=self.device))
        eye_pose = nn.Parameter(torch.zeros(1, 6, dtype=torch.float32, device=self.device))

        e_opt_fine = torch.optim.Adam([
            {'params': [d_exp], 'lr': 0.01},
            {'params': [d_jaw], 'lr': 0.025},
            {'params': [eye_pose], 'lr': 0.03},
        ], weight_decay=0.0001)

        for iter in range(200):
            e_opt_fine.zero_grad()

            if iter == 100:
                e_opt_fine.param_groups[0]['lr'] = 0.005
                e_opt_fine.param_groups[1]['lr'] = 0.01
                e_opt_fine.param_groups[2]['lr'] = 0.01

            optimized_exp = exp + d_exp
            optimized_jaw_pose = jaw_pose + d_jaw
            vertices, _, _ = self.flame(shape_params=shape,
                                         expression_params=optimized_exp,
                                         head_pose_params=head_pose,
                                         jaw_pose_params=optimized_jaw_pose,
                                         eye_pose_params=eye_pose)

            count = 0
            loss = 0
            for i in range(N):
                if recon_dict_list[i] is None or not valid_mask[i]:
                    continue
                count += 1

                ocp = optimized_camera_pose_list[i]
                verts_clip = batch_perspective_projection(
                    verts=vertices, camera_pose=ocp,
                    K=K, image_size=self.H, near=self.znear, far=self.zfar)
                verts_ndc_3d = batch_verts_clip_to_ndc(verts_clip)

                landmarks2d = self.flame.seletec_3d68(verts_ndc_3d)[:, :, :2]
                eyes_landmarks2d = verts_ndc_3d[:, self.R_EYE_INDICES + self.L_EYE_INDICES, :2]

                gt_landmark = gt_landmark_tensor[i][None]
                gt_eyes_landmark = gt_eyes_landmark_tensor[i][None]

                loss_facial = compute_l2_distance_per_sample(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]).sum() * 500
                loss_eyes = compute_l2_distance_per_sample(eyes_landmarks2d, gt_eyes_landmark).sum() * 500
                loss = loss + loss_facial + loss_eyes

            if count == 0:
                return None
            loss = loss / count
            loss.backward()
            e_opt_fine.step()

        ##############################
        ## for displaying results    #
        ##############################
        with torch.no_grad():
            optimized_exp = exp + d_exp
            optimized_jaw_pose = jaw_pose + d_jaw
            vertices, _, _ = self.flame(shape_params=shape,
                                         expression_params=optimized_exp,
                                         head_pose_params=head_pose,
                                         jaw_pose_params=optimized_jaw_pose,
                                         eye_pose_params=eye_pose)

            rendered_mesh_shape_img_list = []
            rendered_mesh_shape_list = []
            cam_numpy_list = []
            for i in range(N):
                ocp = optimized_camera_pose_list[i]
                cam_numpy_list.append(ocp.detach().cpu().numpy())

                if recon_dict_list[i] is None or not valid_mask[i]:
                    rendered_mesh_shape_img_list.append(None)
                    rendered_mesh_shape_list.append(None)
                    continue

                img_resized = imgs_resized[i]

                verts_clip = batch_perspective_projection(
                    verts=vertices, camera_pose=ocp,
                    K=K, image_size=self.H, near=self.znear, far=self.zfar)
                verts_ndc_3d = batch_verts_clip_to_ndc(verts_clip)
                verts_screen_3d = batch_verts_ndc_to_screen(verts_ndc_3d, image_size=self.H)

                landmarks_3d_screen = self.flame.seletec_3d68(verts_screen_3d).detach().cpu().numpy()
                landmarks_2d_screen = landmarks_3d_screen[:,:,:2]
                verts_screen_2d = verts_screen_3d.detach().cpu().numpy()
                eye_landmarks2d_screen = verts_screen_2d[:, self.R_EYE_INDICES + self.L_EYE_INDICES, :]
                ear_landmarks2d_screen = verts_screen_2d[:, self.R_EAR_INDICES + self.L_EAR_INDICES, :]

                # build a minimal in_dict for visualize helper
                in_dict_single = {'img_resized': np.array([img_resized], dtype=np.uint8)}
                img_rendered, mesh_rendered = prepare_batch_visualized_results(
                    vertices, self.faces, in_dict_single, verts_ndc_3d, self.RENDER_SIZE,
                    landmarks_2d_screen, eye_landmarks2d_screen, ear_landmarks2d_screen)

                rendered_mesh_shape_img_list.append(img_rendered[0])
                rendered_mesh_shape_list.append(mesh_rendered[0])

        ####################
        # Prepare results  #
        ####################
        ret_dict = {
            'vertices': vertices[0].detach().cpu().numpy(),
            'shape': params['shape'],
            'exp': optimized_exp.detach().cpu().numpy(),
            'head_pose': head_pose.detach().cpu().numpy(),
            'jaw_pose': optimized_jaw_pose.detach().cpu().numpy(),
            'eye_pose': eye_pose.detach().cpu().numpy(),
            'tex': params['tex'],
            'light': params['light'],
            'cam': cam_numpy_list,
            'img_rendered': rendered_mesh_shape_img_list,
            'mesh_rendered': rendered_mesh_shape_list,
        }

        return ret_dict


    def run_fitting_multiview_photometric(self, imgs, face_mask_list, recon_dict_list, params, shape_code):
        """Photometric multi-view fitting (aligned with single-view tricks).
            - Stage 1: rigid fitting on the camera pose (6DoF) based on detected landmarks
            - Stage 2: fine-tune shape, tex, exp, pose, eye_pose, and light with photometric loss
              Uses staged texture optimization, per-landmark weights, neck pose,
              yaw-aware ear fitting, and texture displacement map.
        """
        N = len(imgs)

        # resize for FLAME fitting
        imgs_resized = [cv2.resize(img, (self.RENDER_SIZE, self.RENDER_SIZE)) for img in imgs]

        gt_imgs = []
        gt_face_masks = []
        for i in range(N):
            gt_img = torch.from_numpy(np.array(imgs_resized[i], dtype=np.float32) / 255.).to(self.device)
            gt_img = gt_img[None].permute(0, 3, 1, 2)  # [1,C,H,W]
            gt_imgs.append(gt_img)

            face_mask_resized = cv2.resize(face_mask_list[i], (self.RENDER_SIZE, self.RENDER_SIZE))
            gt_face_mask = torch.from_numpy(face_mask_resized)[None].to(self.device)
            gt_face_masks.append(gt_face_mask)

        # detect landmarks
        lmk_data = self._detect_landmarks_for_fitting(imgs_resized)
        gt_landmark_tensor = lmk_data['gt_landmark_tensor']
        gt_eyes_landmark_tensor = lmk_data['gt_eyes_landmark_tensor']
        gt_ear_landmarks_list = lmk_data['gt_ear_landmarks_list']
        valid_mask = lmk_data['valid_mask']

        ############################################################
        ## Stage 1: rigid fitting (estimate the 6DoF camera pose)  #
        ############################################################
        rigid_result = self._run_rigid_fitting_multiview(imgs, params, recon_dict_list, lmk_data)
        if rigid_result is None:
            return None

        shape = rigid_result['shape']
        exp = rigid_result['exp']
        head_pose = rigid_result['head_pose']
        jaw_pose = rigid_result['jaw_pose']
        camera_pose = rigid_result['camera_pose']
        K = rigid_result['K']
        d_camera_rotation_list = rigid_result['d_camera_rotation_list']
        d_camera_translation_list = rigid_result['d_camera_translation_list']
        tex = torch.from_numpy(params['tex']).to(self.device).detach()
        light = torch.from_numpy(params['light']).to(self.device).detach()

        ############################
        ## Stage 2: fine fitting   #
        ############################

        # shape offset (shared across views since same identity)
        d_shape = nn.Parameter(torch.zeros([1, params['shape'].shape[1]], dtype=torch.float32, device=self.device))

        # expression offset (shared across views since same moment)
        d_exp = nn.Parameter(torch.zeros(params['exp'].shape, dtype=torch.float32, device=self.device))

        # jaw pose offset
        d_jaw = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32, device=self.device))

        # neck pose (from single-view)
        d_neck = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32, device=self.device))

        # eye pose
        eye_pose = nn.Parameter(torch.zeros(1, 6, dtype=torch.float32, device=self.device))

        # texture code offset (shared)
        d_tex = nn.Parameter(torch.zeros([1, params['tex'].shape[1]], dtype=torch.float32, device=self.device))

        # texture displacement map (from single-view, for fine pixel-level detail)
        d_texture = nn.Parameter(torch.zeros([1, 3, 256, 256], dtype=torch.float32, device=self.device))

        # light offset (per-view, since lighting may differ across cameras)
        d_light = nn.Parameter(torch.zeros(params['light'].shape, dtype=torch.float32, device=self.device))

        # make camera params optimizable again in Stage 2
        for p in d_camera_rotation_list + d_camera_translation_list:
            p.requires_grad_(True)

        finetune_params = [
            {'params': [d_exp], 'lr': 0.005},
            {'params': [d_jaw], 'lr': 0.005},
            {'params': [d_neck], 'lr': 0.005},
            {'params': [eye_pose], 'lr': 0.005},
            {'params': d_camera_rotation_list, 'lr': 0.0025},
            {'params': d_camera_translation_list, 'lr': 0.0025},
            {'params': [d_light], 'lr': 0.005},
            {'params': [d_shape], 'lr': 0.001},
            {'params': [d_texture], 'lr': 0.005},
            {'params': [d_tex], 'lr': 0.005},
        ]

        e_opt_fine = torch.optim.Adam(
            finetune_params,
            weight_decay=0.00001  # matched to single-view
        )

        # initialize texture from FLAME tex model
        texture = torch.clamp(self.flametex(tex + d_tex) + d_texture, 0.0, 1.0).detach()

        # per-landmark confidence weights (from single-view)
        with torch.no_grad():
            lmk_weights = torch.ones([1, 68], dtype=torch.float32, device=self.device)
            lmk_weights[:, 17:]   = 1.5      # face 51 landmarks
            lmk_weights[:, 0:17]  = 0.75     # jawline
            lmk_weights[:, 0:3]   = 0.5      # jawline corners
            lmk_weights[:, 14:17] = 0.5      # jawline corners
            lmk_weights[:, 36:48] = 3.0      # eye contours
            lmk_weights[:, 49:]   = 2.0      # mouth contours
            lmk_weights[:, 31:36] = 0.75     # nose bottom line

        # optimization loop
        max_iterations = 400
        for iter in range(max_iterations):

            optimized_shape = shape + d_shape
            optimized_exp = exp + d_exp
            optimized_jaw_pose = jaw_pose + d_jaw
            optimized_neck_pose = d_neck
            vertices, _, _ = self.flame(shape_params=optimized_shape,
                                         expression_params=optimized_exp,
                                         head_pose_params=head_pose,
                                         jaw_pose_params=optimized_jaw_pose,
                                         neck_pose_params=optimized_neck_pose,
                                         eye_pose_params=eye_pose)

            # staged texture optimization (from single-view)
            if iter < 200:
                texture = torch.clamp(self.flametex(tex + d_tex), 0.0, 1.0)
            else:
                texture = torch.clamp(self.flametex(tex + d_tex) + d_texture, 0.0, 1.0)

            count = 0
            loss = 0
            for i in range(N):
                if recon_dict_list[i] is None or not valid_mask[i]:
                    continue
                count += 1

                optimized_camera_pose = camera_pose + torch.cat([d_camera_rotation_list[i], d_camera_translation_list[i]], dim=-1)

                gt_landmark = gt_landmark_tensor[i][None]
                gt_eyes_landmark = gt_eyes_landmark_tensor[i][None]
                gt_face_mask = gt_face_masks[i]
                gt_img = gt_imgs[i]

                verts_clip = batch_perspective_projection(
                    verts=vertices, camera_pose=optimized_camera_pose,
                    K=K, image_size=self.H, near=self.znear, far=self.zfar)
                verts_ndc_3d = batch_verts_clip_to_ndc(verts_clip)

                landmarks2d = self.flame.seletec_3d68(verts_ndc_3d)[:, :, :2]
                eyes_landmarks2d = verts_ndc_3d[:, self.R_EYE_INDICES + self.L_EYE_INDICES, :2]

                # yaw-aware ear landmarks loss (from single-view)
                EAR_LOSS_THRESHOLD = 0.3
                loss_ear = 0
                if self.use_ear_landmarks and gt_ear_landmarks_list[i] is not None:
                    left_ear_2d = verts_ndc_3d[:, self.L_EAR_INDICES, :2]
                    right_ear_2d = verts_ndc_3d[:, self.R_EAR_INDICES, :2]
                    gt_ear_landmark = gt_ear_landmarks_list[i]

                    with torch.no_grad():
                        euler_angles = rotation_vector_to_euler_angles(rot_vec=optimized_camera_pose[:, :3])
                        yaw_angles = euler_angles[:, 0].detach()
                        mask_use_l = (yaw_angles > 0).float()
                        mask_use_r = (yaw_angles < 0).float()
                        abs_yaw_angles = torch.abs(yaw_angles)
                        abs_yaw_angles[abs_yaw_angles < 0.1] = 0.0

                    loss_l_ear = compute_l2_distance_per_sample(left_ear_2d, gt_ear_landmark)
                    mask_l_ear = (loss_l_ear < EAR_LOSS_THRESHOLD).float()
                    loss_l_ear = loss_l_ear * mask_l_ear * mask_use_l

                    loss_r_ear = compute_l2_distance_per_sample(right_ear_2d, gt_ear_landmark)
                    mask_r_ear = (loss_r_ear < EAR_LOSS_THRESHOLD).float()
                    loss_r_ear = loss_r_ear * mask_r_ear * mask_use_r

                    loss_ear = (loss_l_ear + loss_r_ear) * 1.5 * abs_yaw_angles
                    loss_ear = loss_ear.sum()

                # render textured mesh
                rendered_output = self.flame_texture_render(vertices, verts_ndc_3d, texture, light + d_light)
                rendered_textured = rendered_output['images'][:, :3, :, :]  # [1,3,H,W]

                # losses (weights matched to single-view)
                loss_photo = compute_batch_pixelwise_l1_loss(gt_img, rendered_textured, gt_face_mask) * 2
                loss_lmks = compute_l2_distance_per_sample(landmarks2d[:, :, :2], gt_landmark[:, :, :2], confidence=lmk_weights).sum()
                loss_eyes = compute_l2_distance_per_sample(eyes_landmarks2d, gt_eyes_landmark).sum() * 2
                loss_reg_shape = (torch.sum(d_shape ** 2) / 2) * 0.1
                loss_reg_exp = (torch.sum(optimized_exp ** 2) / 2) * 1e-4
                loss_reg = loss_reg_shape + loss_reg_exp
                loss = loss + loss_photo + loss_lmks + loss_eyes + loss_ear + loss_reg

            if count == 0:
                return None
            loss = loss / count
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4)
            e_opt_fine.zero_grad()
            loss.backward()
            e_opt_fine.step()

        ##############################
        ## for displaying results    #
        ##############################
        optimized_camera_pose_list = []
        for i in range(N):
            ocp = camera_pose + torch.cat([d_camera_rotation_list[i], d_camera_translation_list[i]], dim=-1)
            optimized_camera_pose_list.append(ocp.detach())

        with torch.no_grad():
            optimized_shape = shape + d_shape
            optimized_exp = exp + d_exp
            optimized_jaw_pose = jaw_pose + d_jaw
            optimized_neck_pose = d_neck
            vertices, _, _ = self.flame(shape_params=optimized_shape,
                                         expression_params=optimized_exp,
                                         head_pose_params=head_pose,
                                         jaw_pose_params=optimized_jaw_pose,
                                         neck_pose_params=optimized_neck_pose,
                                         eye_pose_params=eye_pose)
            if max_iterations < 200:
                texture = torch.clamp(self.flametex(tex + d_tex), 0.0, 1.0)
            else:
                texture = torch.clamp(self.flametex(tex + d_tex) + d_texture, 0.0, 1.0)

            rendered_mesh_shape_img_list = []
            rendered_mesh_shape_list = []
            cam_numpy_list = []
            for i in range(N):
                ocp = optimized_camera_pose_list[i]
                cam_numpy_list.append(ocp.detach().cpu().numpy())

                if recon_dict_list[i] is None or not valid_mask[i]:
                    rendered_mesh_shape_img_list.append(None)
                    rendered_mesh_shape_list.append(None)
                    continue

                img_resized = imgs_resized[i]

                verts_clip = batch_perspective_projection(
                    verts=vertices, camera_pose=ocp,
                    K=K, image_size=self.H, near=self.znear, far=self.zfar)
                verts_ndc_3d = batch_verts_clip_to_ndc(verts_clip)
                verts_screen_3d = batch_verts_ndc_to_screen(verts_ndc_3d, image_size=self.H)

                landmarks_3d_screen = self.flame.seletec_3d68(verts_screen_3d).detach().cpu().numpy()
                landmarks_2d_screen = landmarks_3d_screen[:, :, :2]
                verts_screen_2d = verts_screen_3d.detach().cpu().numpy()
                eye_landmarks2d_screen = verts_screen_2d[:, self.R_EYE_INDICES + self.L_EYE_INDICES, :]
                ear_landmarks2d_screen = verts_screen_2d[:, self.R_EAR_INDICES + self.L_EAR_INDICES, :]

                in_dict_single = {'img_resized': np.array([img_resized], dtype=np.uint8)}
                img_rendered, mesh_rendered = prepare_batch_visualized_results(
                    vertices, self.faces, in_dict_single, verts_ndc_3d, self.RENDER_SIZE,
                    landmarks_2d_screen, eye_landmarks2d_screen, ear_landmarks2d_screen)

                rendered_mesh_shape_img_list.append(img_rendered[0])
                rendered_mesh_shape_list.append(mesh_rendered[0])

                # also render textured mesh for this view
                rendered_tex = self.flame_texture_render(vertices, verts_ndc_3d, texture, light + d_light)
                rendered_tex_img = rendered_tex['images'][:, :3, :, :]
                rendered_tex_img = rendered_tex_img.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
                rendered_tex_img = np.array(np.clip(rendered_tex_img * 255, 0, 255), dtype=np.uint8)
                # override img_rendered with textured rendering for photometric mode
                rendered_mesh_shape_img_list[-1] = rendered_tex_img

        ####################
        # Prepare results  #
        ####################
        ret_dict = {
            'vertices': vertices[0].detach().cpu().numpy(),
            'shape': optimized_shape.detach().cpu().numpy(),
            'exp': optimized_exp.detach().cpu().numpy(),
            'head_pose': head_pose.detach().cpu().numpy(),
            'jaw_pose': optimized_jaw_pose.detach().cpu().numpy(),
            'neck_pose': optimized_neck_pose.detach().cpu().numpy(),
            'eye_pose': eye_pose.detach().cpu().numpy(),
            'tex': (tex + d_tex).detach().cpu().numpy(),
            'texture': texture.detach().cpu().numpy(),
            'light': (light + d_light).detach().cpu().numpy(),
            'cam': cam_numpy_list,
            'img_rendered': rendered_mesh_shape_img_list,
            'mesh_rendered': rendered_mesh_shape_list,
        }

        return ret_dict

    # ── UV Texture Extraction ──────────────────────────────────────────

    def extract_uv_texture_from_views(self, ret_dict, uv_size=512):
        """
        Extract UV texture map from aligned images by projecting image pixels into FLAME UV space.
        For multi-view, combines textures from all views weighted by face-normal vs view-direction.

        Args:
            ret_dict: output from load_images_and_run (must include 'vertices', 'cam', 'img_aligned')
            uv_size: resolution of the output UV texture map

        Returns:
            uv_texture: (uv_size, uv_size, 3) uint8 UV texture map
        """
        from submodules.flame_fitting.fitting_util import face_vertices as get_face_vertices
        from submodules.flame_fitting.renderer import Pytorch3dRasterizer

        device = self.device
        renderer = self.flame_texture_render

        vertices = torch.from_numpy(ret_dict['vertices']).float().to(device).unsqueeze(0)  # [1, V, 3]
        cam_list = ret_dict['cam']
        imgs_aligned = ret_dict['img_aligned']  # list of aligned images used during fitting

        batch_size = 1
        faces = renderer.faces.expand(batch_size, -1, -1)  # [1, F, 3]

        # Compute vertex normals for view-weighting
        normals = fitting_util.vertex_normals(vertices, faces)  # [1, V, 3]

        # Build UV rasterizer at desired resolution
        if renderer.uv_rasterizer.raster_settings.image_size == uv_size:
            uv_rasterizer = renderer.uv_rasterizer
        else:
            uv_rasterizer = Pytorch3dRasterizer(uv_size).to(device)

        # Get face vertices in world space: [1, F, 3, 3]
        face_verts_world = get_face_vertices(vertices, faces)

        # Rasterize in UV space to get world-space positions per UV pixel
        uv_rendering = uv_rasterizer(
            renderer.uvcoords.expand(batch_size, -1, -1),
            renderer.uvfaces.expand(batch_size, -1, -1),
            face_verts_world
        )  # [1, 3+1, uv_size, uv_size]
        uv_world_pos = uv_rendering[:, :3, :, :]  # [1, 3, H, W]
        uv_mask = uv_rendering[:, 3:, :, :] > 0   # [1, 1, H, W]

        # Also rasterize normals into UV space
        face_normals_attr = get_face_vertices(normals, faces)
        uv_normal_rendering = uv_rasterizer(
            renderer.uvcoords.expand(batch_size, -1, -1),
            renderer.uvfaces.expand(batch_size, -1, -1),
            face_normals_attr
        )
        uv_normals = F.normalize(uv_normal_rendering[:, :3, :, :], dim=1)  # [1, 3, H, W]

        # Accumulate texture from each view
        accum_texture = torch.zeros(1, 3, uv_size, uv_size, device=device)
        accum_weight = torch.zeros(1, 1, uv_size, uv_size, device=device)

        focal_tensor = torch.tensor([self.DEFAULT_FOCAL], dtype=torch.float32, device=device)
        K = build_intrinsics(focal_tensor, self.H)

        for i, cam_np in enumerate(cam_list):
            if cam_np is None:
                continue

            # Prepare aligned image as tensor [1, 3, H, W]
            img = imgs_aligned[i]
            if img.shape[0] != self.H or img.shape[1] != self.W:
                img = cv2.resize(img, (self.W, self.H))
            img_tensor = torch.from_numpy(img).float().to(device) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

            # Project UV world positions to screen coordinates
            camera_pose = torch.from_numpy(cam_np).float().to(device)  # [1, 6]
            uv_pos_flat = uv_world_pos.permute(0, 2, 3, 1).reshape(1, -1, 3)  # [1, N_pixels, 3]

            verts_clip = batch_perspective_projection(
                verts=uv_pos_flat, camera_pose=camera_pose,
                K=K, image_size=self.H, near=self.znear, far=self.zfar
            )
            verts_ndc = batch_verts_clip_to_ndc(verts_clip)
            verts_screen = batch_verts_ndc_to_screen(verts_ndc, image_size=self.H)

            # Convert screen coords to grid_sample coords [-1, 1]
            screen_xy = verts_screen[0, :, :2]  # [N_pixels, 2]
            grid_x = (screen_xy[:, 0] / (self.W - 1)) * 2 - 1
            grid_y = (screen_xy[:, 1] / (self.H - 1)) * 2 - 1
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, uv_size, uv_size, 2)

            # Sample image at projected positions
            sampled = F.grid_sample(img_tensor, grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=True)  # [1, 3, uv_size, uv_size]

            # Compute view direction weight: dot(surface_normal, view_direction)
            # Camera position in world space = translation from camera_pose
            cam_translation = camera_pose[:, 3:].reshape(1, 1, 3)
            view_dir = F.normalize(cam_translation.expand_as(uv_pos_flat) - uv_pos_flat, dim=-1)
            view_dir_map = view_dir.reshape(1, uv_size, uv_size, 3).permute(0, 3, 1, 2)  # [1, 3, H, W]

            weight = (uv_normals * view_dir_map).sum(dim=1, keepdim=True)  # [1, 1, H, W]
            weight = torch.clamp(weight, min=0.0)

            # Mask: valid UV pixels and in-bounds projections
            in_bounds = (grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1)
            valid = uv_mask & in_bounds.unsqueeze(1)
            weight = weight * valid.float()

            accum_texture += sampled * weight
            accum_weight += weight

        # Normalize accumulated texture
        accum_weight = torch.clamp(accum_weight, min=1e-8)
        uv_texture = accum_texture / accum_weight

        # Convert to numpy uint8
        uv_texture = uv_texture[0].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 3]
        uv_texture = np.clip(uv_texture * 255, 0, 255).astype(np.uint8)

        return uv_texture

    def save_flame_obj_with_texture(self, ret_dict, uv_texture, output_path='flame_textured.obj'):
        """
        Save FLAME mesh as OBJ with UV-mapped texture.

        Args:
            ret_dict: output from load_images_and_run
            uv_texture: (H, W, 3) uint8 UV texture map
            output_path: path for the output .obj file (texture saved as .png alongside)
        """
        import os
        vertices = torch.from_numpy(ret_dict['vertices']).float()  # [V, 3]
        tex_tensor = torch.from_numpy(uv_texture).float() / 255.0  # [H, W, 3]
        tex_tensor = tex_tensor.permute(2, 0, 1)  # [3, H, W]

        self.flame_texture_render.save_obj(output_path, vertices, tex_tensor)

        # Also save UV texture as standalone image
        tex_path = os.path.splitext(output_path)[0] + '_uv_texture.png'
        cv2.imwrite(tex_path, cv2.cvtColor(uv_texture, cv2.COLOR_RGB2BGR))
        print(f'Saved: {output_path}, {tex_path}')

    def save_regressed_head(self, ret_dict, output_path='flame_regressed.obj'):
        """
        Save the regressed FLAME head mesh with the fitted PCA texture (from photometric fitting).
        This uses ret_dict['texture'] [1, 3, 256, 256] which is already in UV space.

        Args:
            ret_dict: output from load_images_and_run (with photometric_fitting=True)
            output_path: path for the output .obj file
        """
        import os
        vertices = torch.from_numpy(ret_dict['vertices']).float()  # [V, 3]
        texture = ret_dict['texture']  # [1, 3, 256, 256]
        tex_tensor = torch.from_numpy(texture[0]).float()  # [3, 256, 256]

        self.flame_texture_render.save_obj(output_path, vertices, tex_tensor)

        # Save UV texture map as standalone image
        uv_img = tex_tensor.permute(1, 2, 0).numpy()  # [256, 256, 3]
        uv_img = np.clip(uv_img * 255, 0, 255).astype(np.uint8)
        tex_path = os.path.splitext(output_path)[0] + '_uv_texture.png'
        cv2.imwrite(tex_path, cv2.cvtColor(uv_img, cv2.COLOR_RGB2BGR))
        print(f'Saved: {output_path}, {tex_path}')
