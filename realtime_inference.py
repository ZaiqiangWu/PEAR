from models.modules.ehm import EHM_v2 
from models.pipeline.ehm_pipeline import Ehm_Pipeline
import os
import torch
from utils.pipeline_utils import to_tensor
from utils.graphics_utils import GS_Camera
from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from pytorch3d.renderer import PointLights
import cv2
import argparse
import numpy as np
import torchvision.transforms as transforms


# Monkey-patch torch.load for PyTorch >= 2.6 (weights_only=True breaks ultralytics)
_torch_load_orig = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(*args, **kwargs)
torch.load = _safe_torch_load

from ultralytics import YOLO
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import (
    ConfigDict, device_parser, add_extra_cfgs
)
from huggingface_hub import hf_hub_download


# ── helpers: copied verbatim from inference_images.py ───────────

def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])

    return bbox


def process_bbox(bbox, img_width, img_height, input_img_shape, ratio=1.25):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = input_img_shape[1] / input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * ratio
    bbox[3] = h * ratio
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    bbox = bbox.astype(np.float32)
    return bbox


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                        inv=True)

    return img_patch, trans, inv_trans


def build_cameras_kwargs(batch_size, focal_length):
    screen_size = torch.tensor([1024, 1024]).float()[None].repeat(batch_size, 1)
    cameras_kwargs = {
        'principal_point': torch.zeros(batch_size, 2).float(),
        'focal_length': focal_length,
        'image_size': screen_size, 'device': "cuda",
    }
    return cameras_kwargs


# ── main realtime loop ─────────────────────────────────────────

def realtime_inference(config_name, devices, camera_id=0):

    meta_cfg = ConfigDict(
        model_config_path=os.path.join('configs', f'{config_name}.yaml')
    )
    meta_cfg = add_extra_cfgs(meta_cfg)
    target_devices = device_parser(devices)
    print(str(meta_cfg))

    body_renderer = BodyRenderer("assets/SMPLX", 1024, focal_length=24.0).cuda()

    repo_id = "BestWJH/PEAR_models"
    filename = "ehm_model_stage1.pt"
    ehm_basemodel = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")

    ehm_model = Ehm_Pipeline(meta_cfg)
    _state = torch.load(ehm_basemodel, map_location='cpu', weights_only=True)
    ehm_model.backbone.load_state_dict(_state['backbone'], strict=False)
    ehm_model.head.load_state_dict(_state['head'], strict=False)
    ehm_model = ehm_model.cuda()

    ehm = EHM_v2("assets/FLAME", "assets/SMPLX")
    ehm = ehm.cuda()

    lights = PointLights(device='cuda:0', location=[[0.0, -1.0, -10.0]])

    detector = YOLO('yolov8x.pt')
    transform = transforms.ToTensor()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: could not open camera {camera_id}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting real-time inference. Press 'q' to quit.")

    import time
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # ── BGR→RGB float32 (mirrors load_img without scale) ──
        original_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        original_img_height, original_img_width = original_img.shape[:2]

        # ── YOLO detection ──
        yolo_bbox = detector.predict(original_img,
                                    device='cuda', classes=0, conf=0.5,
                                    save=False, verbose=False)[0].boxes.xyxy.detach().cpu().numpy()

        vis_img = cv2.cvtColor(original_img.copy(), cv2.COLOR_RGB2BGR)

        if len(yolo_bbox) < 1:
            vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
            cv2.imshow("PEAR Real-Time Mesh", vis_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        num_bbox = len(yolo_bbox)

        # ── loop all detected bboxes (identical to inference_images.py) ──
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])

            bbox = process_bbox(bbox=yolo_bbox_xywh,
                                img_width=original_img_width,
                                img_height=original_img_height,
                                input_img_shape=[256, 256],
                                ratio=1.25)
            if bbox is None:
                continue

            img_patch, trans, inv_trans = generate_patch_image(cvimg=original_img,
                                                              bbox=bbox, scale=1.0, rot=0.0,
                                                              do_flip=False, out_shape=[256, 256])

            img_patch = transform(img_patch.astype(np.float32)) / 255
            img_patch = img_patch.unsqueeze(0).cuda()

            outputs = ehm_model(img_patch)
            pd_smplx_dict = ehm(outputs['body_param'], outputs['flame_param'], pose_type='aa')

            pd_camera = GS_Camera(**build_cameras_kwargs(1, 24),
                                  R=outputs['pd_cam'][0:0+1, :3, :3],
                                  T=outputs['pd_cam'][0:0+1, :3, 3])

            pd_mesh_img = body_renderer.render_mesh(
                pd_smplx_dict['vertices'][None, 0, ...], pd_camera, lights=lights)
            pd_mesh_img = (pd_mesh_img[:, :3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
            pd_mesh_img = cv2.cvtColor(pd_mesh_img.copy(), cv2.COLOR_RGB2BGR)
            pd_mesh_img = cv2.resize(pd_mesh_img, (256, 256), interpolation=cv2.INTER_AREA)

            H, W = original_img.shape[:2]

            mesh_on_orig = cv2.warpAffine(
                pd_mesh_img, inv_trans, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # ── KEY: use same mask logic as inference_images.py ──
            # The renderer background is [50,50,50] so we need threshold > 60 to
            # exclude both the dark-gray bg AND any interpolation noise near borders
            mesh_on_orig_float = mesh_on_orig.astype(np.float32)
            mesh_brightness = np.max(mesh_on_orig_float, axis=-1)
            mask = (mesh_brightness > 60) & (mesh_brightness < 240)

            vis_img[mask] = mesh_on_orig_float[mask]

        if num_bbox == 0:
            continue

        vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)

        frame_count += 1
        elapsed = time.time() - prev_time
        if frame_count % 30 == 0:
            fps = 30.0 / max(elapsed, 1e-6)
            print(f"  FPS (avg): {fps:.1f}   frames: {frame_count}")
            prev_time = time.time()

        cv2.imshow("PEAR Real-Time Mesh", vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-c', default="infer", type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    parser.add_argument('--camera_id', default=0, type=int, help="Camera device index")
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    torch.set_float32_matmul_precision('high')
    realtime_inference(args.config_name, args.devices, args.camera_id)
