from models.modules.ehm import EHM_v2 
from models.pipeline.ehm_pipeline import Ehm_Pipeline
import os
import torch
from utils.pipeline_utils import to_tensor
from utils.graphics_utils import GS_Camera
from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from pytorch3d.renderer import PointLights
import cv2
import os
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch
import cv2
from ultralytics import YOLO
import os
import torch
import argparse
import lightning
import numpy as np
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import (
    ConfigDict, device_parser, add_extra_cfgs
)
from huggingface_hub import hf_hub_download


from utils.get_video import images_to_video

def calculate_iou(bbox1, bbox2):
    # Calculate the Intersection over Union (IoU) between two bounding boxes
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    iou = intersection_area / union_area
    return iou


def non_max_suppression(bboxes, iou_threshold):
    # Sort the bounding boxes by their confidence scores (e.g., the probability of containing an object)
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    
    # Initialize a list to store the selected bounding boxes
    selected_bboxes = []
    
    # Perform non-maximum suppression
    while len(bboxes) > 0:
        current_bbox = bboxes[0]
        selected_bboxes.append(current_bbox)
        bboxes = bboxes[1:]
        
        remaining_bboxes = []
        for bbox in bboxes:
            iou = calculate_iou(current_bbox, bbox)
            if iou < iou_threshold:
                remaining_bboxes.append(bbox)
                
        bboxes = remaining_bboxes
        
    return selected_bboxes

def pad_and_resize(img, target_size=512):
    h, w = img.shape[:2]

    # 等比例缩放
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建黑色背景图像并粘贴到中心
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return padded_img
    

def build_cameras_kwargs(batch_size,focal_length):
    screen_size = torch.tensor([1024, 1024]).float()[None].repeat(batch_size, 1)
    cameras_kwargs = {
        'principal_point': torch.zeros(batch_size, 2).float(), 
        'focal_length': focal_length, 
        'image_size': screen_size, 'device': "cuda",
    }
    return cameras_kwargs

def load_img(path, order='RGB', scale=2):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    # 放大 2 倍
    if scale != 1:
        h, w = img.shape[:2]
        img = cv2.resize(
            img,
            (w * scale, h * scale),
            interpolation=cv2.INTER_CUBIC  
        )

    img = img.astype(np.float32)
    return img

def get_bbox(joint_img, joint_valid, extend_ratio=1.2):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.
    width = xmax - xmin
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio

    y_center = (ymin + ymax) / 2.
    height = ymax - ymin
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        bbox = None

    return bbox

def process_bbox(bbox, img_width, img_height, input_img_shape, ratio=1.25):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    # aspect ratio preserving bbox
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
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
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

def inference( config_name, devices, input_path=None, output_path = None):
    # build config
    # if torch.distributed.is_available():
    #     torch.distributed.init_process_group(backend="nccl")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    meta_cfg = ConfigDict(
        model_config_path=os.path.join('configs', f'{config_name}.yaml')
    )
    meta_cfg = add_extra_cfgs(meta_cfg)
    lightning.fabric.seed_everything(10)
    target_devices = device_parser(devices)
    init_iter = 1
    print(str(meta_cfg))
    # setup model and optimizer

    body_renderer = BodyRenderer("assets/SMPLX", 1024 , focal_length=24.0 ).cuda() 


    repo_id = "BestWJH/PEAR_models"  
    filename = "pear_model.pt"  

    ehm_basemodel = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    ehm_model = Ehm_Pipeline(meta_cfg)
    _state=torch.load(ehm_basemodel, map_location='cpu', weights_only=True)
    ehm_model.backbone.load_state_dict(_state['backbone'], strict=False)
    ehm_model.head.load_state_dict(_state['head'], strict=False)
    ehm_model = ehm_model.cuda()
    ehm_model.eval()

    ehm = EHM_v2( "assets/FLAME", "assets/SMPLX")
    ehm = ehm.cuda()

    lights=PointLights(device='cuda:0', location=[[0.0, -1.0, -10.0]])



    # init detector
    bbox_model = './model_zoo/yolov8x.pt'
    detector = YOLO(bbox_model)



    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_paths = [os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.lower().endswith(image_extensions)]
    all_time = 0   
    transform = transforms.ToTensor()
    image_paths = sorted(image_paths)
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]
        # detection, xyxy
        yolo_bbox = detector.predict(original_img,  # [h,w,3]  np
                                device='cuda', 
                                classes=0, 
                                conf=0.5, 
                                save=False, 
                                verbose=False)[0].boxes.xyxy.detach().cpu().numpy()
        
        vis_img =  cv2.cvtColor(original_img.copy(), cv2.COLOR_RGB2BGR) 

        if len(yolo_bbox) <1:
            continue
        num_bbox = len(yolo_bbox)


        # loop all detected bboxes
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])

            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=[256,256], 
                                ratio=1.25)          
                  
            img_patch, trans, inv_trans = generate_patch_image(cvimg=original_img, 
                                                bbox=bbox, 
                                                scale=1.0, 
                                                rot=0.0, 
                                                do_flip=False, 
                                                out_shape=[256,256])

            img_patch = transform(img_patch.astype(np.float32))/255
            img_patch = img_patch.unsqueeze(0).cuda() 

            outputs = ehm_model(img_patch)
            pd_smplx_dict =  ehm(outputs['body_param'], outputs['flame_param'],  pose_type='aa') # outputs['flame_param']
 

            pd_camera = GS_Camera(**build_cameras_kwargs(1,24), R = outputs['pd_cam'][0:0+1,:3,:3], T = outputs['pd_cam'][0:0+1,:3,3])

            pd_mesh_img = body_renderer.render_mesh(pd_smplx_dict['vertices'][None, 0,...], pd_camera, lights=lights ) #  smplx2flame_ind = ehm.smplx.smplx2flame_ind 
            pd_mesh_img = (pd_mesh_img[:,:3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)

            pd_mesh_img = cv2.cvtColor(pd_mesh_img.copy(), cv2.COLOR_RGB2BGR)

            pd_mesh_img = cv2.resize(pd_mesh_img, ( 256, 256 ), interpolation=cv2.INTER_AREA)

            H, W = original_img.shape[:2]

            mesh_on_orig = cv2.warpAffine(
                pd_mesh_img,
                inv_trans,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            mask = np.any(mesh_on_orig > 0, axis=-1)

            vis_img[mask] = mesh_on_orig[mask]

        if num_bbox == 0:
            continue
        vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"mesh_{img_name}.jpg"), vis_img )

    images_to_video(
        output_path,
        os.path.join(output_path , "video.mp4" ),
        fps=30
    )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-c', default= "infer" ,  type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    parser.add_argument('--input_path',  default='example/images', type=str)
    parser.add_argument('--output_path',  default='example/images_output', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))
    # launch
    torch.set_float32_matmul_precision('high')
    inference(args.config_name, args.devices, args.input_path, args.output_path)




