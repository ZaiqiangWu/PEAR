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
import torch
import argparse
import lightning
import numpy as np
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import (
    ConfigDict, rtqdm, device_parser, add_extra_cfgs
)
import glob
from time import time
import huggingface_hub
from huggingface_hub import hf_hub_download

def pad_and_resize(img, target_size=512):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
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


def inference( config_name, input_path=None, output_path=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    meta_cfg = ConfigDict(
        model_config_path=os.path.join('configs', f'{config_name}.yaml')
    )
    meta_cfg = add_extra_cfgs(meta_cfg)
    print(str(meta_cfg))
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

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_paths = [os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.lower().endswith(image_extensions)]
   
    image_paths = sorted(image_paths)

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        resized = pad_and_resize(img, target_size=256)
        img_patch = to_tensor(resized,'cuda:0')  # (B, C, H, W)
        img_patch =  torch.permute(img_patch/255,(2,0,1)).unsqueeze(0)


        outputs = ehm_model(img_patch)
        pd_smplx_dict =  ehm(outputs['body_param'], outputs['flame_param'],  pose_type='aa') # outputs['flame_param']

        batch_images = pad_and_resize(img, target_size=1024) # 
        _img = batch_images

        pd_camera = GS_Camera(**build_cameras_kwargs(1,24), R = outputs['pd_cam'][0:0+1,:3,:3], T = outputs['pd_cam'][0:0+1,:3,3])

        pd_mesh_img = body_renderer.render_mesh(pd_smplx_dict['vertices'][None, 0,...], pd_camera, lights=lights)
        pd_mesh_img = (pd_mesh_img[:,:3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)

        pd_mesh_img = cv2.cvtColor(pd_mesh_img.copy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path,f"mesh_{img_name}.jpg"), pd_mesh_img )

        foreground_mask = np.any(pd_mesh_img != [0,0,0], axis=2)
        background_mask = ~foreground_mask

        bright_bg = _img.astype(np.float32) * 1.5
        bright_bg = np.clip(bright_bg, 0, 255).astype(np.uint8)
        result = bright_bg.copy()

        # you can change the alpha to control the transparency of the mesh
        alpha = 0.5  

        result = cv2.addWeighted(pd_mesh_img, alpha, bright_bg, 1-alpha, 0)

        cv2.imwrite(os.path.join(output_path,f"result_{img_name}.jpg"), result )




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-c', default= "infer" ,  type=str)
    parser.add_argument('--input_path',  default='data_input/test_source_images', type=str)
    parser.add_argument('--output_path',  default='data_input/test_source_images', type=str)

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))
    # launch
    torch.set_float32_matmul_precision('high')
    inference(args.config_name, args.input_path, args.output_path)




