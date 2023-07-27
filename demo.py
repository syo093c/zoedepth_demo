import torch
from PIL import Image
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
from zoedepth.utils.misc import pil_to_batched_tensor
from zoedepth.utils.misc import get_image_from_url
from zoedepth.utils.misc import save_raw_16bit
import open3d as o3d
import cv2
import ipdb

def visualization_rgb(img,depth_map):
    h,w,c=img.shape
    rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),
                depth=o3d.geometry.Image(depth_map),convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            #o3d.camera.PinholeCameraIntrinsic(1920,1080,1054,1054,1920/2,1080/2))
            o3d.camera.PinholeCameraIntrinsic(w,h,1054,1054,w/2,h/2))
    coord=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.0005)
    o3d.visualization.draw_geometries([pcd,coord])

def load_zoedepth_model():
# load model
# ZoeD_N
    conf = get_config("zoedepth", "infer")
    model_zoe_n = build_model(conf)
## ZoeD_K
#    conf = get_config("zoedepth", "infer", config_version="kitti")
#    model_zoe_k = build_model(conf)
## ZoeD_NK
#    conf = get_config("zoedepth_nk", "infer")
#    model_zoe_nk = build_model(conf)
#
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)
    return zoe

def demo():
    zoe=load_zoedepth_model()

    #input image
    pil_image = Image.open("./cake.jpeg").convert("RGB")  # load
    img=cv2.imread("./cake.jpeg")

    depth = zoe.infer_pil(pil_image)
    colored = colorize(depth)

    visualization_rgb(img=img,depth_map=depth)

    fpath_colored = "output.png"
    Image.fromarray(colored).save(fpath_colored)


if __name__ == '__main__':
    demo()
