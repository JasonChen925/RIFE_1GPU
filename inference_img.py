import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
import time
import torchsummary
from torchstat import stat
from gpu_mem_track import MemTracker
from ptflops import get_model_complexity_info
from memory_profiler import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
# gpu_tracker = MemTracker()
# gpu_tracker.track()
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', dest='img', nargs=2, required=True)
parser.add_argument('--exp', default=1, type=int)
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')#当实际比率下降到给定范围阈值时返回图像
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')#限制对分循环的最大数目
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

args = parser.parse_args()
# time_stamp1=time.time()
# try:
#     try:
#         try:
#             from model.RIFE_HDv2 import Model
#             model = Model()
#             model.load_model(args.modelDir, -1)
#             print("Loaded v2.x HD model.")
#         except:
#             from train_log.RIFE_HDv3 import Model
#             model = Model()
#             model.load_model(args.modelDir, -1)
#             print("Loaded v3.x HD model.")
#     except:
#         from model.RIFE_HD import Model
#         model = Model()
#         model.load_model(args.modelDir, -1)
#         print("Loaded v1.x HD model")
# except:
from model.RIFE import Model
model = Model()
model.load_model(args.modelDir, -1)
print("Loaded ArXiv-RIFE model")


# time_stamp2=time.time()
# print("time_loadingmodel={}".format(time_stamp2-time_stamp1))

model.eval()
model.device()

if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
    img0 = cv2.imread(args.img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    img1 = cv2.imread(args.img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)

else:
    img0 = cv2.imread(args.img[0], cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(args.img[1], cv2.IMREAD_UNCHANGED)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

# time_stamp3=time.time()
# print("time_cv={}".format(time_stamp3-time_stamp2))

n, c, h, w = img0.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
img0 = F.pad(img0, padding)
img1 = F.pad(img1, padding)
# time_stamp4=time.time()
# print("time_pad={}".format(time_stamp4-time_stamp3))

if args.ratio:
    img_list = [img0]
    img0_ratio = 0.0
    img1_ratio = 1.0
    if args.ratio <= img0_ratio + args.rthreshold / 2:
        middle = img0
    elif args.ratio >= img1_ratio - args.rthreshold / 2:
        middle = img1
    else:
        tmp_img0 = img0
        tmp_img1 = img1
        for inference_cycle in range(args.rmaxcycles):
            middle = model.inference(tmp_img0, tmp_img1)
            middle_ratio = ( img0_ratio + img1_ratio ) / 2
            if args.ratio - (args.rthreshold / 2) <= middle_ratio <= args.ratio + (args.rthreshold / 2):
                break
            if args.ratio > middle_ratio:
                tmp_img0 = middle
                img0_ratio = middle_ratio
            else:
                tmp_img1 = middle
                img1_ratio = middle_ratio
    img_list.append(middle)
    img_list.append(img1)
else:
    img_list = [img0, img1]
    # time_stamp5=time.time()
    for i in range(args.exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp

# time_stamp6=time.time()
# print("interpolate_time={}".format(time_stamp6-time_stamp5))
if not os.path.exists('output'):
    os.mkdir('output')
for i in range(len(img_list)):
    if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
        cv2.imwrite('output/img{}.exr'.format(i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    else:
        cv2.imwrite('output/img{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
# gpu_tracker.track()
# tracemalloc.stop()
# torchsummary.summary(model=model.flownet,input_size=(6,256,448),batch_size=8,device='cuda')
#对显存进行研究
flops,params = get_model_complexity_info(model=model.flownet,input_res=(6,960,540),as_strings=True,print_per_layer_stat=True)
print(flops)
print(params)