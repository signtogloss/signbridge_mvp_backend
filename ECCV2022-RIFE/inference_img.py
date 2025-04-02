# import os
# import cv2
# import torch
# import argparse
# from torch.nn import functional as F
# import warnings
# import time



# warnings.filterwarnings("ignore")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_grad_enabled(False)
# if torch.cuda.is_available():
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True

# parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# parser.add_argument('--img', dest='img', nargs=2, required=True)
# parser.add_argument('--exp', default=4, type=int)
# parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
# parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
# parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
# parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

# args = parser.parse_args()

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
#     from model.RIFE import Model
#     model = Model()
#     model.load_model(args.modelDir, -1)
#     print("Loaded ArXiv-RIFE model")
# model.eval()
# model.device()

# if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
#     img0 = cv2.imread(args.img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
#     img1 = cv2.imread(args.img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
#     img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
#     img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)

# else:
#     img0 = cv2.imread(args.img[0], cv2.IMREAD_UNCHANGED)
#     img1 = cv2.imread(args.img[1], cv2.IMREAD_UNCHANGED)
#     img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
#     img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

# n, c, h, w = img0.shape
# ph = ((h - 1) // 32 + 1) * 32
# pw = ((w - 1) // 32 + 1) * 32
# padding = (0, pw - w, 0, ph - h)
# img0 = F.pad(img0, padding)
# img1 = F.pad(img1, padding)


# if args.ratio:
#     img_list = [img0]
#     img0_ratio = 0.0
#     img1_ratio = 1.0
#     if args.ratio <= img0_ratio + args.rthreshold / 2:
#         middle = img0
#     elif args.ratio >= img1_ratio - args.rthreshold / 2:
#         middle = img1
#     else:
#         tmp_img0 = img0
#         tmp_img1 = img1
#         for inference_cycle in range(args.rmaxcycles):
#             middle = model.inference(tmp_img0, tmp_img1)
#             middle_ratio = ( img0_ratio + img1_ratio ) / 2
#             if args.ratio - (args.rthreshold / 2) <= middle_ratio <= args.ratio + (args.rthreshold / 2):
#                 break
#             if args.ratio > middle_ratio:
#                 tmp_img0 = middle
#                 img0_ratio = middle_ratio
#             else:
#                 tmp_img1 = middle
#                 img1_ratio = middle_ratio
#     img_list.append(middle)
#     img_list.append(img1)
# # else:
# #     img_list = [img0, img1]
# #     for i in range(args.exp):
# #         tmp = []
# #         for j in range(len(img_list) - 1):
# #             mid = model.inference(img_list[j], img_list[j + 1])
# #             tmp.append(img_list[j])
# #             tmp.append(mid)
# #         tmp.append(img1)
# #         img_list = tmp




# else:
#     img_list = [img0, img1]
#     for i in range(args.exp):
#         iteration_start = time.time()

#         tmp = []
#         inference_time = 0.0
#         # 当前轮要插值的对数
#         pairs = len(img_list) - 1

#         # 逐对推理
#         for j in range(pairs):
#             t_infer_start = time.time()
#             mid = model.inference(img_list[j], img_list[j + 1])
#             inference_time += (time.time() - t_infer_start)

#             tmp.append(img_list[j])
#             tmp.append(mid)

#         tmp.append(img_list[-1])
#         img_list = tmp

#         iteration_end = time.time()
#         total_iteration_time = iteration_end - iteration_start

#         # 打印每轮的信息
#         print(f"Exp iteration {i+1}/{args.exp}: "
#               f"pairs={pairs}, "
#               f"inference_time={inference_time:.4f}s, "
#               f"total_iteration_time={total_iteration_time:.4f}s")



# if not os.path.exists('output'):
#     os.mkdir('output')
# for i in range(len(img_list)):
#     if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
#         cv2.imwrite('output/img{}.exr'.format(i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
#     else:
#         cv2.imwrite('output/img{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
import time

warnings.filterwarnings("ignore")

# 脚本开始时间
script_start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', dest='img', nargs=2, required=True)
parser.add_argument('--exp', default=4, type=int)
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

args = parser.parse_args()

# ------------------- 模型加载计时开始 -------------------
model_load_start = time.time()
try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded ArXiv-RIFE model")
model.eval()
model.device()
model_load_end = time.time()
model_load_time = model_load_end - model_load_start
print(f"[Timing] Model loading took {model_load_time:.4f} s")

# ------------------- 图像读取计时开始 -------------------
image_read_start = time.time()

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

n, c, h, w = img0.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
img0 = F.pad(img0, padding)
img1 = F.pad(img1, padding)

image_read_end = time.time()
image_read_time = image_read_end - image_read_start
print(f"[Timing] Image read + preprocess took {image_read_time:.4f} s")

# ------------------- 插值计时开始 -------------------
interpolation_start = time.time()

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
    for i in range(args.exp):
        iteration_start = time.time()

        tmp = []
        inference_time = 0.0
        pairs = len(img_list) - 1

        for j in range(pairs):
            t_infer_start = time.time()
            mid = model.inference(img_list[j], img_list[j + 1])
            inference_time += (time.time() - t_infer_start)

            tmp.append(img_list[j])
            tmp.append(mid)

        tmp.append(img_list[-1])
        img_list = tmp

        iteration_end = time.time()
        total_iteration_time = iteration_end - iteration_start
        print(f"Exp iteration {i+1}/{args.exp}: "
              f"pairs={pairs}, "
              f"inference_time={inference_time:.4f}s, "
              f"total_iteration_time={total_iteration_time:.4f}s")

interpolation_end = time.time()
interpolation_time = interpolation_end - interpolation_start
print(f"[Timing] Interpolation took {interpolation_time:.4f} s")

# ------------------- 写结果计时开始 -------------------
save_start = time.time()
if not os.path.exists('output'):
    os.mkdir('output')

for i in range(len(img_list)):
    if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
        cv2.imwrite('output/img{}.exr'.format(i),
                    (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w],
                    [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    else:
        cv2.imwrite('output/img{}.png'.format(i),
                    (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
save_end = time.time()
save_time = save_end - save_start
print(f"[Timing] Saving output took {save_time:.4f} s")

# ------------------- 脚本整体耗时 -------------------
script_end_time = time.time()
total_script_time = script_end_time - script_start_time
print(f"[Timing] Total script runtime: {total_script_time:.4f} s")
