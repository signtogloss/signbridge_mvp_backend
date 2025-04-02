import os
import sys
import re
import time
import subprocess
import pandas as pd
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
import cv2
import torch
from torch.nn import functional as F
import warnings

# 从我们新建的 asl_translator 模块引入统一接口
from asl_translator import asl_translator

warnings.filterwarnings("ignore")

# ----------------- 全局计时数据 -----------------
timing_data = {
    "load_rife_model": 0.0,
    "inference_rife": 0.0,
    "generate_asl_gloss": 0.0,
    "get_video_paths": 0.0,
    "generate_transition_video": 0.0,
    "concatenate_videos": 0.0,
    "main": 0.0
}

def print_timing_data():
    """在 main() 结束后打印所有函数的累计耗时及总和。"""
    print("\n===== Timing Data =====")
    total = 0.0
    for func_name, elapsed in timing_data.items():
        print(f"{func_name} took {elapsed:.4f} s")
        total += elapsed
    print(f"Overall total (sum of all functions): {total:.4f} s")
    print("===== End of Timing =====\n")

def measure_time(func_name):
    """
    一个装饰器，用来测量函数的执行时间并累加到 timing_data[func_name] 中。
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            timing_data[func_name] += elapsed
            return result
        return wrapper
    return decorator

BASE_DIR = os.getcwd()
RIFE_DIR = os.path.join(BASE_DIR, "ECCV2022-RIFE")
sys.path.append(RIFE_DIR)

DEFAULT_MODEL_DIR = "ECCV2022-RIFE/train_log"
OUTPUT_DIR = os.path.join(RIFE_DIR, "output")

DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
CSV_FILE_PATH = os.path.join(DATA_DIR, "data.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

model = None  # 全局 RIFE 模型

@measure_time("load_rife_model")
def load_rife_model(model_dir=DEFAULT_MODEL_DIR):
    """
    只加载一次 RIFE 模型到 GPU 并保持。
    做一个 dummy forward 来确保 GPU memory 不被释放。
    """
    global model
    if model is not None:
        return model  # 已加载过，直接返回

    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(model_dir, -1)
        print("Loaded ArXiv-RIFE model")

    # 设置到 GPU
    model.eval()
    model.device()  # 大部分 RIFE 代码会把 self.xxx.to(device)
    
    # 做一个 dummy forward，保证 cudnn init
    with torch.no_grad():
        dummy_in = torch.zeros((1, 3, 32, 32), device=device)
        _ = model.inference(dummy_in, dummy_in)
    print("Dummy forward done, model pinned to GPU memory.")

    return model

@measure_time("inference_rife")
def inference_rife(img0_path, img1_path, exp=4):
    rife_model = load_rife_model()  # 保证模型在 GPU
    # ...以下逻辑不变
    import cv2
    if img0_path.endswith('.exr') and img1_path.endswith('.exr'):
        img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img0 = torch.tensor(img0.transpose(2, 0, 1), device=device).unsqueeze(0)
        img1 = torch.tensor(img1.transpose(2, 0, 1), device=device).unsqueeze(0)
    else:
        img0 = cv2.imread(img0_path, cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
        img0 = (torch.tensor(img0.transpose(2, 0, 1), device=device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1), device=device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    img_list = [img0, img1]
    for _ in range(exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = rife_model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img_list[-1])
        img_list = tmp

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("img") and f.endswith(".png"):
            os.remove(os.path.join(OUTPUT_DIR, f))

    for i, frame in enumerate(img_list):
        if img0_path.endswith('.exr') and img1_path.endswith('.exr'):
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"img{i}.exr"),
                frame[0].detach().cpu().numpy().transpose(1, 2, 0)[:h, :w],
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
            )
        else:
            rgb = (frame[0] * 255).byte().detach().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"img{i}.png"), rgb)

    return len(img_list)

@measure_time("generate_asl_gloss")
def generate_asl_gloss(sentence, model="deepseek"):
    """
    调用我们在 asl_translator 中暴露的统一接口。
    参数:
        sentence (str): 要翻译的英文句子
        model (str): 选择的翻译模型，如 'gpt' 或 'monica' (deepseek)，默认为 'gpt'
    """
    return asl_translator(sentence, model=model)

@measure_time("get_video_paths")
def get_video_paths(gloss_list, csv_file_path, role='CBC_2'):
    cleaned_gloss_list = [re.sub(r'[^a-zA-Z]', '', gloss) for gloss in gloss_list]
    df = pd.read_csv(csv_file_path)
    filtered_df = df[df['Role'] == role]
    video_paths = []
    for gloss in cleaned_gloss_list:
        if gloss:
            match = filtered_df[(filtered_df['Sign'] == gloss)]
            if not match.empty:
                video_path = os.path.join(VIDEOS_DIR, match.iloc[0]['Video_Path'])
                video_paths.append(video_path)
            else:
                print(f"警告：未找到 gloss '{gloss}' 对应的角色 '{role}' 的视频。")
    return video_paths

@measure_time("generate_transition_video")
def generate_transition_video(img0_path, img1_path, index):
    print("准备转场视频")
    frame_count = inference_rife(img0_path, img1_path, exp=4)
    output_video_path = os.path.join(OUTPUT_DIR, f"slomo_{index}.mp4")
    command = [
        "ffmpeg", "-y", "-r", "10", "-f", "image2",
        "-i", "output/img%d.png",
        "-s", "576x768",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_video_path,
        "-q:v", "0", "-q:a", "0"
    ]
    subprocess.run(f"cd {RIFE_DIR} && " + " ".join(command), shell=True, check=True)
    print("转场视频生成完毕")
    if not os.path.exists(output_video_path):
        raise FileNotFoundError(f"未能找到生成的过渡视频：{output_video_path}")

    return output_video_path

@measure_time("concatenate_videos")
def concatenate_videos(video_paths, output_path):
    video_clips = []
    for i, path in enumerate(video_paths):
        clip = VideoFileClip(path)
        if i == 0:
            clip = clip.subclip(0, clip.duration - 0.2)
        else:
            clip = clip.subclip(0.4, clip.duration - 0.3)
        video_clips.append(clip)

    transition_clips = {}
    transit_folder = os.path.join(BASE_DIR, "transit_img")
    for i in range(len(video_clips) - 1):
        curr_clip = video_clips[i]
        next_clip = video_clips[i + 1]

        # img0_path = os.path.join(BASE_DIR, f"temp_img0_{i}.png")
        # img1_path = os.path.join(BASE_DIR, f"temp_img1_{i}.png")
        img0_path = os.path.join(transit_folder, f"temp_img0_{i}.png")
        img1_path = os.path.join(transit_folder, f"temp_img1_{i}.png")



        curr_clip.save_frame(img0_path, t=curr_clip.duration - 0.1)
        next_clip.save_frame(img1_path, t=0.1)

        start_time_ = time.time()
        transition_video_path = generate_transition_video(img0_path, img1_path, i)
        elapsed_ = time.time() - start_time_

        transition_clip = VideoFileClip(transition_video_path).fx(vfx.speedx, 6)
        transition_clips[i] = (transition_clip, elapsed_)

    print(f"总共生成 {len(transition_clips)} 个转场视频。")
    for idx, (clip_obj, t) in sorted(transition_clips.items()):
        print(f"转场视频 {idx} 生成耗时: {t:.2f} 秒")

    final_clips = []
    for i, clip in enumerate(video_clips):
        final_clips.append(clip)
        if i in transition_clips:
            final_clips.append(transition_clips[i][0])

    if final_clips:
        final_clip = concatenate_videoclips(final_clips, method="compose")
        final_clip = final_clip.fx(vfx.speedx, 1.5)
        original_width, original_height = final_clip.size
        crop_bottom_height = int(original_height * 0.1)
        final_clip = final_clip.crop(x1=0, y1=0, x2=original_width, y2=original_height - crop_bottom_height)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    else:
        print("错误：没有可拼接的视频片段。")

@measure_time("main")
def main(sentence, output_video_path='output.mp4', role='CBC_2', model='deepseek'):
    """
    主流程：
    1. 调用 generate_asl_gloss(sentence, model) 生成手语 gloss
    2. 根据 gloss 在 CSV 中查找对应视频路径
    3. 拼接视频并生成最终输出
    """
    asl_gloss = generate_asl_gloss(sentence, model=model)
    gloss_list = asl_gloss.split()
    print(f"Generated ASL gloss: {asl_gloss}")

    video_paths = get_video_paths(gloss_list, CSV_FILE_PATH, role=role)

    # print(f"Video Paths: {video_paths}")
    concatenate_videos(video_paths, output_path=output_video_path)
    
    print(f"concatenate_videos: {output_video_path}")
    print_timing_data()

if __name__ == "__main__":
    input_sentence = "Hello, how are you?"
    output_path = os.path.join(BASE_DIR, "output1.mp4")
    
    # 你可以通过 model 参数来选择 "gpt" 或 "monica"（deepseek）
    # 默认是 "gpt"
    main(input_sentence, output_video_path=output_path, role='CBC_2', model='deepseek')