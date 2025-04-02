# 标准库和第三方库的导入
import os  # 进行文件路径操作（如拼接、获取当前目录等）
import re  # 正则表达式，用于字符串匹配与提取
import subprocess  # 执行外部命令或脚本
import pandas as pd  # 用于读取和处理 CSV 表格等结构化数据
import torch  # PyTorch，深度学习框架
from torch.nn import functional as F  # PyTorch 的函数式神经网络工具，如插值、激活函数等
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx  # 处理视频片段、拼接、加特效等
import cv2  # OpenCV，用于图像和视频帧的低层处理
import sys  # 与 Python 解释器进行交互，如修改模块路径

# 项目基础目录（当前工作目录）
BASE_DIR = os.getcwd()

# RIFE 模型所在的文件夹路径，通常用于视频帧插值
RIFE_DIR = os.path.join(BASE_DIR, "ECCV2022-RIFE")

# 将 RIFE 模型路径加入 Python 模块搜索路径
# 这样就能 import 这个目录下的模块了
sys.path.append(RIFE_DIR)

# 默认的模型权重存放路径（训练日志文件夹）
DEFAULT_MODEL_DIR = "ECCV2022-RIFE/train_log"

# 视频帧插值后的输出结果目录（如中间帧、合成视频）
OUTPUT_DIR = os.path.join(RIFE_DIR, "output")

# 数据相关目录
DATA_DIR = os.path.join(BASE_DIR, "data")  # 数据总目录
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")  # 存放原始视频的目录
CSV_FILE_PATH = os.path.join(DATA_DIR, "data.csv")  # 存放视频元数据或处理信息的 CSV 文件路径

# 设备选择：如果有 CUDA 就用 GPU，否则退回到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 关闭梯度计算，提高推理阶段的性能和显存利用
torch.set_grad_enabled(False)

# 如果使用 CUDA，加速卷积性能
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 开启自动卷积算法选择（速度更快）

# 全局模型变量（后续会载入 RIFE 模型到这里）
model = None

class GlossToVideoPipeline:
    """
    实时输入gloss列表，快速拼接返回对应手语视频。
    优化策略：视频与帧缓存、GPU常驻模型、异步处理（此代码仅实现缓存和模型常驻优化）。
    """

    def __init__(self, csv_path, video_dir, rife_model_dir, transit_img_dir, output_dir):
        """
        初始化函数
        输入：
            csv_path: str, gloss与视频路径映射的CSV文件路径
            video_dir: str, 视频文件所在目录
            rife_model_dir: str, RIFE模型权重目录
            transit_img_dir: str, 临时过渡帧图片存储目录
            output_dir: str, 输出目录（保存中间帧与最终视频）
        输出：
            无（构造函数）
        """
        self.csv_path = csv_path
        self.video_dir = video_dir
        self.rife_model_dir = rife_model_dir
        self.transit_img_dir = transit_img_dir
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_rife_model()  # 加载RIFE模型到GPU

        # 缓存视频片段，加速访问
        self.video_cache = {}

    def generate_video_from_gloss(self, gloss_list, output_path, role='CBC_2'):
        """
        主调用接口，根据gloss列表生成最终的视频
        输入：
            gloss_list: list[str], 要生成视频的gloss词序列
            output_path: str, 最终输出的视频路径
            role: str, 演员角色过滤条件，默认值为 'CBC_2'
        输出：
            str, 生成视频的路径
        """
        video_paths = self.get_video_paths(gloss_list, role)
        video_clips = self.load_video_clips(video_paths)
        transition_paths = self.extract_transition_frames(video_clips)
        final_video = self.concatenate_videos(video_clips, transition_paths, output_path)
        return final_video

    def get_video_paths(self, gloss_list, role='CBC_2'):
        """
        根据gloss列表从CSV中匹配对应视频路径
        输入：
            gloss_list: list[str], gloss词序列
            role: str, 演员角色过滤条件
        输出：
            list[str], 匹配的视频文件完整路径列表
        """
        df = pd.read_csv(self.csv_path)
        filtered_df = df[df['Role'] == role]
        video_paths = []
        for gloss in gloss_list:
            gloss_clean = re.sub(r'[^a-zA-Z]', '', gloss)  # 只保留字母
            match = filtered_df[filtered_df['Sign'] == gloss_clean]
            if not match.empty:
                video_paths.append(os.path.join(self.video_dir, match.iloc[0]['Video_Path']))
        return video_paths

    def load_video_clips(self, video_paths):
        """
        加载视频片段并使用缓存提升性能
        输入：
            video_paths: list[str], 视频路径列表
        输出：
            list[VideoFileClip], 加载后的视频片段
        """
        clips = []
        for path in video_paths:
            if path not in self.video_cache:
                self.video_cache[path] = VideoFileClip(path)
            clips.append(self.video_cache[path])
        return clips

    def extract_transition_frames(self, video_clips):
        """
        从视频中提取关键帧，生成用于插帧的图片序列
        输入：
            video_clips: list[VideoFileClip], 视频片段
        输出：
            list[str], 插帧生成的过渡视频路径
        """
        transition_paths = []
        for i in range(len(video_clips) - 1):
            img0_path = os.path.join(self.transit_img_dir, f"temp_img0_{i}.png")
            img1_path = os.path.join(self.transit_img_dir, f"temp_img1_{i}.png")

            # 取前一个视频结尾帧
            video_clips[i].save_frame(img0_path, t=video_clips[i].duration - 0.1)
            # 取下一个视频起始帧
            video_clips[i + 1].save_frame(img1_path, t=0.1)

            # 生成过渡视频
            transition_video = self.generate_transition_video(img0_path, img1_path, i)
            transition_paths.append(transition_video)
        return transition_paths

    def generate_transition_video(self, img0_path, img1_path, index):
        """
        使用RIFE模型插帧生成过渡视频
        输入：
            img0_path: str, 起始帧路径
            img1_path: str, 结束帧路径
            index: int, 当前索引，用于命名输出视频
        输出：
            str, 过渡视频路径
        """
        self.inference_rife(img0_path, img1_path)
        output_video_path = os.path.join(self.output_dir, f"slomo_{index}.mp4")
        subprocess.run(
            f"ffmpeg -y -r 10 -f image2 -i {self.output_dir}/img%d.png -s 576x768 -c:v libx264 -pix_fmt yuv420p {output_video_path}",
            shell=True, check=True
        )
        return output_video_path

    def concatenate_videos(self, video_clips, transition_paths, output_path):
        """
        拼接原始视频片段与过渡视频，输出最终文件
        输入：
            video_clips: list[VideoFileClip], 原始手语视频片段
            transition_paths: list[str], 插帧生成的过渡视频路径
            output_path: str, 最终输出路径
        输出：
            str, 最终生成的视频路径
        """
        final_clips = []
        for i, clip in enumerate(video_clips):
            # 修剪视频时长，避免重复帧
            trimmed_clip = clip.subclip(0, clip.duration - (0.2 if i == 0 else 0.3))
            final_clips.append(trimmed_clip)

            # 插入过渡视频
            if i < len(transition_paths):
                transition_clip = VideoFileClip(transition_paths[i]).fx(vfx.speedx, 6)
                final_clips.append(transition_clip)

        # 加快整体视频速度
        final_video = concatenate_videoclips(final_clips, method="compose").fx(vfx.speedx, 1.5)

        # 裁剪底部区域
        w, h = final_video.size
        final_video = final_video.crop(x1=0, y1=0, x2=w, y2=h - int(h * 0.1))

        # 导出视频
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        return output_path

    def load_rife_model(self):
        """
        加载RIFE插帧模型，仅加载一次提升效率
        输入：
            无
        输出：
            model: 加载后的RIFE模型实例
        """
        global model
        if model is not None:
            return model  # 已加载过，直接返回
                
        self.rife_model_dir=DEFAULT_MODEL_DIR
        
        from train_log.RIFE_HDv3 import Model
        model = Model()
        model.load_model(self.rife_model_dir, -1)
        model.eval()
        model.device() 

        # warmup（首次推理避免延迟）
        with torch.no_grad():
            dummy_in = torch.zeros((1, 3, 32, 32), device=self.device)
            model.inference(dummy_in, dummy_in)

        return model

    def inference_rife(self, img0_path, img1_path, exp=4):
        """
        使用RIFE对两个图像进行帧插值
        输入：
            img0_path: str, 开始帧路径
            img1_path: str, 结束帧路径
            exp: int, 插值倍数（生成帧数为 2^exp）
        输出：
            无（直接将中间帧输出到 output_dir）
        """
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        img0 = (torch.tensor(img0.transpose(2, 0, 1), device=self.device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1), device=self.device) / 255.).unsqueeze(0)

        # 图像尺寸对齐到32的倍数（RIFE要求）
        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        img_list = [img0, img1]

        # 逐轮插帧
        for _ in range(exp):
            tmp = []
            for j in range(len(img_list) - 1):
                mid = self.model.inference(img_list[j], img_list[j + 1])
                tmp.extend([img_list[j], mid])
            tmp.append(img_list[-1])
            img_list = tmp

        # 写出所有帧为图片
        for i, frame in enumerate(img_list):
            rgb = (frame[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
            cv2.imwrite(os.path.join(self.output_dir, f"img{i}.png"), rgb)
