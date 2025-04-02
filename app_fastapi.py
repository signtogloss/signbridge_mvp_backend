from fastapi import FastAPI, WebSocket
# from faster_whisper import WhisperModel
import whisper
from TTS.api import TTS
import io
import torchaudio
import uvicorn
import uuid
from pyngrok import ngrok, conf
from starlette.websockets import WebSocketDisconnect
import soundfile as sf  # 必须安装 soundfile
import asyncio

# 从我们新建的 asl_translator 模块引入统一接口
from asl_translator import asl_translator

# 自定义的视频处理函数
from pipeline import GlossToVideoPipeline

# 引入 WhisperLiveKit 中封装的音频处理器，用于实时语音识别
from whisperlivekit.audio_processor import AudioProcessor

app = FastAPI()

# =============================
# 初始化 Faster-Whisper 模型
# =============================
# 加载 whisper 模型，使用 float16 提高推理速度，部署在 GPU 上
# 可选模型：tiny / base / small / medium / large
# whisper_model = WhisperModel("medium", device="cuda", compute_type="float16", local_files_only=False)

# 加载 Whisper 模型（如有 GPU，可指定 device="cuda"）
whisper_model = whisper.load_model("medium", device="cuda")


# =============================
# 初始化 TTS 模型
# =============================
# 加载 Coqui 的 VITS 英语语音模型（VCTK），部署在 GPU
tts_model = TTS("tts_models/en/vctk/vits").to("cuda")


# =============================
# 初始化 视频处理管线
# =============================
# 初始化视频处理管线，部署在常驻内存
pipeline = GlossToVideoPipeline(
    csv_path='data/data.csv',
    video_dir='data/videos',
    rife_model_dir='model/RIFE_HDv2',
    transit_img_dir='temp/',
    output_dir='static/output/'
)

# =========================================
# WebSocket 接口：语音转文字
# 接口路径：/ws/speech-to-text
# =========================================

# 处理识别结果并通过 WebSocket 发送给前端
async def handle_websocket_results(websocket: WebSocket, results_generator):
    try:
        # 异步迭代识别结果生成器
        async for response in results_generator:
            # 发送结果给前端（JSON 格式）
            await websocket.send_json(response)
    except Exception as e:
        print(f"Error in WebSocket results handler: {e}")
        
# 处理识别结果并通过 WebSocket 发送给前端 并且放进去队列
async def handle_websocket_results_queue(websocket: WebSocket, results_generator, result_queue: asyncio.Queue):
    try:
        async for response in results_generator:
            await websocket.send_json(response)
            await result_queue.put(response)  # 放入队列供主循环打印
    except Exception as e:
        print(f"Error in WebSocket results handler: {e}")

        
@app.websocket("/ws/speech2text")
async def speech_to_text(websocket: WebSocket):
    """
    接收来自客户端的音频流（WAV PCM 格式）
    使用 Faster-Whisper 实时进行语音识别
    每识别一次，将结果通过 WebSocket 返回前端

    输入：WebSocket 中发送的二进制音频片段（bytes）
    输出：WebSocket 发送回识别后的文本（str）
    """
    # 接受 WebSocket 连接
    await websocket.accept()
    print("WebSocket connection opened.")

    # 创建一个音频处理器实例（每个连接一个）
    audio_processor = AudioProcessor()

    # 创建识别任务，并获得结果生成器
    results_generator = await audio_processor.create_tasks()

    result_queue = asyncio.Queue()

    # 创建一个后台任务用于发送识别结果给客户端
    websocket_task = asyncio.create_task(
        handle_websocket_results_queue(websocket, results_generator, result_queue)
    )

    try:
        # 用于保存上一次识别的文本
        last_text = ""
        # 持续接收客户端发来的音频数据（原始字节）
        while True:
            data = await websocket.receive_bytes()
            # 将音频数据送入处理器，进行实时识别
            await audio_processor.process_audio(data)

            # 非阻塞地查看是否有识别结果
            while not result_queue.empty():
                result = await result_queue.get()
                
                text_result = " ".join(
                    line["text"].strip() for line in result.get("lines", []) if line.get("text")
                )
                
                # 只输出新增的部分
                new_text = text_result[len(last_text):] if text_result.startswith(last_text) else text_result
                if new_text:
                    print(new_text)
                last_text = text_result

    except WebSocketDisconnect:
        # 客户端断开连接时触发
        print("WebSocket disconnected.")
    finally:
        # 取消后台发送任务，释放资源
        websocket_task.cancel()
        await audio_processor.cleanup()
        print("Cleaned up audio processor.")

# =========================================
# WebSocket 接口：文字转语音
# 接口路径：/ws/text-to-speech
# =========================================
@app.websocket("/ws/text2speech")
async def text_to_speech(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            if not message.strip():
                continue

            # 生成语音（假设模型输出 numpy 数组和采样率）
            audio_array, sample_rate = tts_model.tts(message)

            # 写入到内存中
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format="WAV")
            buffer.seek(0)

            # 发送给前端
            await websocket.send_bytes(buffer.read())

            # 清理（也可以交给 GC）
            buffer.close()

    except Exception as e:
        print(f"[TTS WS Error] {e}")
    finally:
        await websocket.close()


# =========================================
# WebSocket 接口：文字转gloss（使用gpt大模型）
# 接口路径：/ws/text2gloss
# =========================================
@app.websocket("/ws/text2gloss")
async def text_to_gloss(websocket: WebSocket):
    """
    实时接收用户文本，通过大模型翻译为 ASL gloss，并返回。
    输入格式: 纯文本字符串
    输出格式: ASL gloss 字符串（大写）
    """
    await websocket.accept()
    try:
        while True:
            sentence = await websocket.receive_text()
            if not sentence.strip():
                continue

            # 调用统一翻译接口，默认用 Deepseek（你可以换成 gpt / monica）
            gloss = asl_translator(sentence, model="deepseek")
            await websocket.send_text(gloss)
    except Exception as e:
        print(f"[Text→Gloss WS Error] {e}")
    finally:
        await websocket.close()


# =========================================
# WebSocket 接口：输入gloss序列生成手语的视频并且弄成二进制返回
# 接口路径：/ws/gloss2video
# =========================================
@app.websocket("/ws/gloss2video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 接收 gloss 列表和可选参数
            data = await websocket.receive_json()
            gloss_list = data.get("gloss", [])
            if not gloss_list:
                await websocket.send_json({"status": "error", "message": "Empty gloss list"})
                continue

            video_id = str(uuid.uuid4())[:8]
            output_path = f"static/output/{video_id}.mp4"

            # 发送状态通知
            await websocket.send_json({
                "status": "processing",
                "video_id": video_id
            })

            # 生成视频
            final_path = pipeline.generate_video_from_gloss(gloss_list, output_path)

            # 读取视频文件为二进制
            with open(final_path, "rb") as f:
                video_data = f.read()

            # 发送成功状态 + 视频大小（可选）
            await websocket.send_json({
                "status": "success",
                "video_id": video_id,
                "video_size": len(video_data)
            })

            # 发送视频二进制内容
            await websocket.send_bytes(video_data)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"[Generate WS Error] {e}")
        await websocket.send_json({"status": "error", "message": str(e)})
    finally:
        await websocket.close()



# =========================================
# WebSocket 接口： 半个一条龙服务 输入text的手语的视频并且弄成二进制返回
# 接口路径：/ws/text2video
# =========================================
@app.websocket("/ws/text2video")
async def text_to_generate(websocket: WebSocket):
    """
    接收实时文字，实时转换为 ASL gloss，然后立即生成手语视频并通过 WebSocket 返回客户端。
    实现极快响应与低延迟。

    输入：客户端实时发送的文本字符串
    输出：实时生成的手语视频二进制数据（mp4格式）
    """
    await websocket.accept()
    try:
        while True:
            # 实时接收客户端发送的文本
            sentence = await websocket.receive_text()

            # 立即将文本转化为 ASL gloss
            gloss = asl_translator(sentence, model="deepseek")

            # gloss 序列化处理
            gloss_list = gloss.strip().split()

            if not gloss_list:
                await websocket.send_json({"status": "error", "message": "Empty gloss result"})
                continue

            video_id = str(uuid.uuid4())[:8]
            output_path = f"static/output/{video_id}.mp4"

            # 通知客户端当前处理状态
            await websocket.send_json({
                "status": "processing",
                "video_id": video_id,
                "gloss": gloss
            })

            # 调用视频生成管线，生成手语视频
            final_path = pipeline.generate_video_from_gloss(gloss_list, output_path)

            # 读取视频文件为二进制数据
            with open(final_path, "rb") as f:
                video_data = f.read()

            # 发送生成成功状态及视频元信息
            await websocket.send_json({
                "status": "success",
                "video_id": video_id,
                "video_size": len(video_data),
                "gloss": gloss
            })

            # 实时发送视频二进制数据给客户端
            await websocket.send_bytes(video_data)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"[Text2Generate WS Error] {e}")
        await websocket.send_json({"status": "error", "message": str(e)})
    finally:
        await websocket.close()



# =========================================
# WebSocket 接口：一条龙服务 输入speech的语音生成手语的视频并且弄成二进制返回
# 接口路径：/ws/speech2video
# =========================================
@app.websocket("/ws/speech2video")
async def speech_to_video(websocket: WebSocket):
    """
    WebSocket接口：实时接收语音数据，转化为ASL gloss文本，再实时生成手语视频，并以二进制数据流返回客户端。
    实现快速响应、低延迟的实时互动体验。
    
    流程：
    - 客户端发送实时语音数据字节流
    - 后端识别音频，转换成文本
    - 文本实时翻译成ASL gloss
    - gloss实时生成手语视频（MP4）
    - 将视频以二进制流实时返回给客户端

    输入: 客户端实时发送的音频数据（二进制）
    输出: 实时生成的手语视频数据（MP4格式二进制）
    """
    await websocket.accept()
    print("WebSocket connection opened.")

    # 创建一个音频处理器实例（每个连接一个）
    audio_processor_long = AudioProcessor()

    # 创建识别任务，并获得结果生成器
    results_generator_long = await audio_processor_long.create_tasks()

    # 创建一个异步队列用于存放识别出的句子
    sentence_queue = asyncio.Queue()

    # 创建一个后台任务用于发送识别结果给客户端
    websocket_task_long = asyncio.create_task(
        handle_websocket_results_queue(websocket, results_generator_long, sentence_queue)
    )
    
    try:
        # 用于保存上一次识别的文本
        last_text = ""
        while True:
            # 接收客户端发送的音频数据字节流
            audio_bytes = await websocket.receive_bytes()

            # 将音频数据送入处理器进行识别
            await audio_processor_long.process_audio(audio_bytes)

            # 非阻塞地查看是否有识别结果
            while not sentence_queue.empty():
                
                result = await sentence_queue.get()
                
                print(result)
            
                sentence = " ".join(
                    line["text"].strip() for line in result.get("lines", []) if line.get("text")
                 )

                # 只输出新增的部分
                new_text = text_result[len(last_text):] if text_result.startswith(last_text) else text_result
                if new_text:
                    print(new_text)
                last_text = text_result

                if sentence:
                    print("sentence is :", sentence)

                    # 文本翻译为ASL gloss
                    gloss = asl_translator(sentence, model="deepseek")
                    gloss_list = gloss.strip().split()

                    print("gloss_list",gloss_list)

                    if not gloss_list:
                        await websocket.send_json({"status": "error", "message": "Empty gloss result"})
                        continue

                    video_id = str(uuid.uuid4())[:8]
                    output_path = f"static/output/{video_id}.mp4"

                    # 通知客户端开始生成视频
                    await websocket.send_json({
                        "status": "processing",
                        "video_id": video_id,
                        "gloss": gloss
                    })

                    # 使用管道生成手语视频
                    final_path = pipeline.generate_video_from_gloss(gloss_list, output_path)

                    # 读取生成的视频文件
                    try:
                        with open(final_path, "rb") as video_file:
                            video_data = video_file.read()
                    except FileNotFoundError:
                        await websocket.send_json({
                            "status": "error",
                            "message": "video file generate error"
                        })
                        continue

                    # 发送视频生成成功信息
                    await websocket.send_json({
                        "status": "success",
                        "video_id": video_id,
                        "video_size": len(video_data),
                        "gloss": gloss
                    })

                    # 实时发送视频二进制数据给客户端
                    await websocket.send_bytes(video_data)

    except WebSocketDisconnect:
        print("WebSocket连接已断开。")
    except Exception as e:
        print(f"WebSocket处理错误: {e}")
        await websocket.send_json({"status": "error", "message": str(e)})
    finally:
        websocket_task_long.cancel()
        await websocket.close()



if __name__ == "__main__":

    # 设置 ngrok authtoken
    # conf.get_default().auth_token = "2islHXOo5aZc5bDDrxFE0kXKPdx_7Ewqnt85UDFTNTNFooNNa"

    # 启动 ngrok 隧道，映射本地 8000 端口
    port = 8000
    # public_url = ngrok.connect(port, bind_tls=True)

    print(f"\nFastAPI 正在运行: http://127.0.0.1:{port}")
    # print(f"公网访问地址 (ngrok): {public_url}\n")

    # 启动 uvicorn 服务（带 reload）
    # uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, reload=True)


