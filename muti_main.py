import os
import multiprocessing
from whisper_tool import do_whisper


def process_audio(audio_file, output_dir, language, hf_model_path, gpu_id):
    """
    处理单个音频文件，绑定到指定的 GPU。
    :param audio_file: 音频文件路径
    :param output_dir: 输出 SRT 文件夹路径
    :param language: 语言代码（如 "en", "zh"）
    :param hf_model_path: 自定义模型路径（如果为空，则使用默认模型）
    :param gpu_id: 绑定的 GPU ID
    """
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 构造输出 SRT 文件路径
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    srt_path = os.path.join(output_dir, f"{base_name}.srt")

    # 调用 whisper_tool 的 do_whisper 方法
    print(f"Processing {audio_file} on GPU {gpu_id}")
    try:
        do_whisper(audio_file, srt_path, language, hf_model_path, 'cuda')
    except Exception as e:
        print(f"Error processing file {audio_file}: {e}")


def main(input_dir="audio", output_dir="output", language="en", hf_model_path="", num_gpus=8):
    """
    主函数：遍历音频文件夹并动态分配任务到多个 GPU。
    :param input_dir: 输入音频文件夹路径
    :param output_dir: 输出 SRT 文件夹路径
    :param language: 语言代码（如 "en", "zh"）
    :param hf_model_path: 自定义模型路径（如果为空，则使用默认模型）
    :param num_gpus: 可用 GPU 数量
    """
    # 支持的音频格式
    audio_extensions = [".mp3", ".wav", ".flac", ".aac"]

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))

    # 动态任务分配
    lock = multiprocessing.Lock()  # 用于线程安全的任务分配
    current_task_index = multiprocessing.Value('i', 0)  # 当前任务索引

    def worker(gpu_id):
        """每个 GPU 的工作线程"""
        while True:
            # 获取下一个任务
            with lock:
                task_index = current_task_index.value
                if task_index >= len(audio_files):
                    break  # 没有更多任务，退出循环
                current_task_index.value += 1

            audio_file = audio_files[task_index]
            process_audio(audio_file, output_dir, language, hf_model_path, gpu_id)

    # 启动多个进程，每个进程绑定到一个 GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(target=worker, args=(gpu_id,))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()


if __name__ == '__main__':
    # 输入音频文件夹路径
    input_dir = "audio"

    # 输出 SRT 文件夹路径
    output_dir = "output"

    # 语言代码（如 "en" 表示英语，"zh" 表示中文）
    language = "en"

    # 自定义模型路径（如果为空，则使用默认模型）
    hf_model_path = ""

    # 可用 GPU 数量
    num_gpus = 8

    # 开始处理音频文件夹
    main(input_dir=input_dir, output_dir=output_dir, language=language, hf_model_path=hf_model_path, num_gpus=num_gpus)