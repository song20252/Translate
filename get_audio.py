import os
import subprocess


def extract_audio(video_path, output_dir):
    """
    使用 ffmpeg 提取音频并保存到指定目录。
    :param video_path: 视频文件路径
    :param output_dir: 输出音频目录
    """
    # 获取视频文件名（不含扩展名）和音频输出路径
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_output_path = os.path.join(output_dir, f"{video_name}.mp3")

    # 构造 ffmpeg 命令
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,       # 输入视频文件
        "-map", "0:a",          # 选择所有音频流
        "-c:a", "libmp3lame",   # 使用 libmp3lame 编码器进行转码
        "-q:a", "2",            # 设置 MP3 质量
        audio_output_path       # 输出音频文件路径
    ]

    try:
        # 执行 ffmpeg 命令
        subprocess.run(ffmpeg_command, check=True)
        print(f"Audio extracted: {audio_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract audio from {video_path}: {e}")


def main():
    # 当前目录
    current_dir = os.getcwd()

    # 创建 audio 目录
    audio_dir = os.path.join(current_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # 支持的视频格式
    video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"]

    # 遍历当前目录及子目录，找到所有视频文件
    for root, _, files in os.walk(current_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(root, file)
                extract_audio(video_path, audio_dir)


if __name__ == "__main__":
    main()