1、功能：
get_audio.py 批量提取录音文件
whisper_tool.py 语音文件转写成文本
muti_main.py 多gpu并行处理whisper_tool任务
muti_translate.py 多线程翻译成中文
secondary_translate.py 再次处理未正确翻译任务

2、usage:
python get_audio.py 遍历当前目录的视频文件，提取出音频文件并保存到audio目录
muti_main.py 遍历同级的audio目录音频文件，生成.srt字幕并输出到output目录
muti_translate.py 遍历同级srt目录，生成翻译文件[原名]_translated.srt
secondary_translate.py 遍历同级srt目录，生成最终翻译文件[原名]_zn.srt

3、依赖:
whisper
ffmpeg
torch
deepseek（api接口）