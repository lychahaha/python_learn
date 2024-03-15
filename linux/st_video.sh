# yt-dlp

# 安装
pip install yt-dlp
# 使用
yt-dlp url
    -F #查看支持的视频音频规格

    --merge-output-format mp4 #指定视频格式

    -x #只下载音频
    --audio-format mp3 #指定音频格式

    -f bv*+ba #最好画质+最好音质
    -f 114+514 #指定ID的音质和画质(通过-F查表得到ID)

    -o a.mp4 #指定输出文件名
    -o %(title)s.mp4 #文件名带标题

    --write-sub #下载字幕
    --write-auto-sub #下载字幕(youtube)
    --sub-lang zh-Hans #指定语言
    --list-subs #列出支持的语言
    --convert-subs vtt #指定字幕格式

    --skip-download #不下载视频

    --write-thumbnail #下载封面
    
# ffmpeg
