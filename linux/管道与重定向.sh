# 重定向
xxx < input.txt #重定向输入
xxx > output.txt #重定向输出
xxx >> output.txt #重定向输出(追加模式)
xxx 1> out.txt 2> err.txt #重定向输出和错误信息
xxx &> out.txt #重定向输出和错误信息到同一个文件

xxx > /dev/null #抛弃输出

# 管道
ls | wc -w #前者的输出作为后者的输入

