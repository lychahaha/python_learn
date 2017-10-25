#jh:func-base-mixsimilar,key-type

#单文件
pyinstaller -F a.py
#单目录(多文件)(默认)
pyinstaller -D a.py
#输出目录(默认当前目录)
pyinstaller -o D:/xxx a.py
#不要控制台
pyinstaller -w a.py
#添加搜索路径
pyinstaller -p D:/xxx a.py
#添加icon
pyinstaller -i D:/xxx/x.ico a.py
#添加名字
pyinstaller -n xxx.exe a.py
