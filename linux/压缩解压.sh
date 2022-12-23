# zip
zip -r xxx.zip xxx_dir #把xxx_dir目录压缩成xxx.zip

# unzip
unzip xxx

# tar
tar ...
    -z #输出重定向给gzip
    -c #创建新tar文件
    -x #从tar文件提取文件
    -v #显示提取文件名
    -f xxx #输出到xxx或从xxx提取

tar -cvf xxx.tar xxx_dir #压缩成tar文件
tar -czvf xxx.tar.gz xxx_dir #压缩成tar.gz文件
tar -xvf xxx.tar #解压tar文件
tar -zxvf xxx.tar.gz #解压tar.gz文件
