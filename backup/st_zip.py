#jh:class-base,key-type
import zipfile

#zipfile
zf = zipfile.ZipFile('a.zip')
zf = zipfile.ZipFile('a.zip', 'w')
##返回所有文件和目录
zf.namelist()
##文件名
zf.filename
##解压
zf.extractall()
zf.extractall(path=output_dir)
##写入
zf.write(filename)
zf.writestr(name, data=s)
##关闭
zf.close()

#文件信息
info = zf.getinfo(name)
##压缩前/后size
info.file_size
info.compress_size

#文件
f = zf.open(name)
##直接读
s = zf.read(name) #返回二进制字符串

import shutil
#压缩
shutil.make_archive('a', 'zip', 'D:/hehe/')
#解压
shutil.unpack_archive('a.zip', output_dir)
