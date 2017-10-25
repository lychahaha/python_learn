#jh:func-base,key-type
import getpass

#获取用户(操作系统的用户名)
print getpass.getuser()

#输入密码
a = getpass.getpass("passwd:")