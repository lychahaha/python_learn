#jh:proc-base,key-type
import paramiko

# ssh
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.1.1', 22, 'xxx', 'xxx', timeout=5)
stdin,stdout,stderr = ssh.exec_command('ls')
stdin.write('xx\n')
lines = stdout.readlines()
ssh.close()

# sftp
sftp.put(localpath='./a.txt', remotepath='./b.txt')
sftp.get(remotepath, localpath)

# transport
trans = paramiko.TransPort(('192.168.1.1', 22))
trans.connect(username='xxx', password='xxx')
#do_something()
trans.close()
## 启动ssh
ssh = paramiko.SSHClient()
ssh._transport = trans
## 启动sftp
sftp = paramiko.SFTPClient.from_transport(trans)

# 带密钥
pkey = paramiko.RSAKey.from_private_key_file('/home/xx/.ssh/id_rsa', password='xxx')
## ssh
ssh = paramiko.SSHClient()
ssh.connect('192.168.1.1', 22, 'xxx', pkey=pkey)
## transpose
trans = paramiko.TransPort(('192.168.1.1', 22))
trans.connect(username='xxx', pkey=pkey)
