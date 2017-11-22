#jh:proc-base-onlyone,key-type
import paramiko

ssh = paramiko.SSHClient()

ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect('192.168.1.1', 22, 'username', 'password', timeout=5)

stdin,stdout,stderr = ssh.exec_command('gpustat')

stdin.write('xx')
lines = stdout.readlines()

ssh.close()

