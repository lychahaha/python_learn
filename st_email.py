#jh:proc-base-onlyone,key-type

from email.mime.text import MIMEText
import smtplib

msg = MIMEText('Hello,world!', 'plain', 'utf-8')

from_addr = '2521262541@qq.com'
passwd = 'secket code from qq email not password of qq'

smtp_server = 'smtp.qq.com'
to_addr = '1412441716@qq.com'

#创建连接
server = smtplib.SMTP_SSL(smtp_server, 465)
#登陆
server.login(from_addr, passwd)
#发送邮件
server.sendmail(from_addr, [to_addr], msg.as_string())