#jh:proc-base-onlyone,key-type
#coding=utf-8

from email.mime.text import MIMEText
import smtplib

msg = MIMEText('Hello,world!', 'plain', 'utf-8')

from_addr = '2521262541@qq.com'
passwd = '785412896523JHwt'

smtp_server = 'smtp.qq.com'
to_addr = '1412441716@qq.com'

#创建连接
server = smtplib.SMTP(smtp_server)
#ssl安全连接
server.starttls()
#设置log
server.set_debuglevel(1)
#登陆
server.login(from_addr, passwd)
#发送邮件
server.sendmail(from_addr, [to_addr], msg.as_string())
#退出
server.quit()