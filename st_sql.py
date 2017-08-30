import MySQLdb

connect = MySQLdb.connect(host='127.0.0.1', port=3306, user='root', passwd='520lwt', db='jspweb', charset='utf8')
cursor = connect.cursor()

#查询

sql = 'select * from user'
cursor.execute(sql)

print cursor.rowcount

rs = cursor.fetchone()
print rs

rs = cursor.fetchmany(3)
print rs

rs = cursor.fetchall()
print rs

#修改

#自动提交取消
connect.autocommit(False)

#正常结束事务
connect.commit()

#异常结束事务
connect.rollback()

try:
	sql = "insert into user(username, password) values('haha', 'haha'))"
	cursor.execute(sql)
	connect.commit()
except:
	connect.rollback()

cursor.close()
connect.close()