#coding=utf-8
import sqlite3

conn = sqlite3.connect('xxx.db')
conn = sqlite3.connect('"memory:')#内存数据库

conn.execute('''create table xxx
	(id int primary key not null,
	name text,
	address char(50),
	salary real);''')

conn.execute('insert into xxx(xx,xx) values(xx,xx)')

conn.commit()
conn.rollback()

cursor = conn.execute('select * from xxx')

for row in cursor:
	print row[0], row[1]

rs = cursor.fetchone()
rs = cursor.fetchmary(3)
rs = cursor.fetchall()	

cursor = conn.cursor()
rs = cursor.execute('........')
cursor.close()

conn.close()