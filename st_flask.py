from flask import request,make_response,session
from flask import render_template,url_for,redirect,abort
from flask import Blueprint
from flask import flash


@app.route('/login', methods=['GET','POST'])
def login():
	return render_template('login.html', title='login')

@app.route('/user/<id>')
def user(id):
	return redirect(url_for('index', id=id))

#request
#获取方法
request.method
#表单参数获取
request.form.get('username')
request.form['username']
#获取url中参数
request.args.get('username', '')#左边是键,右边是默认值
#文件获取和保存
from werkzeug import secure_filename
f = request.files['myFile']
f.save('/static/'+secure_filename(f.filename))
#获取请求头部
request.headers#一个字典
#url信息,例如(http://www.example.com/myapplication/page.html?x=y)
request.path#/page.html
request.base_url#http://www.example.com/myapplication/page.html
request.url_root#http://www.example.com/myapplication/
request.script_root#/myapplication
request.url#http://www.example.com/myapplication/page.html?x=y
#cookie
request.cookies.get('username')

@app.route('/login')
def login():
	resp = make_response(render_template('login.html'))
	resp.set_cookie('username', 'myUser')
	return resp


#错误
@app.route('/login')
def login():
	abort(404)

@app.errorhandler(404)
def page_not_found():
	return render_template('error.html'), 404	


#视图函数返回对象
#1.str:flask会自动补充http头
#2.response对象
#3.元组(response, status, headers)
@app.errorhandler(404)
def page_not_found():
	resp = make_response(render_template('error.html'), 404)
	resp.set_cookie('username', 'myUser')
	return resp


#response
#头部信息
response.headers
#状态
response.status
response.status_code#整数状态
#cookie
response.set_cookie('username', 'myUser')


#session
if 'username' not in session:
	session['username'] = 'myUser'
session.pop('username', None)	


#message
flash('haha')
#in jinja2
get_flashed_messages()


#日志
app.logger.debug('haha')
app.logger.warning('haha')
app.logger.error('haha')


#jinja2中可用变量和函数
request,session,g,config
url_for,get_flashed_messages
#jinja2自定义可用变量和函数
@app.context_processor
def add():
	def myfx():
		return 1
	return dict(myfx=myfx, user=1)#字典里的键即为自定义可用变量和函数

#蓝图
admin = Blueprint('admin', __name__)
@admin.route('/test')
def test():
	return 'test'
app.register_blueprint(admin, url_prefix='/admin')	