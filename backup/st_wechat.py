#jh:func-base,lazy-type
#coding=utf-8

import itchat
from itchat.content import *

#登录
'''
loginCallback:登录回调函数
exitCallback:登出回调函数
hotReload:是否热启动(不用每次都扫码)
enableCmdQR:是否在命令行显示二维码,0表示否,正数表示背景色黑色,负数表示白色,数值表示字符宽度
'''
itchat.auto_login()

#登出
itchat.logout()

#运行
itchat.run()

#发送消息
'''
msg:文本内容
toUserName:接收者id(不是昵称),留空则发送给自己
返回值:一个结构体,可以转换成bool,True表示发送成功,False表示发送失败

发送时:
文件:@fil@文件地址
图片:@img@图片地址
视频:@vid@视频地址

这个地址为程序目录下的地址
'''
itchat.send(msg, toUserName)

#发送文件,图片,视频
itchat.send_file(fileDir, toUserName)
itchat.send_image(fileDir, toUserName)
itchat.send_video(fileDir, toUserName)#要求mp4格式

#发送消息返回值结构体
{
	u'MsgID': u'3652519825998128687',
	u'LocalID': u'14867834828810',
	u'BaseResponse':
	{
		u'ErrMsg': u'\u8bf7\u6c42\u6210\u529f',
		u'Ret': 0,
		'RawMsg': u'\u8bf7\u6c42\u6210\u529f'
	}
}

#接收消息句柄
'''
参数msg:消息
返回值:如果返回值不是None,表示要回复的内容
'''
@itchat.msg_register([TEXT])
def simple_reply(msg):
	return msg['Text']

#消息msg
msg['Text']#消息内容(或参数)
msg['FromUserName']#发送者id
msg['Content']#消息文本(非Text时为空)
msg['Type']#消息类型:TEXT,PICTURE等
msg['FileName']#文件名

{
	u'AppInfo': {u'Type': 0, u'AppID': u''},
	u'ImgWidth': 120,
	u'FromUserName': u'@ebba124559fc13392d03ddeb98afc9e94d3fc49031abd133eb9d0a45fdaf08c3',
	u'PlayLength': 0,
	u'OriContent': u'',
	u'ImgStatus': 2,
	u'RecommendInfo':
	{
		u'UserName': u'',
		u'Province': u'',
		u'City': u'',
		u'Scene': 0,
		u'QQNum': 0,
		u'Content': u'',
		u'Alias': u'',
		u'OpCode': 0,
		u'Signature': u'',
		u'Ticket': u'',
		u'Sex': 0,
		u'NickName': u'',
		u'AttrStatus': 0,
		u'VerifyFlag': 0
	},
	u'Content': u'<?xml version="1.0"?>\n<msg>\n\t<img aeskey="23dbd8331176498b809ab22d5c18bda9" encryver="1" cdnthumbaeskey="23dbd8331176498b809ab22d5c18bda9" cdnthumburl="3050020100044930470201000204af54b95d02033d14bb0204d58033790204589ee9060425617570696d675f663562383162316230663034336530645f313438363830393335323138360201000201000400" cdnthumblength="18674" cdnthumbheight="120" cdnthumbwidth="120" cdnmidheight="0" cdnmidwidth="0" cdnhdheight="0" cdnhdwidth="0" cdnmidimgurl="3050020100044930470201000204af54b95d02033d14bb0204d58033790204589ee9060425617570696d675f663562383162316230663034336530645f313438363830393335323138360201000201000400" length="84369" cdnbigimgurl="3050020100044930470201000204af54b95d02033d14bb0204d58033790204589ee9060425617570696d675f663562383162316230663034336530645f313438363830393335323138360201000201000400" hdlength="58921" md5="cf083eab35c1b9ca3c520becff735c8f" />\n</msg>\n',
	u'MsgType': 3,
	u'ImgHeight': 120,
	u'StatusNotifyUserName': u'',
	u'StatusNotifyCode': 0,
	'Type': 'Picture',
	u'NewMsgId': 4704246212235256117L,
	u'Status': 3,
	u'VoiceLength': 0,
	u'MediaId': u'',
	u'MsgId': u'4704246212235256117',
	u'ToUserName': u'@c1500d4ac6bfbaeb2c27d06e2eebd64b025f323c1ee35a78ea42f198006010f2',
	u'ForwardFlag': 0,
	u'FileName': '170211-183759.png',
	u'Url': u'',
	u'HasProductId': 0,
	u'FileSize': u'',
	u'AppMsgType': 0,
	'Text': <function download_fn at 0x03705AF0>,
	u'Ticket': u'',
	u'CreateTime': 1486809478,
	u'SubMsgType': 0
}

#消息msg类型含义与msg['Text']的含义
'''
TEXT 		文本 		文本内容
PICTURE 	图片/表情	下载方法
RECORDING	语音		下载方法
ATTACHMENT	附件		下载方法
VIDEO		视频		下载方法
FRIENDS		好友邀请	添加好友所需参数
SHARING		分享		分享名称
NOTE		通知		通知文本
CARD		名片		推荐人字典
MAP			地图		位置文本
Useless		无用信息	?
'''

#导入登录信息
'''
如之前登录时选择热启动,则自动登录,并返回true
fileDir:导入文件名
'''
itchat.load_login_status()

#导出登录信息
'''
fileDir:导出文件名
'''
itchat.dump_login_status()

#获取朋友列表
'''
update:False->本地获取
返回值:list,每个元素是一个字典
'''
itchat.get_friends(True)

#搜索用户信息
'''
name:备注|昵称|微信号
userName:用户id
nickName:昵称
remarkName:备注
wechatAccount:微信号
'''
itchat.search_friends(name='abc', wechatAccount='dfs')

#friend结构体
{
	#用户id
	u'UserName': u'@d4a31cfc1790ce8347194e63b429a3312e71c463ceff92059e72da41caf32e7f',
	#城市
	u'City': u'',
	#
	u'DisplayName': u'',
	#
	u'UniFriend': 0,
	#
	u'MemberList': [],
	#昵称拼音
	u'PYQuanPin': u'liangyimi',
	#
	u'RemarkPYInitial': u'',
	#性别(男1女2)
	u'Sex': 2,
	#
	u'AppAccountFlag': 0,
	#
	u'VerifyFlag': 0,
	#
	u'Province': u'',
	#
	u'KeyWord': u'',
	#备注名称
	u'RemarkName': u'',
	#昵称拼音首字母
	u'PYInitial': u'LYM',
	#
	u'IsOwner': 0,
	#
	u'ChatRoomId': 0,
	#
	u'HideInputBarFlag': 0,
	#
	u'EncryChatRoomId': u'',
	#
	u'AttrStatus': 16782119,
	#
	u'SnsFlag': 1,
	#
	u'MemberCount': 0,
	#
	u'OwnerUin': 0,
	#微信号
	u'Alias': u'xxiaoyimi',
	#
	u'Signature': u'',
	#
	u'ContactFlag': 2051,
	#昵称
	u'NickName': u'\u6881\u858f\u7c73',
	#备注名称拼音
	u'RemarkPYQuanPin': u'',
	#头像url
	u'HeadImgUrl': u'/cgi-bin/mmwebwx-bin/webwxgeticon?seq=647795112&username=@d4a31cfc1790ce8347194e63b429a3312e71c463ceff92059e72da41caf32e7f&skey=@crypt_6c9834bb_8dcc62e8fb04198bd542a32eb7a4303b',
	#
	u'Uin': u'wxid_bligjgb22any22',
	#
	u'StarFriend': 0,
	#
	u'Statues': 0
}