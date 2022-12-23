# 外部操作
tmux [new -s sess_name -n win_name] #创建新会话
tmux kill-session -t sess_name #删除会话
tmux rename-session -t old_sess_name new_sess_name #重命名会话
tmux ls #显示会话列表
tmux a #连接上一次的会话
tmux at [-t sess_name] #连接会话
# 会话操作
cb s #列出所有会话,然后切换
cb d #退出会话
cb : #进入命令行模式
cb [ #进入复制模式?

# 窗口
cb c #创建新窗口
cb & #关闭当前窗口
cb n #进入下一个窗口
cb p #进入上一个窗口
cb w #列出所有窗口,然后切换
cb 数字 #进入该id的窗口
cb , #重命名当前窗口

# 窗格
cb % #当前窗格分成左右两块
cb " #当前窗格分成上下两块
cb x #删除当前窗格
cb ↑↓←→ #切换窗格
cb q #显示窗格id
cb o #切换到下一个id的窗格
cb { #与下一个窗格交换位置
cb } #与上一个窗格交换位置
cb 空格 #重新布局窗格
cb ! #将当前窗格放到一个新窗口
cb ctrl+箭头 #分割线移动1个单位
cb alt+箭头 #分割线移动5个单位


#命令(~/.tmux.conf)
## 前缀绑定 (Ctrl+a)
set -g prefix ^a
unbind ^b
bind a send-prefix
## 分割窗口快捷键修改
unbind '"'
bind - splitw -v
unbind %
bind | splitw -h
## 其他
set -g history-limit 65535 #修改缓冲区上限
set -g mouse on #启用鼠标