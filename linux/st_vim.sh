# vim是vi的升级版本

# vim有三大模式：命令模式、输入模式、底线模式

# 命令模式
i #切换到输入模式
: #切换到底线模式
x #删除当前字符
gg #跳转到开头
G #跳转到结尾
/xxx #向后搜索字符串
?xxx #向前搜索字符串
n #搜索下一个
N #搜索上一个
dd #剪切本行
yy #复制当前行
p #粘贴
o #切换到输入模式，并另起空行

# 输入模式
esc #切换到命令模式（或ctrl+c）

#底线模式
q #退出
w #保存
wq #保存并退出
q! #强制退出
s/aaa/bbb/g #替换当前行的所有aaa变成bbb（没有/g则只替换一个）
%s/aaa/bbb #替换全文的所有aaa变成bbb（注意不用/g）



#home目录下的.vimrc是vim的配置文件
set nocompatible #去掉与vi有关的一致性执行
set nu #显示行号
filetype on #检测文件类型
set history 1000 #记录历史行数?
set background=dark #背景黑色
syntax on #语法高亮
set autoindent #自动缩进
set smartindent #智能缩进
set tabstop=4 #设置tab的宽度为4个空格
set expandtab #使用空格代替tab
set shiftwidth=4 #?
set showmatch #显示括号匹配
set guioptions-=T #去掉vimGUI版本的toolbar
set vb t_vb= #去掉错误提示音
set ruler #右下角显示光标位置
set hlsearch #高亮搜索
set incsearch #自动启动搜索
colorscheme murphy #颜色风格
set encoding=utf-8 #设置编码格式

## 十字架光标
set cursorcolumn #高亮光标所在列
set cursorline #高亮光标所在行
hi CursorLine cterm=None ctermbg=239 term=bold guibg=NONE guifg=NONE #高亮光标所在列样式
hi CursorColumn cterm=None ctermbg=239 term=bold guibg=NONE guifg=NONE #高亮光标所在行样式
