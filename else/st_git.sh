工作区,暂存区,版本库(本地仓库),远程仓库


当前版本:HEAD
上一个版本:HEAD^
上两个版本:HEAD^^
上十个版本:HEAD~10
特定版本:2113a


# init
git init

# add
# 工作区->暂存区
git add a.txt
git add *
git add .
git rm a.txt

# checkout(文件)
## 暂存区->工作区,如果暂存区为空就是当前版本
git checkout -- a.txt

# commit 
## 暂存区->版本库
git commit -m "modify a.txt"

# reset
git reset --hard HEAD^
## 版本库->暂存区
git reset HEAD a.txt

# push
git push #将当前分支推送到默认主机
git push origin #将当前分支推送到origin仓库
git push origin master #将本地的master分支推送到origin的master分支
git push origin a:b #将本地的a分支推送到origin的b分支
git push -u origin master #推送的同时将origin设为默认主机
git push --set-upstream origin dev #创建新分支,第一次push上去
## 删除远程分支
git push origin :master
git push origin --delete master

# pull
git pull
git pull origin dev

# remote
## 查看远程仓库信息
git remote
git remote -v
## 给远程仓库地址起名字
git remote add origin git@github.com:xxx/xxx.git

# clone
git clone git@github.com:xxx/xxx.git
git clone -b dev git@github.com:xxx/xxx.git #clone指定分支
git clone --depth 1 git@github.com:xxx/xxx.git #指定拷贝的历史节点数



# checkout
## 切换当前分支
git checkout dev
## 创建并切换
git checkout -b dev
## 从远程仓库拉取本地没有的分支
git checkout -b dev origin/dev

# branch
## 查看分支
git branch
git branch -v
git branch -vv #查看本地分支和远程分支的对应关系
## 创建分支
git branch dev
## 删除分支
git branch -d dev
git branch -D dev #强行删除
## 给本地分支和远程分支建立关系
git branch -u origin/dev dev

# merge
git merge dev
## 不使用fast forward
git merge --no-ff -m 'info' dev

# stash
## 临时保存
git stash
git stash save "xxx"
## 查看保存列表
git stash list
## 取出保存内容
git stash apply #最近一次
git stash apply stash@{1}
git stash pop #取出最近一次并删除它
## 删除保存内容
git stash drop #删除最近一次
git stash drop stash@{1}
git stash clear #全部删除
## 把保存内容放到新的分支上
git stash branch dev #把最近一次弄到新分支dev上
git stash branch dev stage@{1}
## 查看保存内容和当前工作区的区别
git stash show??

# status
git status

# diff
## 对比工作区和暂存区的差别(暂存区没东西则相当于暂存区是HEAD)
git diff
## 对比工作区和版本库的差别
git diff HEAD^
## 对比暂存区和版本库的差别
git diff --cached #当前版本
git diff HEAD^ --cached
## 对比版本库和版本库的差别
git diff HEAD HEAD^
## 只查看某个文件
git diff a.txt
## 简单统计差别(多的行数和少的行数)
git diff --stat


# log
git log
## 加上内容差别
git log -p
## 加上统计差别
git log --stat
git log --shortstat #全部文件一起统计
## 加上新增修改删除文件清单
git log --name-status
git log --name-only #没有A,M,D信息
## 加上分支图
git log --graph
## 版本号只显示前几位
git log --abbrev-commit
## 时间使用相对时间
git log --relative-date
## 使用特定格式打印
git log --pretty=oneline #只打印版本号和commit信息
git log --pretty=format:"%Cred%h%Creset -%C(yellow)%d%Cblue %s %Cgreen(%cd) %C(bold blue)<%an>"
    %H 完全版本号
    %h 简短版本号
    %cn 提交者
    %ce 提交者邮件
    %cd 提交日期
    #cr 相对提交日期
    %s 提交说明
    %C 设置颜色字体(在字前面)
        %Cred
        %C(red)
        %C(red bold)
        颜色:reset(默认),normal,black,red,green,yellow,blue,magenta,cyan,white
        字体:bold,dim,ul,blink,reverse
## 设置日期格式
git --date=relative
    relative 相对日期
    local local时间
    iso iso时间
    rfc rfc时间
    short 只打印日期(YYYY-MM-DD)
    raw 时间戳
    default
git --date=format:"%Y-%m-%d %H:%M:%S"
    %y,%Y 年
    %m 月
    %b,%B 月份名字
    %d 日
    %j 一年的第几日
    %H 小时(24小时制)
    %I 小时(12小时制)
    %M 分钟
    %S 秒
    %w 星期
    %W 一年的第几个星期
    %a,%A 星期名字
    %p AM/PM
    %c 一种很好看的格式(%m/%d/%y %H:%M:%S)
## 筛选前n条
git log -n 5
## 按照日期筛选
git log --after='2014-7-1'
git log --before='2014-7-1'
## 过滤掉merge commit
git log --no-merges
## 筛选出与某个文件有关的commit
git log a.txt
## 筛选出与某句代码有关的commit
git log -S "xxx"
git log -G "xxx" #正则表达式
## 一种网上好看的写法
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"


# reflog
git reflog

# config
#git config --global user.name 'name'
#git config --global user.email '123@qq.com'