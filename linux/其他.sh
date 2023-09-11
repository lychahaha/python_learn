# ignore:[ansible],[pxe+kickstart]

# bc
bc #计算器

# alias
alias -p #显示命令别名
alias li='ls -il' #设置命令别名
alias ..='cd ..'
dssh() { docker exec -t -i $1 /bin/bash; }

# history
history #查看历史命令执行记录（可用类型!13的写法执行历史记录）
history -c #清空历史记录
