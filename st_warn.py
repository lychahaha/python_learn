#jh:func-base
import warnings

#警告
##有过滤,相同的字符串只会出现一次
warnings.warn('xxxx')

#重置过滤
warnings.resetwarnings()

#忽略警告
warnings.filterwarnings('ignore')
