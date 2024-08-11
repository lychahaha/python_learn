#jh:proc-base,lazy-type
#coding=utf-8

from PIL import Image,ImageFilter,ImageDraw,ImageFont
import pillow_avif

#读取保存图片
im = Image.open('image.jpg')
im.save('image2.jpg')

#读取gif
im = Image.open('image.gif')#默认在第一帧
im.seek(2)#切换到第二帧

#图片缩小
im = Image.open('image.jpg')
w, h = im.size
im.thumbnail((w//2,h//2))#貌似只能缩小,不能放大
im.save('image2.jpg')

#图片模糊
im = Image.open('image.jpg')
im = im.filter(ImageFilter.BLUR)#蓝色模糊
im.save('image2.jpg')

#创建新图片
im = Image.new('RGB', (50, 50), (255,255,255))#模式,大小,填充颜色
im.save('image2.jpg')

#画点
im = Image.new('RGB', (50, 50), (255,255,255))
draw = ImageDraw.Draw(im)#创建画笔
draw.point((25, 25), fill=(0, 0, 0))#坐标,颜色
im.save('image2.jpg')

#画文字
im = Image.new('RGB', (50, 50), (255,255,255))
draw = ImageDraw.Draw(im)
font = ImageFont.truetype('C:\\WINDOWS\\Fonts\\Arial.ttf', 36)#创建字体(字体类型,字体大小)
draw.text((10, 10), 'a', font=font, fill=(0, 0, 0))#坐标,文字,字体,颜色
im.save('image2.jpg')

#调整大小
im = Image.open('image.jpg')
im = im.resize((123,123))
im.save('image2.jpg')

#裁剪
im = Image.open('image.jpg')
im = im.crop((10,20,30,40))#左上角坐标,右下角坐标
im.save('image2.jpg')

#反转
im = Image.open('image.jpg')
im = im.transpose(Image.ROTATE_180)
im.save('image2.jpg')

#粘贴
im = Image.open('image.jpg')
im2 = im.crop((0,0,50,60))
im2 = im2.transpose(Image.ROTATE_180)
im.paste(im2, (0,0,50,60))#图片,左上右下坐标
im.save('image2.jpg')

#旋转
im = Image.open('image.jpg')
im = im.rotate(45)#角度
im.save('image2.jpg')

#去色
im = Image.open('image.jpg')
im = im.convert('L')
im.save('image2.jpg')

#取RGB通道
im = Image.open('image.jpg')
r, g, b = im.split()
r.save('image2.jpg')

#png转jpg(avif/webp也适用)
im = Image.open('a.png')
im = im.convert('RGB')
im.save('a.jpg')




#模式(RGB等)
im.mode

#大小
im.size

#格式(jpg等)
im.format

#图片转矩阵
w,h = im.size
data = np.array(im.getdata(), 'float').reshape((h,w,3))

#矩阵转图片
im = Image.fromarray(data.astype(np.uint8))