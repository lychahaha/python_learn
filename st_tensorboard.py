import tensorboardX
from tensorboardX import SummaryWriter
import torchvision

#tensorboard --logdir log
#tensorboard --logdir log --port 8888
#tensorboard --logdir log --samples_per_plugin images=100

#open-close
writer = SummaryWriter(log_dir)
writer.close()

#把scalar转换到json
writer.export_scalars_to_json('a.json')




#graph
#名字为model,inputs为一个demo input
writer.add_graph(model, inputs)


#scalar
writer.add_scalar('scalar/test', acc, global_step=epoch)
#scalars
writer.add_scalars('scalar/test', {'loss1':loss1,'loss2':loss2}, global_step=epoch)
'''
#custom scalars
layout = {
    'det':{
        'acc':['Multiline',['acc1','acc2']],
        'loss':['Multiline',['loss1','loss2']],
    },
    'reid':{
        'acc':['Multiline',['acc1','acc2']],
        'loss':['Multiline',['loss1','loss2']],
    }
}
writer.add_custom_scalars(layout)
'''


#hist
#x是tensor
writer.add_histogram('hist/test', x)


#image
#img是(3,h,w)的tensor
writer.add_image('img/test', img)
#images
#imgs是(n,3,h,w)
writer.add_images('img/test', imgs)
#image with box
#boxes是(k,4){xyxy_d}
writer.add_image_with_boxes('img/test', img, boxes)


#feature
#feats是(n,d),imgs是(n,c,h,w)
writer.add_embedding(feats, metadata=labels, label_img=imgs)


'''
#pr曲线
writer.add_pr_curve('pr/test', labels, preds, num_thresholds=127)

#audio
#snd是(k,)的tensor
writer.add_audio('audio/test', snd)

#video
#vid_tensor是(k,t,c,h,w)的tensor
writer.add_video('video/test', vid_tensor, fps=4)

#text
writer.add_text('text/test', 'text')
'''
