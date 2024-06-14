import numpy as np
import matplotlib.pyplot as plt
depthmap = np.load('cae_lstm_predict.npy')
dep=np.load('sod.npy')
depth=np.load('cae_lstm_predict4.npy')
depp=np.load('cae_lstm_predict2.npy')
print(dep.shape)
print(depthmap.shape)
print(depth.shape)#使用numpy载入npy文件
print(depp.shape)
#time_true = np.arange(0, 461)
y1=[]
y2=[]
y3=[]
y4=[]
x=[]
cha=0
for time in range(0,256):
    #timeture=time+cha
    x1=0.00392156862745*cha
    cha=cha+1
    x.append(x1)
    ytrue=np.mean(dep[:,time])
    ypre=np.mean(depthmap[:,time])
    ypre4=np.mean(depth[:,time])
    ypre2=np.mean(depp[:,time])
    y1.append(ytrue)
    y2.append(ypre)
    y3.append(ypre4)
    y4.append(ypre2)
plt.plot(x, y1, 'k-',label='true')
plt.plot(x,y2,'b--',label='prediction_8')
plt.plot(x,y3,'r--',label='prediction_4')
plt.plot(x,y4,'g--',label='prediction_2')
plt.legend()
plt.ylim(0,1.1)
plt.xlim(0,1.0)
my_y_ticks = np.arange(0, 1.1, 0.25)
my_x_ticks = np.arange(0, 1.2, 0.2)
plt.yticks(my_y_ticks)
plt.xticks(my_x_ticks)
plt.title('Sod-CAE-LSTM-reconstruction')
plt.ylabel('ρ')
plt.xlabel('x')
plt.savefig('cae-lstm_chonggou.png')

#plt.imshow(depthmap)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# plt.colorbar()                   #添加colorbar
#plt.savefig('depthmap.png')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
#plt.show()     
