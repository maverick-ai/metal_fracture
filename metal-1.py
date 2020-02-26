import keras
import numpy as np,pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
path=(r"C:\Users\sbans\Desktop\steel")
train=pd.read_csv(path+r"\train-1.csv")
train['ImageId']=train['ImageId_ClassId'].map(lambda x:x.split('.')[0]+'.jpg')
train2=pd.DataFrame({'ImageId':train['ImageId'][::4]})
train2['e1']=train['EncodedPixels'][::4].values
train2['e2']=train['EncodedPixels'][1::4].values
train2['e3']=train['EncodedPixels'][2::4].values
train2['e4']=train['EncodedPixels'][3::4].values
train2.reset_index(inplace=True,drop=True)
train2.fillna(' ',inplace=True)
train2['count']=np.sum(train2.iloc[:,1:]!=' ',axis=1).values
class DataGenerator(keras.utils.Sequence):
    def __init__(self,df,batch_size=16,subset="train",shuffle=False,preprocess=None,info={}):
        super().__init__()
        self.df=df
        self.shuffle=shuffle
        self.subset=subset
        self.batch_size=batch_size
        self.preprocess=preprocess
        self.info=info
        if self.subset=="train":
            self.data_path=path+'\\train_image\\'
        elif self.subset=='test':
            self.data_path=path+'\\test_image\\'
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.df))/(self.batch_size))
    def on_epoch_end(self):
        self.indexes=np.arange(len(self.df))
        if self.shuffle==True:
            np.random.shuffle(self.indexes)
    def __getitem__(self,index):
        X=np.empty((self.batch_size,128,800,3),dtype=np.float32)
        y=np.empty((self.batch_size,128,800,4),dtype=np.int8)
        indexes=self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size]=f
            X[i,]=Image.open(self.data_path+f).resize((800,128))
            if self.subset=='train':
                for j in range(4):
                    y[i,:,:,j]=rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])
        if self.preprocess!=None: X=self.preprocess(X)
        if self.subset=='train': return X,y
        else: return X
def rle2maskResize(rle):
    if(pd.isnull(rle))|(rle==' '):
        return np.zeros((128,800),dtype=np.uint8)
    height=256
    width=1600
    mask=np.zeros(width*height,dtype=np.uint8)
    array=np.asarray([int(x) for x in rle.split()])
    starts=array[0::2]-1
    lengths=array[1::2]
    for index,start in enumerate(starts):
        mask[int(start):int(start+lengths[index])]=1
    return mask.reshape((height,width),order='F')[::2,::2]
def mask2contour(mask,width=3):
    w=mask.shape[1]
    h=mask.shape[0]
    mask2=np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2=np.logical_xor(mask,mask2)
    mask3=np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3=np.logical_xor(mask,mask3)
    return np.logical_xor(mask2,mask3)
def mask2pad(mask,pad=2):
    w=mask.shape[1]
    h=mask.shape[0]
    for a in range(1,pad,2):
        temp=np.concatenate([mask[a:,:],np.zeros((a,w))],axis=0)
        mask=np.logical_or(mask,temp)
    for a in range(1,pad,2):
        temp=np.concatenate([np.zeros((a,w)),mask[:-a,:]],axis=0)
        mask=np.logical_or(mask,temp)
    for a in range(1,pad,2):
        temp = np.concatenate([mask[:,a:],np.zeros((h,a))],axis=1)
        mask = np.logical_or(mask,temp)
    for a in range(1,pad,2):
        temp = np.concatenate([np.zeros((h,a)),mask[:,:-a]],axis=1)
        mask = np.logical_or(mask,temp)
    return mask
import matplotlib.pyplot as plt
plt.figure(figsize=(13.5,2.5))
bar=plt.bar([1,2,3,4],100*np.mean(train2.iloc[:,1:5]!=' ',axis=0))
plt.title('Percent Training Image with Defects',fontsize=16)
plt.ylabel('Percent of Images');plt.xlabel('Defect Type')
plt.xticks([1,2,3,4])
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f %%' % height,
             ha='center', va='bottom',fontsize=16)
plt.ylim((0,50)); plt.show()
from keras import backend as k
def dice_coef(y_true,y_pred,smooth=1):
    y_true_f=k.flatten(y_true)
    y_pred_f=k.flatten(y_pred)
    intersection=k.sum(y_true_f*y_pred_f)
    return (2.*intersection+smooth)/(k.sum(y_true_f))+k.sum(y_pred_f)+smooth
from segmentation_models import Unet
#from segmentation_models.backbones import get_preprocessing
#preprocess=get_preprocessing('resnet34')
model=Unet('resnet34',input_shape=(128,800,3),classes=4,activation='sigmoid')
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[dice_coef])
idx=int(0.8*len(train2))
train_batches = DataGenerator(train2.iloc[:idx],shuffle=True)
valid_batches=DataGenerator(train2.iloc[idx:], shuffle=True )
callbacks=[keras.callbacks.TensorBoard(log_dir=r'C:\Users\sbans\my_log_dir',histogram_freq=0)]
history=model.fit_generator(train_batches,validation_data=valid_batches,epochs=30,callbacks=callbacks)
plt.figure(figsize=(15,5))
plt.plot(range(history.epoch[-1]+1),history.history['val_dice_coef'],label='val_dice_coef')
plt.plot(range(history.epoch[-1]+1),history.history['dice_coef'],label='train_dice_coef')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Dice_coefs')
plt.legend()
plt.show()
val_set=train2.iloc[idx:]
defects=list(val_set[val_set['e1']!=' '].sample(6).index)
defects+=list(val_set[val_set['e2']!=' '].sample(6).index)
defects+=list(val_set[val_set['e3']!=' '].sample(14).index)
defects+=list(val_set[val_set['e4']!=' '].sample(6).index)
valid_batches = DataGenerator(val_set[val_set.index.isin(defects)])
preds = model.predict_generator(valid_batches)
for i,batch in enumerate(valid_batches):
    plt.figure(figsize=(20,36))
    for a in range(16):
        plt.subplot(16,2,2*a+1)
        img = batch[0][a,]
        img = Image.fromarray(img.astype('uint8'))
        img = np.array(img)
        dft = 0
        extra = '  has defect '
        for j in range(4):
            msk = batch[1][a,:,:,j]
            if np.sum(msk)!=0: 
                dft=j+1
                extra += ' '+str(j+1)
            msk = mask2pad(msk,pad=2)
            msk = mask2contour(msk,width=3)
            if j==0: # yellow
                img[msk==1,0] = 235 
                img[msk==1,1] = 235
            elif j==1: img[msk==1,1] = 210 # green
            elif j==2: img[msk==1,2] = 255 # blue
            elif j==3: # magenta
                img[msk==1,0] = 255
                img[msk==1,2] = 255
        if extra=='  has defect ': extra =''
        plt.title('Train '+train2.iloc[16*i+a,0]+extra)
        plt.axis('off') 
        plt.imshow(img)
        plt.subplot(16,2,2*a+2) 
        if dft!=0:
            msk = preds[16*i+a,:,:,dft-1]
            plt.imshow(msk)
        else:
            plt.imshow(np.zeros((128,800)))
        plt.axis('off')
        mx = np.round(np.max(msk),3)
        plt.title('Predict Defect '+str(dft)+'  (max pixel = '+str(mx)+')')
    plt.subplots_adjust(wspace=0.05)
    plt.show()