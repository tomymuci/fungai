import numpy as np
import os
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True # important to avoid an error (the truncated picture error)
from google.cloud import storage
from FungAI.ml.params import LOCAL_DATA_PATH
import cv2
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers

def make_dataframes(train_dir,test_dir, val_dir):

    bad_images=[]
    dirlist=[train_dir, test_dir, val_dir]
    names=['train','test', 'valid']
    zipdir=zip(names, dirlist)
    for name,d in zipdir:
        filepaths=[]
        labels=[]
        classlist=sorted(os.listdir(d) )
        for klass in classlist:
            classpath=os.path.join(d, klass)
            flist=sorted(os.listdir(classpath))
            desc=f'{name:6s}-{klass:25s}'
            for f in tqdm(flist, ncols=130,desc=desc, unit='files', colour='blue'):
                fpath=os.path.join(classpath,f)
                try:
                    img=cv2.imread(fpath)
                    shape=img.shape
                    filepaths.append(fpath)
                    labels.append(klass)
                except:
                    bad_images.append(fpath)
        Fseries=pd.Series(filepaths, name='filepaths')
        Lseries=pd.Series(labels, name='labels')
        df=pd.concat([Fseries, Lseries], axis=1)
        if name =='valid':
            valid_df=df
        elif name == 'test':
            test_df=df
        else:
            if test_dir == None and val_dir == None:
                pdf=df
                train_df, dummy_df=train_test_split(pdf, train_size=.8, shuffle=True, random_state=123, stratify=pdf['labels'])
                valid_df, test_df=train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])
            elif test_dir == None:
                pdf=df
                train_df,test_df=train_test_split(pdf, train_size=.8, shuffle=True, random_state=123, stratify=pdf['labels'])
            else : # create a  validation dataframe
                pdf=df
                train_df,valid_df=train_test_split(pdf, train_size=.8, shuffle=True, random_state=123, stratify=pdf['labels'])
    return train_df, test_df, valid_df

def trim(df, max_samples, min_samples, column):
        df=df.copy()
        classes=df[column].unique()
        class_count=len(classes)
        length=len(df)
        groups=df.groupby(column)
        trimmed_df = pd.DataFrame(columns = df.columns)
        groups=df.groupby(column)
        for label in df[column].unique():
            group=groups.get_group(label)
            sampled_group=group
            trimmed_df=pd.concat([trimmed_df, sampled_group], axis=0)

        return trimmed_df, classes, class_count


##########-----------TRAINING/EVALUATING/PREDICTING MODEL-----------##########

'''What make_gens does is creates is data augmentation in the training ds.
For the validation and the test it gives the img_size and the correct batch size.'''

def make_gens(batch_size, ycol, train_df, test_df, valid_df, img_size):
    trgen=ImageDataGenerator(horizontal_flip=True)
    t_and_v_gen=ImageDataGenerator()
#Create the training set based on the training_df created above
    train_gen=trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

#Create the validation set based
    valid_gen=t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

#CREATING TEST SET

    # for the test_gen we want to calculate the batch size and test steps such that batch_size X test_steps= number of samples in test set
    # this insures that we go through all the sample in the test set exactly once.

    length=len(test_df)
    test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]
    test_steps=int(length/test_batch_size)


    test_gen=t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

    return train_gen, test_gen, valid_gen, test_steps





def load_cloud() :
    '''❗️NOT WORKING❗️'''

    BUCKET_NAME = "zipped_mushrooms"

    storage_filename = "data/raw/train_1k.csv"
    local_filename = "train_1k.csv"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(storage_filename)
    blob.download_to_filename(local_filename)
