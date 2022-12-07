import pandas as pd
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # important to avoid an error (the truncated picture error)
from google.cloud import storage
# from FungAI.ml.params import LOCAL_DATA_PATH
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def make_dataframes(train_dir):
    bad_images=[]
    dirlist=[train_dir]
    names = ['train']

    for name,d in zip(names, dirlist):
        filepaths=[]
        labels=[]
        classlist=sorted(os.listdir(d) )
        for klass in classlist:
            classpath=os.path.join(d, klass)
            flist=sorted(os.listdir(classpath))
            desc=f'{name:6s}-{klass:25s}'
            for f in flist:
                fpath=os.path.join(classpath,f)
                try:
                    filepaths.append(fpath)
                    labels.append(klass)
                except:
                    bad_images.append(fpath)
        Fseries=pd.Series(filepaths, name='filepaths')
        Lseries=pd.Series(labels, name='labels')
        df=pd.concat([Fseries, Lseries], axis=1)

        pdf=df
        train_df, dummy_df=train_test_split(pdf, train_size=.8, shuffle=True, random_state=123, stratify=pdf['labels'])
        valid_df, test_df=train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])

    return train_df, test_df, valid_df



# def trim(df, max_samples=240, column='labels'):
#     df=df.copy()
#     groups=df.groupby(column)

#     trimmed_df = pd.DataFrame(columns = df.columns)
#     groups=df.groupby(column)
#     for label in df[column].unique():
#         group=groups.get_group(label)
#         count=len(group)
#         sampled_group=group.sample(n=max_samples, random_state=123,axis=0)
#         trimmed_df=pd.concat([trimmed_df, sampled_group], axis=0)


#     return trimmed_df




##########-----------TRAINING/EVALUATING/PREDICTING MODEL-----------##########

'''What make_gens does is creates is data augmentation in the training ds.
For the validation and the test it gives the img_size and the correct batch size.'''

def make_gens(batch_size, ycol, train_df, test_df, valid_df, img_size):
    trgen=ImageDataGenerator()
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



    test_gen=t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

    return train_gen, test_gen, valid_gen


def load_cloud() :
    '''❗️NOT WORKING❗️'''

    BUCKET_NAME = "zipped_mushrooms"

    storage_filename = "data/raw/train_1k.csv"
    local_filename = "train_1k.csv"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(storage_filename)
    blob.download_to_filename(local_filename)
