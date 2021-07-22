import pandas as pd
import math
import numpy as np
import warnings
import sklearn

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

import os
from time import time
from tensorflow import keras
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical

# truthdf = label.copy()

def removeDifficulty(df):
    df.drop(['difficulty'], axis=1, inplace=True)

# def progressBar(current, total, barLength = 20):
    # percent = float(current) * 100 / total
    # arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    # spaces  = ' ' * (barLength - len(arrow))

    # print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

# dos = back, land, neptune, pod, smurf, teardrop
# r2l = ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster
# u2r = buffer_overflow, loadmodule, perl, rootkit,
# probe = ipsweep, nmap, portsweep, satan

def attackClassify(df):

    df.replace({
        'attack': {
        'back' : 'dos',
        'land' : 'dos',
        'neptune' : 'dos',
        'pod' : 'dos',
        'smurf' : 'dos',
        'teardrop' : 'dos',
        'udpstorm' : 'dos',
        'processtable' : 'dos',
        'mailbomb' : 'dos',
        'apache2': 'dos'}}, #apache2 was not in KDDTrain
        inplace=True)
    df.replace({
        'attack': {
        'ftp_write' : 'r2l', 
        'guess_passwd' : 'r2l',
        'imap' : 'r2l',
        'multihop' : 'r2l',
        'phf' : 'r2l',
        'spy' : 'r2l',
        'warezclient' : 'r2l',
        'warezmaster' : 'r2l',
        'snmpgetattack' : 'r2l',
        'named' : 'r2l',
        'xlock' : 'r2l',
        'xsnoop' : 'r2l',
        'sendmail' : 'r2l'}},
        inplace=True)
    df.replace({
        'attack': {
        'buffer_overflow' : 'u2r', 
        'loadmodule' : 'u2r',
        'perl' : 'u2r',
        'rootkit' : 'u2r',
        'xterm' : 'u2r',
        'ps' : 'u2r',
        'httptunnel' : 'u2r',
        'sqlattack' : 'u2r',
        'worm' : 'u2r',
        'snmpguess' : 'u2r'}},
        inplace=True)
    df.replace({
        'attack': {
        'ipsweep' : 'probe',
        'nmap' : 'probe',
        'portsweep' : 'probe',
        'satan' : 'probe',
        'saint' : 'probe',
        'mscan' : 'probe'}},
        inplace=True)
    return df
    
dict_data_types = {
        'duration': 'int64',#1
        'protocol_type': 'object',#2
        'service': 'object',#3
        'flag': 'object',#4
        # 'src_bytes': 'int32', #5
        # 'dst_bytes': 'int32', #6
        # 'land': 'int32', #7
        # 'wrong_fragment': 'int32', #8
        # 'urgent': 'int32', #9
        # 'hot': 'int32', #10
        # 'num_failed_logins': 'int32', #11
        # 'logged_in': 'int32', #12
        # 'num_compromised',#13
        # 'root_shell',#14
        # 'su_attempted',#15
        # 'num_root',#16
        # 'num_file_creations',#17
        # 'num_shells',#18
        # 'num_access_files',#19
        # 'num_outbound_cmds',#20
        # 'is_host_login',#21
        # 'is_guest_login',#22
        # 'count',#23
        # 'srv_count',#24
        # 'serror_rate', #25
        # 'srv_serror_rate',#26
        # 'rerror_rate',#27
        # 'srv_rerror_rate',#28
        # 'same_srv_rate',#29
        # 'diff_srv_rate',#30
        # 'srv_diff_host_rate',#31
        # 'dst_host_count',#32
        # 'dst_host_srv_count',#33
        # 'dst_host_same_srv_rate',#34
        # 'dst_host_diff_srv_rate',#35
        # 'dst_host_same_src_port_rate',#36
        # 'dst_host_srv_diff_host_rate',#37
        # 'dst_host_serror_rate',#38
        # 'dst_host_srv_serror_rate',#39
        # 'dst_host_rerror_rate',#40
        'attack': 'object'}#42
        # 'dst_host_srv_rerror_rate',#41
        

def oneHotEncode(train, test):
    print("One Hot Encoding")
    categorical_features = ['protocol_type', 'service', 'flag']

    #enc.fit(train)
    trainX = train.drop('attack',axis=1).copy()
    trainY = train[['attack']].copy()
    testX = test.drop('attack',axis=1).copy()
    testY = test[['attack']].copy() 

    print(trainY)

    trainX_object = trainX.select_dtypes('object')
    
    print(trainX_object)
    testX_object = testX.select_dtypes('object')
    x_ohe = OneHotEncoder(sparse=False)
    x_ohe.fit(trainX_object)
    
    trainX_codes = x_ohe.transform(trainX_object)
    
    x_feature_names = x_ohe.get_feature_names(categorical_features)
    
    print(testX_object.info())
    
    train_enc_X = pd.concat([trainX.select_dtypes(exclude='object'), 
               pd.DataFrame(trainX_codes,columns=x_feature_names)], axis=1)

    testX_codes = x_ohe.transform(testX_object)
    
    test_enc_X = pd.concat([testX.select_dtypes(exclude='object'), 
               pd.DataFrame(testX_codes,columns=x_feature_names)], axis=1)
    


    #train1 = enc.transform(train)
    # xEnc = OneHotEncoder(sparse=False)
    # encTrainX = xEnc.fit_transform(X_train)
    # encTrainY = xEnc.transform(X_test)
    
    y_ohe = OneHotEncoder(sparse=False)
    y_ohe.fit(trainY)
    train_enc_y = y_ohe.transform(trainY)
    test_enc_y = y_ohe.transform(testY)
    
    print(x_ohe.categories_)
    
    print(y_ohe.categories_)

    print("Train")
    print(train_enc_X)
    print(test_enc_X)
    print("Test")
    print(train_enc_y)
    print(test_enc_y)
    return train_enc_X, train_enc_y, test_enc_X, test_enc_y
    
    
# def logTransform(df):
    # logTransformer = FunctionTransformer(
    # np.log1p, validate=True)
    # ct = ColumnTransformer(
        # [("log",logTransformer, [0,
        
    # print("Log Transform")

    # //df1 = transformer.transform(df)
    # print(df1)
    # return df1

# def get_column_names_from_ColumnTransformer(column_transformer):    
    # col_name = []
    # for transformer_in_columns in column_transformer.transformers_[:-1]:#the last transformer is ColumnTransformer's 'remainder'
        # raw_col_name = transformer_in_columns[2]
        # raw_col_name_reverse = raw_col_name[::-1]
        # if isinstance(transformer_in_columns[1],Pipeline): 
            # transformer = transformer_in_columns[1].steps[-1][1]
        # else:
            # transformer = transformer_in_columns[1]
        # try:
            # names = transformer.get_feature_names()
            # exchange_name = [(_.split("_")) for _ in preprocessor.transformers_[:-1][0][1].steps[-1][1].get_feature_names()]
            # last_pre_name = ""
            # last_raw_name = ""
            # for pre_name,value in exchange_name:
                # if pre_name==last_pre_name:
                    # col_name.append(last_raw_name+"_"+value)
                # if pre_name!=last_pre_name:
                    # last_pre_name=pre_name
                    # last_raw_name=raw_col_name_reverse.pop()
                    # col_name.append(last_raw_name+"_"+value)
        # except AttributeError: # if no 'get_feature_names' function, use raw column name
            # names = raw_col_name
        # if isinstance(names,np.ndarray): # eg.
            # col_name += names.tolist()
        # elif isinstance(names,list):
            # col_name += names    
        # elif isinstance(names,str):
            # col_name.append(names)
    # return col_name





def logTransform(trainX, testX):

    for col in trainX:
        colmax = trainX[col].max()
        if colmax > 100:
            trainX[col] = trainX[col].apply(np.log1p)
            testX[col] = testX[col].apply(np.log1p)
    
    
    scaler = MinMaxScaler((0,1))
    scale_trainX = scaler.fit_transform(trainX)
    print(scaler.data_min_)
    print(scaler.data_max_)
    scale_testX = scaler.transform(testX)
    
    
    print(scale_trainX)
    print(scale_testX)
    
    x_feature_names = trainX.columns.values.tolist()
    
    norm_trainX = pd.DataFrame(scale_trainX,columns=x_feature_names)
    norm_testX = pd.DataFrame(scale_testX,columns=x_feature_names)
    
    print(norm_trainX)
    print(norm_testX)
    return norm_trainX, norm_testX

names = ['duration', #1
    'protocol_type', #2
    'service', #3
    'flag', #4
    'src_bytes', #5
    'dst_bytes', #6
    'land',#7
    'wrong_fragment',#8
    'urgent',#9
    'hot',#10
    'num_failed_logins',#11
    'logged_in',#12
    'num_compromised',#13
    'root_shell',#14
    'su_attempted',#15
    'num_root',#16
    'num_file_creations',#17
    'num_shells',#18
    'num_access_files',#19
    'num_outbound_cmds',#20
    'is_host_login',#21
    'is_guest_login',#22
    'count',#23
    'srv_count',#24
    'serror_rate',#25
    'srv_serror_rate',#26
    'rerror_rate',#27
    'srv_rerror_rate',#28
    'same_srv_rate',#29
    'diff_srv_rate',#30
    'srv_diff_host_rate',#31
    'dst_host_count',#32
    'dst_host_srv_count',#33
    'dst_host_same_srv_rate',#34
    'dst_host_diff_srv_rate',#35
    'dst_host_same_src_port_rate',#36
    'dst_host_srv_diff_host_rate',#37
    'dst_host_serror_rate',#38
    'dst_host_srv_serror_rate',#39
    'dst_host_rerror_rate',#40
    'dst_host_srv_rerror_rate',#41
    'attack',#42
    'difficulty']#43
train = pd.read_csv('KDDTrain+.txt', names=names, header=None, dtype=dict_data_types, index_col=False, low_memory = False)
test = pd.read_csv('KDDTest+.txt', names=names, header=None, dtype=dict_data_types, index_col=False, low_memory = False)

removeDifficulty(train)
removeDifficulty(test)

print(train.info())

trainReplaced = attackClassify(train)


testReplaced = attackClassify(test)

trainX1, trainY, testX1, testY = oneHotEncode(trainReplaced, testReplaced)

trainX2, testX2 = logTransform(trainX1, testX1)


train_X = np.asarray(trainX2.to_numpy())
train_y = np.asarray(trainY)
 
test_X = np.asarray(testX2.to_numpy())
test_y = np.asarray(testY)

input_dim = train_X.shape[1]

train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

model = keras.Sequential()

model.add(layers.SimpleRNN(units=80, input_dim=input_dim, activation="sigmoid", return_sequences=False))

model.add(layers.Dropout(0.1))

model.add(layers.Dense(5,activation='softmax'))


print(train_X)

print(train_y)

print(test_X)

print(test_y)

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="rnn-ids-checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True,monitor='val_accuracy',mode='max')
csv_logger = CSVLogger('training_set_iranalysis.csv',separator=',', append=False)

model.summary()

model.fit(train_X, train_y, batch_size=32, epochs=100, validation_data=(test_X, test_y), callbacks=[checkpointer,csv_logger])

model.save("rnnIDS_model.hdf5")

loss, accuracy = model.evaluate(test_X, test_y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(test_X)

#np.savetxt('rnnIDSpredicted.txt', np.transpose([testY,y_pred]), fmt='%01d')

# trainX2 = logTransform(trainX1)
# testX2 = logTransform(testX1)

# protocol_type_category = train.iloc[:,1].unique()
# service_category = train.iloc[:,2].unique()
# flag_category = train.iloc[:,3].unique()
# service_category = train.iloc[:,3].unique()





#df.drop(df.columns[42], axis=1, inplace=True)


# label = df[[df.columns[41]]]


# test1, train1 = oneHotEncode(train, test)


# pd.DataFrame(test1).to_csv(path_or_buf="KDDTrain+test1.csv", header=None, index=None)
# pd.DataFrame(train1).to_csv(path_or_buf="KDDTrain+train1.csv", header=None, index=None)

# print("Train")
# train2 = logTransform(train1)
# print("Test")
# test2 = logTransform(test1)

# logNumericize(df);

#df.to_csv(index=False, path_or_buf='KDDTrain+processed.csv', header=False)


# train = pd.read_csv('KDDTrain+processed.csv', header=None)

   

    # pen_column = df[col_index]
    # max_val = pen_column.max()
    # min_val = pen_column.min()
    # print("normalizing. \n")
    # print(min_val)
    # print("\n")
    # print(max_val)
    # if (max_val - min_val > 1):
        # for row_index in range(df.shape[0]):
            # currval = df.loc[row_index, col_index]
            # df.loc[row_index, col_index] = (currval - min_val)/(max_val - min_val)
            # progressBar(row_index, df.shape[0])
    
	
