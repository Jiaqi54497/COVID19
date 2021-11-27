#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm
import tensorflow as tf


# In[2]:


x_train=pd.read_csv('China_save/x_train.csv')
y_train=pd.read_csv('China_save/y_train.csv')


# In[3]:


x_train=x_train.drop(['Unnamed: 0'],axis=1)
y_train=y_train.drop(['Unnamed: 0'],axis=1)


# # Test a few architectures

# In[4]:


tf.keras.backend.set_floatx('float64')
def build_model1():
    #15/8/2
    # define the keras model
    model = Sequential()
    model.add(Dense(8, input_dim=15, activation='relu'))
    model.add(Dense(2, activation='relu'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_models(build_model, num_ite=1000):
    models=[]
    for i in tqdm(range(num_ite)):
        model=build_model()
        his=model.fit(x_train, y_train, epochs=150, batch_size=5, verbose=0)
        pre=model.predict(x_train)
        if np.all(pre > 0) and np.all(pre < 1):
#             print(pre)
            models.append(model)
#         if hishis.history['loss'][-1] < 1:
#             print('Loss:',hishis.history['loss'][-1])
#             models.append(model)
    return models

def models_predict(models, x):
    predictions=np.zeros((len(x),2))
    for model in models:
#         pre = model.predict(x)
        predictions += model.predict(x)
    return predictions/len(models)

def train_model(build_model):
    model=build_model()
    his=model.fit(x_train, y_train, epochs=150, batch_size=5, verbose=0)
    print('Loss:',hishis.history['loss'][-1])
    return model

def build_model2():
    # 15/10/6/2
    # define the keras model
    model = Sequential()
    model.add(Dense(10, input_dim=15, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(2, activation='relu'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model3():
    # 15/12/8/4/2
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=15, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model4():
    # 15/12/10/6/4/2
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=15, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[5]:


models1=train_models(build_model1)
models2=train_models(build_model2)
models3=train_models(build_model3)


# In[6]:


models4=train_models(build_model4)


# In[7]:


print(len(models1),len(models2),len(models3),len(models4))


# In[8]:


pre1=models_predict(models1,x_train)
pre2=models_predict(models2,x_train)
pre3=models_predict(models3,x_train)


# In[9]:


pre4=models_predict(models4,x_train)


# In[10]:


np.save('China_save/y_prediction1.npy',pre1)
np.save('China_save/y_prediction2.npy',pre2)
np.save('China_save/y_prediction3.npy',pre3)


# In[11]:


np.save('China_save/y_prediction4.npy',pre4)


# # Find the average policy during each peak considered

# In[12]:


split=[93,93+56]
x_train1=x_train[:split[0]]
x_train2=x_train[split[0]:split[1]]
# x_train3=x_train[split[1]:split[2]]
x_train3=x_train[split[1]:]


# In[13]:


x_mean1=x_train1.mean()
x_mean2=x_train2.mean()
x_mean3=x_train3.mean()
# x_mean4=x_train4.mean()


# In[14]:


x_mean=pd.DataFrame({'x_mean1':x_mean1,'x_mean2':x_mean2,'x_mean3':x_mean3})


# In[15]:


policy_name=['testing_policy', 'contact_tracing', 'vaccination_policy',
       'debt_relief', 'facial_coverings', 'income_support',
       'restrictions_internal_movements', 'international_travel_controls',
       'public_information_campaigns', 'cancel_public_events',
       'restriction_gatherings', 'close_public_transport', 'school_closures',
       'stay_home_requirements', 'workplace_closures']


# In[16]:


x_mean.index=policy_name


# In[17]:


x_mean.to_csv('China_save/x_mean.csv')


# Generate more inputs for prediction

# In[18]:


def more_inputs(x_mean,num=1):
    inputs=[]
    for i in range(len(x_mean)):
        x_p=x_mean.copy()
        x_p[i] = num
        inputs.append(x_p)
    return inputs


# In[19]:


x_test1=more_inputs(x_mean1)
x_test2=more_inputs(x_mean2)
x_test3=more_inputs(x_mean3)
# x_test4=more_inputs(x_mean4)


# In[20]:


x_test1=pd.DataFrame(np.array(x_test1))
x_test2=pd.DataFrame(np.array(x_test2))
x_test3=pd.DataFrame(np.array(x_test3))
# x_test4=pd.DataFrame(np.array(x_test4))


# Generate inputs with policy 0

# In[21]:


xx_test1=more_inputs(x_mean1,num=0)
xx_test2=more_inputs(x_mean2,num=0)
xx_test3=more_inputs(x_mean3,num=0)
# xx_test4=more_inputs(x_mean4,num=0)

xx_test1=pd.DataFrame(np.array(xx_test1))
xx_test2=pd.DataFrame(np.array(xx_test2))
xx_test3=pd.DataFrame(np.array(xx_test3))
# xx_test4=pd.DataFrame(np.array(xx_test4))


# Predict the outcome after changing the policies.

# In[22]:


predict1_1=models_predict(models1,x_test1)
predict1_2=models_predict(models1,x_test2)
predict1_3=models_predict(models1,x_test3)
# predict1_4=models_predict(models1,x_test4)

predict2_1=models_predict(models2,x_test1)
predict2_2=models_predict(models2,x_test2)
predict2_3=models_predict(models2,x_test3)
# predict2_4=models_predict(models2,x_test4)

predict3_1=models_predict(models3,x_test1)
predict3_2=models_predict(models3,x_test2)
predict3_3=models_predict(models3,x_test3)
# predict3_4=models_predict(models3,x_test4)


# In[23]:


predict4_1=models_predict(models4,x_test1)
predict4_2=models_predict(models4,x_test2)
predict4_3=models_predict(models4,x_test3)
# predict4_4=models_predict(models4,x_test4)


# In[24]:


np.save('China_save/predict1_1.npy',predict1_1)
np.save('China_save/predict1_2.npy',predict1_2)
np.save('China_save/predict1_3.npy',predict1_3)
# np.save('China_save/predict1_4.npy',predict1_4)

np.save('China_save/predict2_1.npy',predict2_1)
np.save('China_save/predict2_2.npy',predict2_2)
np.save('China_save/predict2_3.npy',predict2_3)
# np.save('China_save/predict2_4.npy',predict2_4)

np.save('China_save/predict3_1.npy',predict3_1)
np.save('China_save/predict3_2.npy',predict3_2)
np.save('China_save/predict3_3.npy',predict3_3)
# np.save('China_save/predict3_4.npy',predict3_4)


# In[25]:


np.save('China_save/predict4_1.npy',predict4_1)
np.save('China_save/predict4_2.npy',predict4_2)
np.save('China_save/predict4_3.npy',predict4_3)
# np.save('China_save/predict4_4.npy',predict4_4)


# In[26]:


predict_low1_1=models_predict(models1,xx_test1)
predict_low1_2=models_predict(models1,xx_test2)
predict_low1_3=models_predict(models1,xx_test3)
# predict_low1_4=models_predict(models1,xx_test4)

predict_low2_1=models_predict(models2,xx_test1)
predict_low2_2=models_predict(models2,xx_test2)
predict_low2_3=models_predict(models2,xx_test3)
# predict_low2_4=models_predict(models2,xx_test4)

predict_low3_1=models_predict(models3,xx_test1)
predict_low3_2=models_predict(models3,xx_test2)
predict_low3_3=models_predict(models3,xx_test3)
# predict_low3_4=models_predict(models3,xx_test4)

predict_low4_1=models_predict(models4,xx_test1)
predict_low4_2=models_predict(models4,xx_test2)
predict_low4_3=models_predict(models4,xx_test3)


# In[27]:


np.save('China_save/predict_low1_1.npy',predict_low1_1)
np.save('China_save/predict_low1_2.npy',predict_low1_2)
np.save('China_save/predict_low1_3.npy',predict_low1_3)
# np.save('China_save/predict_low1_4.npy',predict_low1_4)

np.save('China_save/predict_low2_1.npy',predict_low2_1)
np.save('China_save/predict_low2_2.npy',predict_low2_2)
np.save('China_save/predict_low2_3.npy',predict_low2_3)
# np.save('China_save/predict_low2_4.npy',predict_low2_4)

np.save('China_save/predict_low3_1.npy',predict_low3_1)
np.save('China_save/predict_low3_2.npy',predict_low3_2)
np.save('China_save/predict_low3_3.npy',predict_low3_3)
# np.save('China_save/predict_low3_4.npy',predict_low3_4)

np.save('China_save/predict_low4_1.npy',predict_low4_1)
np.save('China_save/predict_low4_2.npy',predict_low4_2)
np.save('China_save/predict_low4_3.npy',predict_low4_3)


# ## Try the input policy of South Africa

# In[28]:


x_mean_sa=pd.read_csv('South_Africa_save/x_mean.csv')


# In[33]:


x_mean_sa=x_mean_sa.drop(['Unnamed: 0'],axis=1)


# In[43]:


x_mean_sa=x_mean_sa.to_numpy().T


# In[44]:


predict_sa1=models_predict(models1,x_mean_sa)
predict_sa2=models_predict(models2,x_mean_sa)
predict_sa3=models_predict(models3,x_mean_sa)
predict_sa4=models_predict(models4,x_mean_sa)

np.save('China_save/predict_sa1.npy',predict_sa1)
np.save('China_save/predict_sa2.npy',predict_sa2)
np.save('China_save/predict_sa3.npy',predict_sa3)
np.save('China_save/predict_sa4.npy',predict_sa4)


# In[ ]:




