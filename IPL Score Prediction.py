#!/usr/bin/env python
# coding: utf-8

# # Exloratory Data Analysis

# In[178]:


# import necessary library
import numpy as np # for mathematical calculation
import pandas as pd # for dataframe manipulation 
import matplotlib.pyplot as plt # for visualisation
import seaborn as sns # for various visualisation

get_ipython().run_line_magic('matplotlib', 'inline')


# In[179]:


#import dataframe 

df = pd.read_csv("ipl.csv")
df.head()


# In[180]:


# about dataframe
df.info()
df.describe()


# In[181]:


# different features
df.columns


# In[182]:


#seprate numerical and catagorical feature 
numerical = df.select_dtypes(np.number).columns.tolist()
print (numerical)

catagorical = df.select_dtypes('object').columns.tolist()
print (catagorical)


# # Feature engineering and Selection Process

# In[183]:


# check is null or missing labels are in dataframe.
df.isnull().sum()


# In[184]:


# Removing uncesssary feature 
remove_columns = [ 'batsman', 'bowler','mid',  'striker', 'non-striker','venue','date']
df.drop(labels=remove_columns,axis = 1, inplace  = True)


# In[185]:


# unique team
df['bat_team'].unique()


# In[186]:


# select top 8 teams that are playing ipl now a days
playing_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Delhi Daredevils',
        'Sunrisers Hyderabad']
df = df[df['bat_team'].isin(playing_teams)&df['bowl_team'].isin(playing_teams)]


# In[187]:


df.head()


# In[188]:


# remove first 5 overs from every match 
df = df[df['overs']>=5]
df['overs']


# In[189]:


# # convert the column from date string to datetime object
# from datetime import datetime
# df['date'] = df['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d') )


# In[190]:


## handling catagorical variable 
 # one hot encoding
encoded_df = pd.get_dummies(df,columns=['bat_team','bowl_team'])


# In[191]:


encoded_df.head()


# In[192]:


encoded_df.columns


# In[193]:


df = encoded_df[[ 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 
       'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad','total']]
df.head()


# In[194]:


## to getting correlation of every feautres in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,15) )
# ploting heat map for visualisation ..
m = sns.heatmap(df[top_corr_features].corr(),annot= True, cmap="RdYlGn")


# In[195]:


# Split dataframe into train  and test 

from sklearn.model_selection import train_test_split
x = df.iloc[:,0:-1]  #independent columns
y = df.iloc[:,-1]    #target column i.e outcome

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state = 42)


# # Try different types of model
# 

# In[196]:


# linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[197]:


prediction = regressor.predict(x_test)


# In[198]:


# visualize my model
sns.distplot(y_test-prediction)


# In[199]:


from sklearn import metrics
print('MAE :',metrics.mean_absolute_error(y_test,prediction))

print('MSE :',metrics.mean_squared_error(y_test,prediction))

print('RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[200]:


# rigde Regression
from sklearn.linear_model import Ridge
Ridge_regressor = Ridge()
Ridge_regressor.fit(x_train,y_train)


# In[201]:


prediction = Ridge_regressor.predict(x_test)


# In[202]:


# visualize my model
sns.distplot(y_test-prediction)


# In[203]:


from sklearn import metrics
print('MAE :',metrics.mean_absolute_error(y_test,prediction))

print('MSE :',metrics.mean_squared_error(y_test,prediction))

print('RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[204]:


from sklearn.linear_model import Lasso
Lasso_regressor = Lasso()
Lasso_regressor.fit(x_train,y_train)


# In[205]:


prediction =Lasso_regressor.predict(x_test)


# In[206]:


# visualize my model
sns.distplot(y_test-prediction)


# In[207]:


from sklearn import metrics
print('MAE :',metrics.mean_absolute_error(y_test,prediction))

print('MSE :',metrics.mean_squared_error(y_test,prediction))

print('RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[208]:


# for more finetuned our model we can used hyperparameter tunning .
# such as RandomsearchCV.........


# In[216]:


# Creating the pickel file for the regressor
import pickle 
filename = 'IPL_Score_Prediction.pkl'
pickle.dump( Lasso_regressor,open(filename,'wb')) 


# In[214]:





# In[ ]:




