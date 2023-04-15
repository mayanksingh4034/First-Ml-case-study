#!/usr/bin/env python
# coding: utf-8

# In[112]:


#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler


# In[113]:


data = pd.read_csv("leads.csv")
data.head()


# In[114]:


data.shape


# In[115]:


data.info()


# In[116]:


data.describe()


# In[117]:


#check for duplicates
sum(data.duplicated(subset = 'Prospect ID')) == 0


# In[118]:


#check for duplicates
sum(data.duplicated(subset = 'Lead Number')) == 0


# In[119]:


#dropping Lead Number and Prospect ID since they have all unique values

data.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[120]:


#Converting 'Select' values to NaN.

data = data.replace('Select', np.nan)


# In[121]:


#checking null values in each rows

data.isnull().sum()


# In[122]:


#checking percentage of null values in each column

round(100*(data.isnull().sum()/len(data.index)), 2)


# In[123]:


#dropping cols with more than 45% missing values

cols=data.columns

for i in cols:
    if((100*(data[i].isnull().sum()/len(data.index))) >= 45):
        data.drop(i, 1, inplace = True)


# In[124]:


#checking null values percentage

round(100*(data.isnull().sum()/len(data.index)), 2)


# In[125]:


#checking value counts of Country column

data['Country'].value_counts(dropna=False)


# In[126]:


#plotting spread of Country columnn 
plt.figure(figsize=(15,5))
s1=sns.countplot(data.Country, hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[127]:


# Since India is the most common occurence among the non-missing values we can impute all missing values with India

data['Country'] = data['Country'].replace(np.nan,'India')


# In[128]:


#plotting spread of Country columnn after replacing NaN values

plt.figure(figsize=(15,5))
s1=sns.countplot(data.Country, hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[129]:


#creating a list of columns to be droppped

cols_to_drop=['Country']


# In[130]:


#checking value counts of "City" column

data['City'].value_counts(dropna=False)


# In[131]:


data['City'] = data['City'].replace(np.nan,'Mumbai')


# In[132]:


#plotting spread of City columnn after replacing NaN values

plt.figure(figsize=(10,5))
s1=sns.countplot(data.City, hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[133]:


#checking value counts of Specialization column

data['Specialization'].value_counts(dropna=False)


# In[134]:


# Lead may not have mentioned specialization because it was not in the list or maybe they are a students 
# and don't have a specialization yet. So we will replace NaN values here with 'Not Specified'

data['Specialization'] = data['Specialization'].replace(np.nan, 'Not Specified')


# In[135]:


#plotting spread of Specialization columnn 

plt.figure(figsize=(15,5))
s1=sns.countplot(data.Specialization, hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[136]:


#combining Management Specializations because they show similar trends

data['Specialization'] = data['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                    'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Specializations')


# In[137]:


#visualizing count of Variable based on Converted value


plt.figure(figsize=(15,5))
s1=sns.countplot(data.Specialization, hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[138]:


#What is your current occupation

data['What is your current occupation'].value_counts(dropna=False)


# In[139]:


#imputing Nan values with mode "Unemployed"

data['What is your current occupation'] = data['What is your current occupation'].replace(np.nan, 'Unemployed')


# In[140]:


#checking count of values
data['What is your current occupation'].value_counts(dropna=False)


# In[141]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(data['What is your current occupation'], hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[142]:


#checking value counts

data['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[143]:


#replacing Nan values with Mode "Better Career Prospects"

data['What matters most to you in choosing a course'] = data['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[144]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(data['What matters most to you in choosing a course'], hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[145]:


#checking value counts of variable
data['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[146]:


#Here again we have another Column that is worth Dropping. So we Append to the cols_to_drop List
cols_to_drop.append('What matters most to you in choosing a course')
cols_to_drop


# In[147]:


#checking value counts of Tag variable
data['Tags'].value_counts(dropna=False)


# In[148]:


#replacing Nan values with "Not Specified"
data['Tags'] = data['Tags'].replace(np.nan,'Not Specified')


# In[149]:


#visualizing count of Variable based on Converted value

plt.figure(figsize=(15,5))
s1=sns.countplot(data['Tags'], hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[150]:


#replacing tags with low frequency with "Other Tags"
data['Tags'] = data['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized'], 'Other_Tags')

data['Tags'] = data['Tags'].replace(['switched off',
                                      'Already a student',
                                       'Not doing further education',
                                       'invalid number',
                                       'wrong number given',
                                       'Interested  in full time MBA'] , 'Other_Tags')


# In[151]:


#checking percentage of missing values
round(100*(data.isnull().sum()/len(data.index)), 2)


# In[152]:


#checking value counts of Lead Source column

data['Lead Source'].value_counts(dropna=False)


# In[153]:


#replacing Nan Values and combining low frequency values
data['Lead Source'] = data['Lead Source'].replace(np.nan,'Others')
data['Lead Source'] = data['Lead Source'].replace('google','Google')
data['Lead Source'] = data['Lead Source'].replace('Facebook','Social Media')
data['Lead Source'] = data['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others')       


# In[154]:


#visualizing count of Variable based on Converted value
plt.figure(figsize=(15,5))
s1=sns.countplot(data['Lead Source'], hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[155]:


# Last Activity:

data['Last Activity'].value_counts(dropna=False)


# In[156]:


#replacing Nan Values and combining low frequency values

data['Last Activity'] = data['Last Activity'].replace(np.nan,'Others')
data['Last Activity'] = data['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                        'Had a Phone Conversation', 
                                                        'Approached upfront',
                                                        'View in browser link Clicked',       
                                                        'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[157]:


# Last Activity:

data['Last Activity'].value_counts(dropna=False)


# In[158]:


#Check the Null Values in All Columns:
round(100*(data.isnull().sum()/len(data.index)), 2)


# In[159]:


#Drop all rows which have Nan Values. Since the number of Dropped rows is less than 2%, it will not affect the model
data = data.dropna()


# In[160]:


#Checking percentage of Null Values in All Columns:
round(100*(data.isnull().sum()/len(data.index)), 2)


# In[161]:


#Lead Origin
data['Lead Origin'].value_counts(dropna=False)


# In[162]:


#visualizing count of Variable based on Converted value

plt.figure(figsize=(8,5))
s1=sns.countplot(data['Lead Origin'], hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[163]:


#Do Not Email & Do Not Call
#visualizing count of Variable based on Converted value

plt.figure(figsize=(15,5))

ax1=plt.subplot(1, 2, 1)
ax1=sns.countplot(data['Do Not Call'], hue=data.Converted)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)

ax2=plt.subplot(1, 2, 2)
ax2=sns.countplot(data['Do Not Email'], hue=data.Converted)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
plt.show()


# In[164]:


#checking value counts for Do Not Call
data['Do Not Call'].value_counts(dropna=False)


# In[165]:


#checking value counts for Do Not Email
data['Do Not Email'].value_counts(dropna=False)


# In[166]:


cols_to_drop.append('Do Not Call')
cols_to_drop


# In[167]:


data.Search.value_counts(dropna=False)


# In[168]:


data.Magazine.value_counts(dropna=False)


# In[169]:


data['Newspaper Article'].value_counts(dropna=False)


# In[170]:


data['X Education Forums'].value_counts(dropna=False)


# In[171]:


data['Newspaper'].value_counts(dropna=False)


# In[172]:


data['Digital Advertisement'].value_counts(dropna=False)


# In[173]:


data['Through Recommendations'].value_counts(dropna=False)


# In[174]:


data['Receive More Updates About Our Courses'].value_counts(dropna=False)


# In[175]:


data['Update me on Supply Chain Content'].value_counts(dropna=False)


# In[176]:


data['Get updates on DM Content'].value_counts(dropna=False)


# In[177]:


data['I agree to pay the amount through cheque'].value_counts(dropna=False)


# In[178]:


data['A free copy of Mastering The Interview'].value_counts(dropna=False)


# In[179]:


#adding imbalanced columns to the list of columns to be dropped

cols_to_drop.extend(['Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
                 'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content',
                 'Get updates on DM Content','I agree to pay the amount through cheque'])


# In[180]:


#checking value counts of last Notable Activity
data['Last Notable Activity'].value_counts()


# In[181]:


#clubbing lower frequency values

data['Last Notable Activity'] = data['Last Notable Activity'].replace(['Had a Phone Conversation',
                                                                       'Email Marked Spam',
                                                                         'Unreachable',
                                                                         'Unsubscribed',
                                                                         'Email Bounced',                                                                    
                                                                       'Resubscribed to emails',
                                                                       'View in browser link Clicked',
                                                                       'Approached upfront', 
                                                                       'Form Submitted on Website', 
                                                                       'Email Received'],'Other_Notable_activity')


# In[182]:


#visualizing count of Variable based on Converted value

plt.figure(figsize = (14,5))
ax1=sns.countplot(x = "Last Notable Activity", hue = "Converted", data = data)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
plt.show()


# In[183]:


#checking value counts for variable

data['Last Notable Activity'].value_counts()


# In[184]:


#list of columns to be dropped
cols_to_drop


# In[185]:


#dropping columns
data = data.drop(cols_to_drop,1)
data.info()


# In[186]:


#Check the % of Data that has Converted Values = 1:

Converted = (sum(data['Converted'])/len(data['Converted'].index))*100
Converted


# In[187]:


#Checking correlations of numeric values
# figure size
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[188]:


#Total Visits
#visualizing spread of variable

plt.figure(figsize=(6,4))
sns.boxplot(y=data['TotalVisits'])
plt.show()


# In[189]:


#checking percentile values for "Total Visits"

data['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[190]:


#Outlier Treatment: Remove top & bottom 1% of the Column Outlier values

Q3 = data.TotalVisits.quantile(0.99)
data = data[(data.TotalVisits <= Q3)]
Q1 = data.TotalVisits.quantile(0.01)
data = data[(data.TotalVisits >= Q1)]
sns.boxplot(y=data['TotalVisits'])
plt.show()


# In[191]:


data.shape


# In[192]:


#checking percentiles for "Total Time Spent on Website"

data['Total Time Spent on Website'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[193]:


#visualizing spread of numeric variable

plt.figure(figsize=(6,4))
sns.boxplot(y=data['Total Time Spent on Website'])
plt.show()


# In[194]:


#checking spread of "Page Views Per Visit"

data['Page Views Per Visit'].describe()


# In[195]:


#visualizing spread of numeric variable

plt.figure(figsize=(6,4))
sns.boxplot(y=data['Page Views Per Visit'])
plt.show()


# In[196]:


#Outlier Treatment: Remove top & bottom 1% 

Q3 = data['Page Views Per Visit'].quantile(0.99)
data = data[data['Page Views Per Visit'] <= Q3]
Q1 = data['Page Views Per Visit'].quantile(0.01)
data = data[data['Page Views Per Visit'] >= Q1]
sns.boxplot(y=data['Page Views Per Visit'])
plt.show()


# In[197]:


data.shape


# In[198]:


#checking Spread of "Total Visits" vs Converted variable
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = data)
plt.show()


# In[199]:


#checking Spread of "Total Time Spent on Website" vs Converted variable

sns.boxplot(x=data.Converted, y=data['Total Time Spent on Website'])
plt.show()


# In[200]:


#checking Spread of "Page Views Per Visit" vs Converted variable

sns.boxplot(x=data.Converted,y=data['Page Views Per Visit'])
plt.show()


# In[201]:


#checking missing values in leftover columns/

round(100*(data.isnull().sum()/len(data.index)),2)


# In[202]:


#getting a list of categorical columns

cat_cols= data.select_dtypes(include=['object']).columns
cat_cols


# In[203]:


# List of variables to map

varlist =  ['A free copy of Mastering The Interview','Do Not Email']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
data[varlist] = data[varlist].apply(binary_map)


# In[204]:


#getting dummies and dropping the first column and adding the results to the master dataframe
dummy = pd.get_dummies(data[['Lead Origin','What is your current occupation',
                             'City']], drop_first=True)

data = pd.concat([data,dummy],1)


# In[205]:


dummy = pd.get_dummies(data['Specialization'], prefix  = 'Specialization')
dummy = dummy.drop(['Specialization_Not Specified'], 1)
data = pd.concat([data, dummy], axis = 1)


# In[206]:


dummy = pd.get_dummies(data['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
data = pd.concat([data, dummy], axis = 1)


# In[207]:


dummy = pd.get_dummies(data['Last Activity'], prefix  = 'Last Activity')
dummy = dummy.drop(['Last Activity_Others'], 1)
data = pd.concat([data, dummy], axis = 1)


# In[208]:


dummy = pd.get_dummies(data['Last Notable Activity'], prefix  = 'Last Notable Activity')
dummy = dummy.drop(['Last Notable Activity_Other_Notable_activity'], 1)
data = pd.concat([data, dummy], axis = 1)


# In[209]:


dummy = pd.get_dummies(data['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
data = pd.concat([data, dummy], axis = 1)


# In[210]:


#dropping the original columns after dummy variable creation

data.drop(cat_cols,1,inplace = True)


# In[211]:


data.head()


# In[212]:


from sklearn.model_selection import train_test_split

# Putting response variable to y
y = data['Converted']

y.head()

X=data.drop('Converted', axis=1)


# In[213]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[214]:


X_train.info()


# In[215]:


#scaling numeric columns

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_train.head()


# In[221]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, n_features_to_select=15)  # specify the number of features to select
rfe = rfe.fit(X_train, y_train)


# In[224]:


import statsmodels.api as sm


# In[225]:


#from sklearn.linear_model import LogisticRegression
#logreg = LogisticRegression()

#from sklearn.feature_selection import RFE
#rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
#rfe = rfe.fit(X_train, y_train)


# In[226]:


rfe.support_


# In[227]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[228]:


#list of RFE supported columns
col = X_train.columns[rfe.support_]
col


# In[229]:


X_train.columns[~rfe.support_]


# In[230]:


#BUILDING MODEL #1

X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[231]:


#dropping column with high p-value

col = col.drop('Lead Source_Referral Sites',1)


# In[232]:


#BUILDING MODEL #2

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[233]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[234]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[235]:


#dropping variable with high VIF

col = col.drop('Last Notable Activity_SMS Sent',1)


# In[236]:


#BUILDING MODEL #3
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[237]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[238]:


# Getting the Predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[239]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[240]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[241]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[242]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[243]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[244]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[245]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[246]:


# Let us calculate specificity
TN / float(TN+FP)


# In[247]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[248]:


# positive predictive value 
print (TP / float(TP+FP))


# In[249]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[250]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[251]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[252]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[253]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[254]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[255]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[256]:


#### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[257]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[258]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[259]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[260]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[261]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[262]:


# Let us calculate specificity
TN / float(TN+FP)


# In[263]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[264]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[265]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[266]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[267]:


##### Precision
TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[268]:


##### Recall
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[269]:


from sklearn.metrics import precision_score, recall_score


# In[270]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[271]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[272]:


from sklearn.metrics import precision_recall_curve


# In[273]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[274]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[275]:


#scaling test set

num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[276]:


X_test = X_test[col]
X_test.head()


# In[277]:


X_test_sm = sm.add_constant(X_test)


# In[278]:


y_test_pred = res.predict(X_test_sm)


# In[279]:


y_test_pred[:10]


# In[280]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[281]:


# Let's see the head
y_pred_1.head()


# In[282]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[283]:


# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[284]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[285]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[286]:


y_pred_final.head()


# In[287]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[288]:


y_pred_final.head()


# In[289]:


# Rearranging the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# In[290]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[291]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)


# In[292]:


y_pred_final.head()


# In[293]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[294]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[295]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[296]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[297]:


# Let us calculate specificity
TN / float(TN+FP)


# In[298]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[299]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[ ]:





# In[ ]:





# In[ ]:




