#!/usr/bin/env python
# coding: utf-8

# In[435]:


import pandas as pd;
import numpy as np;


# In[234]:


dfa = pd.read_csv("classification.csv")


# In[235]:


dfa.head()


# In[236]:


dfa.tail()


# In[237]:


dfa.describe()


# In[238]:


dfa[dfa['age']==27]


# In[239]:


dfa.info()


# ##Question 2 - replace missing values with np.nan; HOw many np.nan values are in the dataset

# In[240]:


dfa['age'].replace(999, np.nan, inplace=True)


# In[241]:


dfa['pdays']


# In[242]:


dfa.isna().sum()


# In[243]:


dfa.isna().sum().sum()


# In[244]:


dfa.describe()


# ##Question 3 - Percentage of rows that have at least one missing value

# In[245]:


rowsmissing = sum(dfa.apply(lambda x: sum(x.isnull().values), axis = 1)>0)
percentrowmissing = rowsmissing / dfa.shape[0]
print(percentrowmissing)


# In[246]:


#Get new dataframe with rows that have at least one missing value
df_missing = dfa[dfa.isna().any(axis=1)]


# In[247]:


(len(df_missing) / len(dfa)) * 100


# ##Question4 - Number of columns that have missing values of any form

# In[248]:


sum(dfa.apply(lambda x: sum(x.isnull().values), axis = 0)>0)


# ##Question 5 - Median of column 'duration'

# In[249]:


dfa['duration'].median()


# ##Question 6 - How missing values are marked in the dataset

# In[250]:


dfb = pd.read_csv("regression.csv")


# In[251]:


dfb.head()


# In[252]:


dfb.tail()


# In[253]:


dfb.info()


# In[254]:


dfb.describe()


# In[255]:


#missing values in each column
dfb.isna().sum()


# In[256]:


#count of missing values in the entire dataset
dfb.isna().sum().sum()


# #Question 7 - Change datatype of column DateofHire and DateofTermination;
# ##Replace missing values in the column DateOfTermination with 1/1/2022 
# Add new column "Years of Tenure" - holds the number of years a person worked in the company until the beginning of 2022
#     (find the difference between the termination or 1/1/2022 and the hiring date)
# What is the average of this new column? (4 digits after the decimal point) 
# Drop columns DateofHire and DateofTermination

# In[257]:


dfb['DateofHire'] = pd.to_datetime(dfb['DateofHire'])
dfb['DateofTermination'] = pd.to_datetime(dfb['DateofTermination'])

#new way
#dfb['DateofHire'] = pd.to_datetime(dfb['DateofHire']).dt.date
#dfb['DateofTermination'] = pd.to_datetime(dfb['DateofTermination']).dt.date


# In[258]:


dfb.info()
dfb.head()


# In[259]:


#Replace missing values in column DateofTermination with 1/1/2022


# In[260]:


dfb['DateofTermination'].replace(np.nan, '1/1/2022', inplace=True)


# In[261]:


dfb.head()


# In[262]:


#Add new column "Years of Tenure" # of years a person worked in the company until the beginning of 2022


# In[263]:


dfb['DateofHire'] = dfb['DateofHire'].astype(np.datetime64)
#dfb['DateofTermination'].fillna('1/1/2022', inplace = True)
dfb['DateofTermination'] = dfb['DateofTermination'].astype(np.datetime64)
dfb['Years of Tenure'] = dfb['DateofTermination'].dt.year - dfb['DateofHire'].dt.year
dfb['Years of Tenure'] = dfb['Years of Tenure'].astype(int)


# In[264]:


dfb[['DateofHire', 'DateofTermination', 'Years of Tenure']]


# In[265]:


dfb['Years of Tenure'].mean()


# In[266]:


#drop DateofHire and DateofTermination columns from df
dfb2 = dfb.drop(['DateofHire', 'DateofTermination'], axis=1)


# ##Question 8 - Add new column "Age". Diference between 2022 and year of birth of each employee. 
# Drop DOB from dataset
# Median age of employees in the dataset?

# In[267]:


dfb2['Age'] = 2022 - pd.to_datetime(dfb2['DOB']).dt.year


# In[268]:


dfb2.head()


# In[269]:


dfb2 = dfb2.drop(['DOB'], axis=1)


# In[270]:


dfb2['Age'].median()


# In[271]:


#Question 9 - How many rows in the dataset have at least one missing value
dfb_missing = dfb[dfb.isna().any(axis=1)]
print(dfb_missing)
len(dfb_missing)


# In[272]:


(len(dfb_missing) / len(dfb)) * 100


# In[273]:


dfb2['Absences'].mean()


# # Lab 2

# ##Question 1 - Replace missing values in 'marital' column with most frequent value. What is the frequency of the value after imputation?

# In[425]:


df1 = dfa 
df1.mode()


# In[426]:


df1['marital'].replace(np.nan, 'married', inplace=True)


# In[427]:


df1['marital'].value_counts()


# ##Question2 - Plot histogram of 'duration' column. What is the distribution?

# In[428]:


df1.hist(column='duration')


# ##Question 3 - Replace the missing values in the column 'duration' with the median of all available values in this column. What is the mean of all values in this column after data imputation? Enter the answer with four digits after the decimal point.

# In[429]:


df1['duration'].median()


# In[430]:


df1['duration'].replace(np.nan, dfa['duration'].median(), inplace=True)


# In[431]:


df1['duration'].mean()


# ##Question 4 - pdays

# In[434]:


df1['pdays'] = np.where(
    (df1['pdays'] < 999) , "contact", df1['pdays'])


# In[433]:


df1['pdays'].value_counts()


# In[283]:


df1['pdays'].replace("999.0", "no contact", inplace=True)


# In[284]:


df1['pdays'].value_counts() #double check 4, 5


# ##Question 5

# In[285]:


df1['pdays'].replace("nan", "no contact", inplace=True)


# In[286]:


df1['pdays'].value_counts()


# ##Question 6 - Replace missing values in the age column with the mean of available values

# In[287]:


df2 = dfa[dfa.notnull()]
meanage = df2['age'].mean()


# In[288]:


df1['age'].describe()


# In[289]:


df1['age'].replace(np.nan, meanage, inplace=True)


# In[290]:


df1['age'].describe()


# In[291]:


df1['age'].mean()


# ##Question 7 - Job column transformation (apply one-hot vector of 4)

# In[292]:


df1['job'].describe()


# In[293]:


df1['job'].value_counts()


# In[294]:


jobs = pd.get_dummies(df1['job'], prefix='job', drop_first=False)
jobs


# In[295]:


#jobs = jobs.drop(['unknown'], axis=1)
#jobs


# In[296]:


df_trans = pd.concat([df1, jobs], axis=1)


# In[297]:


df_trans.head()
df_trans.tail()


# In[298]:


df_trans = df_trans.drop(['job'], axis=1)


# In[299]:


##Question 8 - Marital column transformations


# In[300]:


df_trans['marital'].value_counts()


# In[301]:


status = pd.get_dummies(df_trans['marital'], prefix='marital')
status


# In[302]:


#status = status.drop(['unknown'], axis=1)
#status


# In[303]:


df_trans = pd.concat([df_trans, status], axis=1)


# In[304]:


df_trans = df_trans.drop(['marital'], axis=1)


# In[305]:


df_trans.head()
df_trans.tail()


# ##Question 9 & 10 - Replace the unknown values in 'education with the highest frequency of the column'

# In[306]:


df_trans['education'].value_counts()


# In[307]:


df_trans['education'].replace('unknown', 'university.degree', inplace=True)


# In[308]:


df_trans['education'].value_counts()


# In[309]:


educ_cleanup = {"education": {"illiterate":0, "basic.4y":1, "basic.6y":2, "basic.9y":3, "high.school":4, "professional.course":5, "university.degree": 6}}
educ_cleanup


# In[310]:


educ = df_trans.replace(educ_cleanup)
educ

df_trans = educ


# ##Question 11 - default column

# In[311]:


df_trans['default'].value_counts()


# In[312]:


default_cleanup = {'default': {"no": 1, "yes":0}}


# In[313]:


default_dum = pd.get_dummies(df_trans['default'], prefix='default')
default_dum


# In[314]:


df_trans = pd.concat([df_trans, default_dum], axis=1)


# In[315]:


df_trans


# In[316]:


#df_trans.drop(['default_no', 'default_unknown'], axis=1)


# ##Question 12 - imput unknown values in 'housing' with the value of the highest frequency

# In[317]:


df_trans['housing'].value_counts()


# In[318]:


df_trans = df_trans.copy()
df_trans['housing'].replace('unknown', 'yes', inplace=True)


# In[319]:


df_trans['housing'].value_counts()


# ##Question 13 - Appropriate data transformation strategy for the column 'housing'. Apply on set

# In[320]:


housing_cleanup = {"housing": {"yes": 1, "no": 0}}


# In[321]:


df_trans = df_trans.replace(housing_cleanup)


# In[322]:


df_trans.head()


# ##Question 14 - impute known values in 'loan' with the value of the highest frequency

# In[323]:


df_trans['loan'].value_counts()


# In[324]:


df_trans = df_trans.copy()
df_trans['loan'].replace('unknown', 'no', inplace=True)


# In[325]:


df_trans['loan'].value_counts()


# In[326]:


##Question 15 - appropriate transformation for contact


# In[327]:


df_trans['contact'].value_counts()


# In[328]:


contact_cleanup = {"contact": {"cellular": 1, "telephone":0}}


# In[329]:


contactss = pd.get_dummies(df_trans['contact'], prefix="contact")
contactss


# In[330]:


df_trans = df_trans.replace(contact_cleanup)


# In[331]:


df_trans['contact']


# In[332]:


##Question 16 - same for pdays


# In[333]:


df_trans['pdays'].value_counts()


# In[334]:


pdays_cleanup = {"pdays": {"no contact": 1, "contact": 0}}


# In[335]:


df_trans.replace(pdays_cleanup)


# In[336]:


#Question 17 - Appropriate transformation for 'poutcome'


# In[337]:


df_trans['poutcome'].value_counts()


# ###LAB2 - Part B

# In[338]:


df6 = dfb2


# In[339]:


df6.head()


# In[340]:


df6.isnull().sum()


# In[341]:


df6['EmpStatusID'].value_counts()


# In[342]:


df6['EmpStatusID'].fillna(1.0, inplace=True)


# In[343]:


df6['EmpStatusID'].value_counts()


# In[344]:


df6['MaritalDesc'].value_counts()


# In[345]:


df6['MaritalDesc'].fillna('Single', inplace=True)


# In[346]:


df6['MaritalDesc'].value_counts()


# ##Question 24 - What would be the best feature transformation for the column CitizenDesc? Apply the appropriate transformation to the dataset
# - Map the values to 0 to 2
# - Replace it with a one-hot-vector of length 3
# - Replace it with a one-hot-vector of length 3, then drop off the columns
# - No feature transformation is needed for this column

# In[347]:


df6['CitizenDesc'].value_counts()


# In[348]:


df6['CitizenDesc'].fillna('US Citizen', inplace=True)


# In[349]:


df6['CitizenDesc'].value_counts()


# ##Question 21 - no missing values anymore. Goal is predict the salary of each employee based on available data in the dataset. Select columns that do not contribute meaninful information for this task and drop them from th dataset.

# In[350]:


df6.isna().sum()


# In[351]:


#Columns with little meaning to predict salary - engagementsurvey, empsatisfaction, dayslatelast30, absences, termd


# In[352]:


df6.head()


# In[353]:


df7 = df6.drop(['Employee_Name', 'EmpID'], axis=1)


# In[354]:


df7.head()


# ##Question 22 - What would be the best feature transoformation for the column EmpStatusID? Apply the proper transformation to your dataset
# - Map the values to 0 to 4
# - Replace it with a one-hot vector of length 5
# - Replace it with a one-hot vector of length 5, then drop off the columns
# - No feature transformation is needed for this column

# In[355]:


df7['EmpStatusID'].value_counts()


# In[356]:


empstat_clean = pd.get_dummies(df7['EmpStatusID'], prefix = 'EmpStatus')


# In[357]:


df7 = pd.concat([df7, empstat_clean], axis=1)


# In[358]:


df7 = df7.drop(['EmpStatusID'], axis=1)


# In[359]:


df7['MaritalDesc'].value_counts()


# In[360]:


mardesc = pd.get_dummies(df7['MaritalDesc'], prefix="maritaldesc")
mardesc


# In[361]:


df8 = pd.concat([df7, mardesc], axis=1)
df8


# In[362]:


df8.drop(['MaritalDesc'], axis=1)


# ##Question 23 - What would be the best feature transformation for the column MaritalDesc? Apply the proper transformation to the dataset
# - Mapt he values to 0 to 4
# - Replace it with a one-hot-vector of length 5
# - Replace it with a one-hot-vector of length 5, then drop off the columns
# - No feature transformation is needed for this column

# In[363]:


df8['CitizenDesc'].value_counts()


# In[364]:


citz = pd.get_dummies(df8['CitizenDesc'], prefix="CitizenDesc")
citz


# In[365]:


df8 = pd.concat([df8, citz], axis=1)
df8


# In[366]:


#df8.drop(['CitizenDesc'], axis=1)
df9 = df8.drop(['MaritalDesc', 'CitizenDesc'], axis=1)


# ##Question 25 - What would be the best feature transformation for the column PerformanceScore? Apply the proper transformation on the dataset.
# - Map the values using the dictionary: {"Exceeds": 3, "Fully Meets": 2, "Needs Improvement":1, "PIP":0}
# - Map the values using the dictionary: {"Fully Meets": 3, "Exceeds":2, "Needs Improvement":1, "PIP":0}
# - Replace it with a one-hot-vector of length 4
# - Replace it with a one-hot-vector of length 4, then drop off the columns
# - No feature transformation is needed for this column

# In[367]:


df9['PerformanceScore'].value_counts()


# In[368]:


perf = {"PerformanceScore": {"Exceeds": 3, "Fully Meets": 2, "Needs Improvement": 1, "PIP": 0}}
perf


# In[369]:


df10 = df9.replace(perf)
df10


# ##Question 26 - What would be the best feature transformation for the column Gender? Apply the proper transformation on the dataset
# - Map the values using the dictionary: {"F":1,"M":0}
# - Map the values using the dictionary: {"M":1,"F":0 }
# - Replace it with a one-hot vector of length 2
# - Replace it with a one-hot-vector of length 2, then drop on of the columns
# - NO feature transformation is needed for this column

# In[370]:


df10['Gender'].value_counts()


# In[371]:


genders = {"Gender": {"F":1, "M":0}}


# In[372]:


df11 = df10.replace(genders)


# In[373]:


##don't drop Gender


# In[374]:


##Question 27 - What is the number of columns in your dataset after applying all the above transformations?


# In[375]:


df11.info()


# In[376]:


df11 = df11.drop(['CitizenDesc_Eligible NonCitizen', 'CitizenDesc_Non-Citizen', 'CitizenDesc_US Citizen'], axis=1)


# In[377]:


df12 = pd.concat([df11, citz], axis=1)


# In[378]:


df12.info()


# ## Lab 3 - Make model

# In[ ]:





# In[379]:


#Question 1 - For this lab first, make the target column to be Salary and the rest of the columns as the feature set. This would be our target column for this lab and all the future labs. Then split the dataset into train and test with all default values of the train_test_split method and random_state = 0. 
#We will use the StandardScaler to scale this dataset. 
#What is the mean of the target column in the test dataset?
#Answer with 4 digits


# In[380]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df12.drop(['Salary'], axis=1)
y = df12['Salary']

X_train_sal, X_test_sal, y_train, y_test = train_test_split(X, y, random_state = 0)


# In[381]:


scaler = StandardScaler()


# In[382]:


X.info()


# In[383]:


y.describe()


# In[384]:


X_train = scaler.fit_transform(X_train_sal)
X_test = scaler.transform(X_test_sal)


# In[385]:


X_train


# In[386]:


y_train = y_train.to_numpy()


# In[387]:


y_test = y_test.to_numpy()


# In[388]:


y_test.mean()


# In[389]:


y_test.describe()


# In[390]:


##Question 2 - Train a linear regression model on the train dataset. What is the test score of the model? 
# Enter your answer with 4 digits after the decimal point


# In[391]:


from sklearn.linear_model import LinearRegression


# In[392]:


lreg = LinearRegression()
lreg.fit(X_train, y_train)
print(lreg.score(X_train, y_train))
print(lreg.score(X_test, y_test))


# In[393]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#metrics

pred_train_lr = lreg.predict(X_train)
print(np.sqrt(mean_squared_error(y_train, pred_train_lr)))
print(r2_score(y_train, pred_train_lr))

pred_test_lr = lreg.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred_test_lr)))
print(r2_score(y_test, pred_test_lr))


# In[394]:


print(lreg.coef_)
print(lreg.intercept_)


# In[395]:


##Question 3 - What is the train score of the LinearRegression model. 
# Enter your answer with 4 digits after the decimal point (See above for work )


# In[396]:


##Question 4 - Suppposed you are asked to present a linear model with 10 important features. 
# Which approach will you take to create such a model?
# Set random_state = 0


# In[397]:


#using ridge

from sklearn.linear_model import Ridge

train_score_list = []
test_score_list = []

for alpha in [10, 100, 1000]:
    ridge = Ridge(alpha, random_state=0)
    ridge.fit(X_train, y_train)
    train_score_list.append(ridge.score(X_train, y_train))
    test_score_list.append(ridge.score(X_test, y_test))


# In[398]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

x_range = [10, 100, 1000]

plt.subplots(figsize=(20,5))
plt.plot(x_range, train_score_list, c = 'g', label='Train Score')
plt.plot(x_range, test_score_list, c = 'b', label='Test Score')
plt.xscale('log')
plt.legend(loc=3)
plt.xlabel(r'$\alpha$')
plt.grid()


# In[399]:


print(train_score_list)
print(test_score_list)


# In[400]:


#another way
from sklearn.linear_model import RidgeCV
regr_cv = RidgeCV(alphas=[10,100,1000])

fit_cv = regr_cv.fit(X_train, y_train)
fit_cv.alpha_


# In[401]:


fit_cv.coef_


# In[402]:


##Do with Lasso


# In[403]:


from sklearn.linear_model import Lasso
x_range = [10, 100, 1000]
train_score_list2 = []
test_score_list2 = []

for alpha in x_range:
    lasso = Lasso(alpha)
    lasso.fit(X_train, y_train)
    train_score_list2.append(lasso.score(X_train, y_train))
    test_score_list2.append(lasso.score(X_test, y_test))


# In[404]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplots(figsize = (20,5))
plt.plot(x_range, train_score_list2, c='g', label = 'Train Score')
plt.plot(x_range, test_score_list2, c='b', label='Test Score')
plt.xscale('log')
plt.legend(loc=3)
plt.xlabel(r'$\alpha$')
plt.grid()


# In[405]:


print(train_score_list2)
print(test_score_list2)


# In[406]:


#another way
from sklearn.linear_model import Lasso, LassoCV


# In[407]:


lassocv = LassoCV(alphas=[10, 100, 1000])
fitlass = lassocv.fit(X_train, y_train)

fitlass.alpha_


# In[408]:


fitlass.coef_


# In[409]:


##Question 7 - PolynomialFeatures to transform the dataset with polynomial features with degree 2. 
# What is the number of columns in the resulting dataset?


# In[410]:


from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

# np.random.seed(0)
from sklearn.preprocessing import PolynomialFeatures

# X2 = df12.drop(['Salary'], axis=1)
# y2 = df12['Salary']

# X2_train_sal, X2_test_sal, y2_train, y2_test = train_test_split(X2, y2, random_state = 0)


# X_train2 = scaler.fit_transform(X2_train_sal)
# X_test2 = scaler.transform(X2_test_sal)

poly = PolynomialFeatures(degree = 2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# df_poly = DataFrame(X_train_poly)
# print(df_poly.describe())
# df_poly_test =            DataFrame(X_poly_test)
# print(df_poly_test.describe())

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

print(poly_reg.score(X_train_poly, y_train))
print(poly_reg.score(X_poly_test, y_test))

poly_test_score = poly_reg.score(X_poly_test, y_test)



# In[411]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df12.drop(['Salary'], axis=1)
y = df12['Salary']

X_train_org, X_test_org, y_train, y_test = train_test_split(X,y, random_state = 0)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)

poly = PolynomialFeatures(degree = 2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

print(poly_reg.score(X_train_poly, y_train))
print(poly_reg.score(X_test_poly, y_test))


# In[412]:


'{0:.10f}'.format(poly_test_score)


# In[413]:


X_train


# In[414]:


X_test


# In[415]:


y_train


# In[416]:


y_test


# In[417]:


from platform import python_version
print(python_version())


# ## Lab 4

# ### Part a

# In[418]:


#Question 1 - Consider 'y' as the target column of Part a data frame. Split this data frame into train and test with a test size of 25% of the whole data. Then use the MinMAxScaler to scale the train and the test. What is the best value of the hyperparameter n_neighbors? Build your model given the following range of values for the model
#hyperparameters. n_neighbors= [1, 2, 3, 4, 5], metric = 'manhattan'


# In[419]:


y = df_trans['y']
X = df_trans.drop(['y'], axis=1)


# In[420]:


y.head()


# In[421]:


X.head()


# In[422]:


X.info()


# In[423]:


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[424]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:





# In[ ]:




