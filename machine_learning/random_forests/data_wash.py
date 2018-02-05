
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data=pd.read_csv('data/train.csv')
data.head(5)


# In[2]:


data.columns


# In[3]:


len(data)


# In[4]:


data.isnull().sum()


# In[5]:


sns.countplot(x='Survived', data=data)


# In[6]:


data.drop(['PassengerId', 'Name'], axis=1, inplace=True)


# In[7]:


data.columns


# In[8]:


g = sns.heatmap(data[['Survived', 'SibSp', 'Parch', 'Age', 'Fare', 'Pclass']].corr(), cmap='RdYlGn', annot=True)


# In[9]:


Age0 = data[(data['Survived']==0) & (data['Age'].notnull())]['Age']
Age1 = data[(data['Survived']==1) & (data['Age'].notnull())]['Age']
g = sns.kdeplot(Age0, legend=True, shade=True, color='r', label='NotSurvived')
g = sns.kdeplot(Age1, legend=True, shade=True, color='b', label='Survived')


# In[10]:


g = sns.factorplot(x='Sex', y='Age', data=data, kind='box')
g = sns.factorplot(x='Pclass', y='Age', data=data, kind='box')


# In[11]:


index = list(data[data['Age'].isnull()].index)
Age_mean = np.mean(data[data['Age'].notnull()]['Age'])
copy_data = data.copy()

for i in index:
    filling_age = np.mean(copy_data[(copy_data['Pclass'] == copy_data.iloc[i]['Pclass'])
                                   & (copy_data['SibSp'] == copy_data.iloc[i]['SibSp'])
                                   & (copy_data['Parch'] == copy_data.iloc[i]['Parch'])
                                   ]['Age'])
    if not np.isnan(filling_age):
        data['Age'].iloc[i] = filling_age
    else:
        data['Age'].iloc[i] = Age_mean
        
g = sns.kdeplot(Age0, legend=True, shade=True, color='r', label='NotSurvived')
g = sns.kdeplot(Age1, legend=True, shade=True, color='b', label='Survived')


# In[12]:


data[data['Cabin'].notnull()]['Cabin'].head(10)


# In[13]:


data['Cabin'].fillna('U', inplace=True)
data['Cabin'] = data['Cabin'].map(lambda i: list(i)[0])

g = sns.factorplot(x='Cabin', y='Survived', data=data, ci=False, kind='bar', order=['A', 'B', 'C', 'D', 'E', 'F', 'T', 'U'])


# In[14]:


g = sns.countplot(x='Cabin', hue='Pclass', data=data, order=['A', 'B', 'C', 'D', 'E', 'F', 'T', 'U'])


# In[15]:


g = sns.kdeplot(data[data['Survived']==0]['Fare'], shade=True, label='NotSurvived', color='r')
g = sns.kdeplot(data[data['Survived']==1]['Fare'], shade=True, label='Survived', color='b')


# In[16]:


print('%.2f' %(data['Fare'].skew()))


# In[17]:


data['Fare'] = data['Fare'].map(lambda i: np.log(i) if i>0 else 0)
g = sns.distplot(data['Fare'])
print('Skew Coefficient: %.2f' %(data['Fare'].skew()))


# In[18]:


Ticket = []
import re

r = re.compile(r'\w*')

for i in data['Ticket']:
    sp = i.split(' ')
    if len(sp) == 1:
        Ticket.append('U')
    else:
        t = r.findall(sp[0])
        Ticket.append(''.join(t))
        
data['Ticket'] = Ticket
data = pd.get_dummies(data, columns=['Ticket'], prefix='T')


# In[19]:


data.columns


# In[20]:


data['Sex'].replace('male', 0, inplace=True)
data['Sex'].replace('female', 1, inplace=True)


# In[24]:


from collections import Counter

def outlier_detect(n, df, features):
    outlier_index = []
    
    for feature in features:
        Q1 = np.percentile(df[feature], 25)
        Q3 = np.percentile(df[feature], 75)
        
        IRQ = Q3 - Q1
        outlier_span = 1.5 * IRQ
        col = ((data[data[feature] > Q3 + outlier_span]) | 
              (data[data[feature] < Q1 - outlier_span])).index
        outlier_index.extend(col)
        
        print('%s: %f (Q3+1.5*IQR), %f (Q1-1.5*IQR)' %(feature, Q3 + outlier_span, Q1 - outlier_span))
        
    outlier_index = Counter(outlier_index)
    outlier = list(i for i, j in outlier_index.items() if j >= n)
    print('number of outliers: %d' % len(outlier))
    print(df[['Age', 'Parch', 'SibSp', 'Fare']].loc[outlier])
    
    return outlier

outlier = outlier_detect(3, data, ['Age', 'Parch', 'SibSp', 'Fare'])

