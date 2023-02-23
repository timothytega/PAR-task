#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


file='C:\\Users\\toshevire\\Downloads\\data_CaseStudy-analytics.csv'


# In[3]:


A=pd.read_csv(file)


# In[4]:


A.head()


# In[5]:


A['Year of Birth']=A['date_of_birth'].str[-4:]


# In[6]:


A.head()


# In[7]:


A.dtypes


# In[8]:


A['Year of Birth']=A['Year of Birth'].apply(pd.to_numeric)


# In[9]:


A['Age']=2021-A['Year of Birth']


# In[10]:


A.columns


# In[11]:


name=['reason','date','loan_id','principal ','balance ','repaid','approval_date','bank','state','Age','employment_status']


# In[12]:


B=A[name]


# In[13]:


B.head()


# In[14]:


c=B[['reason','Age']] #c[['Country','GDPperCapita']].value_counts(ascending=True)


# In[15]:


B.dtypes


# In[16]:


c.head()


# In[17]:


d=np.arange(c['Age'].min(),c['Age'].max(),10)


# In[18]:


d


# In[19]:


names=['Business']
e=c.isin(names)


# In[20]:


e.head()


# In[21]:


F=c.groupby('reason').plot(kind='bar')


# In[22]:


F


# In[23]:


list(F)


# In[24]:


c['reason'].count()


# In[25]:


F.columns


# In[26]:


'''g=sns.FacetGrid(c,row='reason',col='Age',margin_titles=True)
g.map(plt.hist,'Age',color='blue',bins=d,lw=0)'''


# In[ ]:





# In[ ]:





# In[27]:


'''plt.hist(F,
        alpha=0.5,
        color='blue')
plt.title('Age and Reason Plot')
plt.xlabel('reason')
plt.ylabel('Age')
plt.show()'''


# In[ ]:





# In[28]:


'''plt.bar(c['Age'],c['reason']['Business'],
        alpha=0.5,
        color='blue')
plt.title('Age and Reason Plot')
plt.xlabel('Age')
plt.ylabel('reason')
plt.show()'''


# In[29]:


#This is to plot the highest reason people take loan


# In[30]:


c.head()


# In[32]:


A=c['reason']


# In[33]:


B=[]
for i in (A):
    if i in B:
        pass
    else:
        B.append(i)
print(B)


# In[34]:


h=c['reason']


# In[35]:


h.head()

c.set_index('reason').plot.bar(rot=0,title='Purpose Plot',figsize=(12,8),fontsize=12)
# In[37]:


B


# In[38]:


c


# In[40]:


A


# In[52]:


D=[]
counter=0
for i in A:
    if i=='Business':
        counter+=1
D.append(counter)
print(D)


# In[46]:


c.head()


# In[61]:


d=c.groupby(['reason']).count()


# In[62]:


d.head()


# In[64]:


my_plot=d.plot(kind='bar',color='green')
my_plot.set_ylabel('Number of People')


# In[59]:





# In[ ]:




