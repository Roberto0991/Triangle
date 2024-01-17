#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# In[3]:


d = {'origin': [2008+i for i in range(10)],
     '1': [508700, 20100, 342200, 573600, 117700, 156300, 55800, 142600, 314300, 206800],
     '2': [836800, 433600, 906000, 1159200, 962400, 644700, 404200, 682400, 539700, np.nan],
     '3': [1094400, 543800, 1391100, 1581800, 1587100, 1172300, 1095500, 1314800, np.nan, np.nan],
     '4': [1189900, 1073300, 1623400, 2133200, 2226600, 1296400, 1237900, np.nan, np.nan, np.nan],
     '5': [1358700, 1380400, 1875700, 2352100, 2603300, 1589600, np.nan, np.nan, np.nan, np.nan],
     '6': [1619300, 1569300, 2224100, 2610400, 2619400, np.nan, np.nan, np.nan, np.nan, np.nan],
     '7': [1801200, 1559000, 2287600, 2709700, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
     '8': [1865700, 1622800, 2347600, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
     '9': [1866200, 1675700, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
     '10': [1886100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}


# In[20]:


triangle = pd.DataFrame(d)
triangle.set_index(triangle['origin'], inplace = True)
del triangle['origin']
triangle


# In[21]:


triangle.T.plot(figsize=(12,6))
plt.show()


# In[22]:


factors = []
for col in triangle.columns[:-1]:
    print(triangle[str(int(col)+1)])
    print(triangle[col][:-int(col)])
    factors.append(triangle[str(int(col)+1)].sum() / triangle[col][:-int(col)].sum())
factors = np.array(factors)
factors


# In[24]:


dev_period = np.array([(i+1) for i in range(9)])
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(dev_period,factors)
ax.set_xlabel('dev_period')
ax.set_ylabel('Factor')
plt.show


# In[28]:


fig,ax = plt.subplots(figsize=(12,6))
ax.set_title('Plot of Calcaulated Factors')
ax.set_xlabel('dev_period')
ax.set_ylabel('Factor')
sns.regplot(x=dev_period, y=np.log(factors-1))
#sns.regplot(x=dev_period, y=factors)
plt.show()


# In[29]:


tail_model = LinearRegression().fit(dev_period.reshape(-1,1), np.log(factors -1))


# In[30]:


tail_model.intercept_


# In[31]:


tail_model.coef_


# In[34]:


tail = np.array([(i+10) for i in range(101)])
tail = np.exp(tail_model.intercept_ + tail_model.coef_ * tail) + 1
tail_factor = tail.prod()
tail_factor


# In[35]:


for i, col in enumerate(triangle.columns[1:]):
    for j in range(i+1):
        triangle[col].at[2017-j]=factors[i] * triangle[str(int(col)-1)].at[2017-j]
        
triangle


# In[36]:


triangle['ultimate'] = triangle['10'] * tail_factor
triangle


# In[38]:


triangle['IBNR'] = triangle['ultimate'].subtract(triangle['10'])
triangle


# In[ ]:


triangle.sum()


# In[ ]:




