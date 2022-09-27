#!/usr/bin/env python
# coding: utf-8

# # Superstore data
# ### data from:  https://www.kaggle.com/rohitsahoo/sales-forecasting
# ### Retail dataset of a US superstore for 4 years
# 
# ## 1. Look around the data

# In[118]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn")
sns.set(font_scale=1)

import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)


# In[119]:


sp = pd.read_csv('Superstore Data.csv')
sp.head(5)


# In[120]:


sp.columns.transpose()


# In[121]:


sp.info()


# In[122]:


sp.describe()


# In[123]:


sp.describe(include = object)


# In[124]:


print("The total number of data: ", sp.shape[0]*sp.shape[1])
print("The total number of null values: {} and it occupies {:.2f}% of the total ".format(sp.isnull().sum().sum(), (sp.isnull().sum().sum()*100)/(sp.shape[0]*sp.shape[1])))
print("Data period: {} ~ {} ".format(min(sp['Order Date']), max(sp['Order Date'])))
print("The total number of items sold: ", sp['Product ID'].nunique())


# ## 2.Questions
# ### *What kind of items(sub-category) sold the most?
# ### *Which city had the highest order volume?
# ### *The order amounts and sales amounts by 'segment' ,' region'?
# ### *Does the higher the discount rate, the lower the margin?
# ### *What kind of items was recorded the highest sales?
# ### *Can I show the sales amount on the map?(Visualization)

# ## 3. Data Preprocessing

# ### 3-1. preprocess the duplicated columns.

# In[125]:


sp.duplicated().sum()


# ### 3-1. Rename the columns.

# In[126]:


sp.columns


# In[127]:


sp.columns = ['row_id','order_id','order_date','ship_date','ship_mode','cust_id','cust_name','seg'
,'country','city','state','post_code','region','product_id','category','subcategory','product_name'
,'sales','quantity','discount','profit']


# In[128]:


sp.columns


# ### 3-3. Remove unnecessary columns.

# In[129]:


sp = sp.drop(['post_code'], axis=1)
sp.info()


# ## 4.EDA & Visualization

# In[130]:


#Top 10 items
sp['product_name'].value_counts().head(10)


# In[131]:


#violin plot >> check the 'sales' by category and 'discount'
f, ax = plt.subplots(1,2, figsize=(13,7))

sns.violinplot(x=sp['category'], y=sp['sales'],ax=ax[0])
sns.violinplot(x=sp['category'], y=sp['discount']*100, ax=ax[1])


# In[132]:


# Which items sold the most by sub-category? ratio?
#pie-graph
plt.figure(figsize=(12,10))
sp['subcategory'].value_counts().plot.pie(autopct= "%1.1f%%")

plt.show()


# ### 4-2.Check the top 10 cities

# In[133]:


Top_cities = sp['city'].value_counts().nlargest(20)
Top_cities


# In[134]:


# Visualize the order amounts by states
f, ax = plt.subplots(1,1, figsize=(18,8))

g = sns.countplot(sp['state'].sort_values(), ax=ax)
g.set_title('State', size=15)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_ylabel('')

plt.show()


# ### 4-3.The order amounts by 'segment' and 'region'

# In[135]:


# consumer, west
f, ax = plt.subplots(1,2, figsize=(15,6))

sns.countplot(x = sp['seg'], palette = 'magma', ax=ax[0])
ax[0].set_title('Segment')

sns.countplot(x = sp['region'], palette = 'magma', ax=ax[1])
ax[1].set_title('Region')
ax[1].set_ylabel('')

plt.show()


# ### 4-4. Margins depending on the discount rate.

# In[136]:


## sns.lineplot shows 'Confidence Interval' and regression line.
## The area means 'Confidence Interval'.
## estimator = None >> various x values
f, ax = plt.subplots(1,1, figsize=(18,10))

sns.lineplot('discount', 'profit', data = sp, color = 'r', ax=ax)
ax.set_xlabel("Discount(%)")
ax.set_ylabel("Profit($)")

plt.show()


# ### 4-5. Correlation with Heatmap.

# In[137]:


#Continuous Variables only
sp_corr = sp[['sales', 'quantity', 'discount', 'profit']].corr()
sp_corr


# In[138]:


f, ax = plt.subplots(1,1, figsize=(9,8))

sns.heatmap(data = sp_corr, annot=True, cmap='YlGnBu', ax=ax)
ax.set_title("Correlation", size=15)
#insights: discount higher >> margin(profit) lower


# ### 4-6.Top10 items by sales amount

# In[139]:


top_prd1 = sp.groupby(['product_name']).sum().sort_values('sales', ascending=False).head(10)
top_prd1


# In[140]:


top_prd1.reset_index(inplace=True)
top_prd1


# In[141]:


f, ax = plt.subplots(1,1, figsize=(9,8))

ax.pie(top_prd1['sales'], labels = top_prd1['product_name'], autopct="%1.1f%%",
      startangle=0)
ax.set_ylabel('')

center_circle = plt.Circle((0,0), 0.5, fc='white')
fig = plt.gcf()
fig.gca().add_artist(center_circle)

ax.axis('equal')
label = ax.annotate("Top 10 items\n by Sales amount", color='black', xy=(0,-0.07), fontsize=15, ha="center")

plt.show()


# ### 4-7. The total sales amouts by states on the map.

# In[142]:


sales = sp.groupby(['state']).sum().sort_values('sales', ascending = False)
sales.reset_index(level=0, inplace = True)
#sales = sales.sort_values('state', ascending = True)
sales


# In[143]:


# Technically, DC is not the state. Let's remove it!
dc = sales[sales['state'] == 'District of Columbia'].index
sales = sales.drop(dc)
sales


# In[144]:


# Add the state_code for the graph.
state = ['Alabama', 'Arizona' ,'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 
         'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
         'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana','Nebraska', 'Nevada', 'New Hampshire',
         'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
         'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
         'West Virginia', 'Wisconsin','Wyoming']
state_code = ['AL','AZ','AR','CA','CO','CT','DE','FL','GA','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
              'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
              'TX','UT','VT','VA','WA','WV','WI','WY']

# Merge the 'state_code' with 'state'
state_cd = pd.DataFrame(state, state_code)
state_cd
state_cd.reset_index(inplace = True)
state_cd.columns = ['state_cd', 'state']
state_cd


# In[145]:


# Arrange the dataset by 'state' in order to add 'state_cd' column.
sales = sales.sort_values('state', ascending = True)
sales.reset_index(inplace = True)
sales.insert(1, 'state_cd', state_cd['state_cd'])
sales


# In[146]:


sales.drop('index',1,inplace = True)
sales


# In[1]:


# Use plotly to show a responsive graph.
import plotly.express as px

fig = px.choropleth(locations = sales['state_cd'], locationmode="USA-states", color = sales['sales'], scope="usa",
                   color_continuous_scale='peach', title = "The total sales amounts by the states")

fig.show()


# ## 5.Review
# 
# ### The total number of data:  209874
# ### The total number of null values: 0 and it occupies 0.00% of the total 
# ### Data period:    2014-01-03 ~ 2017-12-30 
# ### The total number of items sold:  1862 
# 
# ### Q: Which items sold the most by sub-category? ratio?
# #### A: Binders, 15.2%
# 
# ### Q: Which city had the highest order volume?
# #### A: 1. CA, 2. NY, 3. TX 
# 
# ### Q: The order amounts by 'segment' and 'region'?
# #### A: The consumer seg was recorded the highest and the West had the most order volume. 
# 
# ### Q: Does the higher the discout rate, the lower the profit?
# #### A: When the discount rate was 50%, the profit was the shortest.
# 
# ### Q: What kind of items was recorded the highest sales?
# #### A: The sales of Canon imageCLASS 2200 Advanced Copier was the highest.
# 
# ### Q: Can I show sales on the map?
# #### A: Yes, I made it!
