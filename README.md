
# Food Desert And Health
Blake Palder, John Jenkins, Jennifer Kregor, Matthew Benner

Food Desert Theory: There is a direct relationship between a population’s access to fresh food and health. The purpose of this analysis explores whether the data supports the Food Desert Theory. 

Definition of Food Desert as Defined by the USDA

“The USDA defines what's considered a food desert and which areas will be helped by this initiative:  To qualify as a “low-access community,” at least 500 people and/or at least 33 percent of the census tracts population must reside more than one mile from a supermarket or large grocery store (for rural census tracts, the distance is more than 10 miles).”



### Questions

1. Does a change in access to grocery stores show correlation to change in health over time?
2. Are there outliers to this theory?
3. Are there other factors that impact health beyond access such as socio economics or region or culture?



```python
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import seaborn as sns
```


```python
#Import CSV File
desert = pd.read_csv("Food Desert_Data_jj.csv")
food_desert = desert.groupby('State').mean()
```


```python
#DataFrame 
food_desert.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Grocery 2009</th>
      <th>Convenience 2009</th>
      <th>Fast Food Restaurants 2009</th>
      <th>Adult Diabetes % 2009</th>
      <th>Adult Obsesity % 2009</th>
      <th>Median Household Income 2009</th>
      <th>Limited Access Population% 2009</th>
      <th>Limited Access Population 2009</th>
      <th>Population 2009</th>
      <th>Grocery 2014</th>
      <th>...</th>
      <th>Population 2014</th>
      <th>FF per capita 2009</th>
      <th>FF per capita 2014</th>
      <th>% change FF Per Capita</th>
      <th>Grocery per capita 2009</th>
      <th>Grocery per capita 2014</th>
      <th>% change in grocery per capita</th>
      <th>% change in Access</th>
      <th>% Change in Obesity</th>
      <th>% change in income</th>
    </tr>
    <tr>
      <th>State</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AK</th>
      <td>9.400000</td>
      <td>6.000000</td>
      <td>7.300000</td>
      <td>7.880000</td>
      <td>31.760000</td>
      <td>50652.700000</td>
      <td>36.800000</td>
      <td>4066.000000</td>
      <td>9371.700000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>9553.100000</td>
      <td>0.001179</td>
      <td>0.000469</td>
      <td>-18.909000</td>
      <td>0.001536</td>
      <td>0.001250</td>
      <td>-10.104000</td>
      <td>13.310000</td>
      <td>-3.868000</td>
      <td>14.522000</td>
    </tr>
    <tr>
      <th>AL</th>
      <td>11.590909</td>
      <td>46.954545</td>
      <td>49.606061</td>
      <td>13.906061</td>
      <td>35.119697</td>
      <td>36158.409091</td>
      <td>16.909091</td>
      <td>15479.530303</td>
      <td>71153.681818</td>
      <td>11.303030</td>
      <td>...</td>
      <td>72071.696970</td>
      <td>0.000570</td>
      <td>0.000584</td>
      <td>1.949394</td>
      <td>0.000209</td>
      <td>0.000186</td>
      <td>-6.459545</td>
      <td>29.911364</td>
      <td>3.277424</td>
      <td>9.576970</td>
    </tr>
    <tr>
      <th>AR</th>
      <td>10.112676</td>
      <td>32.042254</td>
      <td>41.661972</td>
      <td>12.401408</td>
      <td>33.661972</td>
      <td>35439.126761</td>
      <td>19.056338</td>
      <td>13986.169014</td>
      <td>57926.732394</td>
      <td>9.197183</td>
      <td>...</td>
      <td>58807.563380</td>
      <td>0.000568</td>
      <td>0.000562</td>
      <td>2.233521</td>
      <td>0.000221</td>
      <td>0.000181</td>
      <td>-12.675493</td>
      <td>41.280000</td>
      <td>7.626197</td>
      <td>10.319155</td>
    </tr>
    <tr>
      <th>AZ</th>
      <td>64.692308</td>
      <td>144.153846</td>
      <td>309.615385</td>
      <td>9.984615</td>
      <td>28.069231</td>
      <td>41814.615385</td>
      <td>34.307692</td>
      <td>92083.769231</td>
      <td>486469.846154</td>
      <td>62.307692</td>
      <td>...</td>
      <td>511791.769231</td>
      <td>0.000517</td>
      <td>0.000512</td>
      <td>-1.579231</td>
      <td>0.000141</td>
      <td>0.000136</td>
      <td>-1.438462</td>
      <td>7.690769</td>
      <td>3.651538</td>
      <td>8.396154</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>75.590909</td>
      <td>90.750000</td>
      <td>241.022727</td>
      <td>8.279545</td>
      <td>25.579545</td>
      <td>48118.727273</td>
      <td>21.340909</td>
      <td>51613.681818</td>
      <td>362603.431818</td>
      <td>79.113636</td>
      <td>...</td>
      <td>376846.227273</td>
      <td>0.000623</td>
      <td>0.000629</td>
      <td>1.688182</td>
      <td>0.000278</td>
      <td>0.000281</td>
      <td>2.268864</td>
      <td>-3.377500</td>
      <td>-1.526136</td>
      <td>9.622727</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
#Set Index
food_desert.reset_index(inplace=True)

```


```python
#Access Versus Health 2009- Correlation not found
sns.set()

# Household income against Limited Access
g = sns.lmplot(x="Limited Access Population% 2009", y="Adult Obsesity % 2009", truncate=True, size=7, data=food_desert)
plt.show()

plt.savefig("Access_v_Health_2009.png")
```


![png](output_8_0.png)



```python
#Access Versus Health 2014- Correlation not found
sns.set()

# Household income against Limited Access
g = sns.lmplot(x="Limited Access % 2014", y="Adult Obesity 2014", truncate=True, size=7, data=food_desert)
plt.show()


plt.savefig("Access_v_Health_2014.png")
```


    <matplotlib.figure.Figure at 0x111a40f60>



![png](output_9_1.png)


### 1. Does the data support the Food Desert Theory, where we expect to see a direct relationship between the percentage of population’s access to fresh food and their relative health?

We found that the change in data between 2009 and 2014 showed correlation between access to adequate food and populations health. The results are that as limited access increased health became worse while inversley when limited access decreased health became better. The data thus supports the food desert theory. 


```python
#Change Access Versus Change Obesity- Correlation Found
sns.set()

# Household income against Limited Access
w = sns.lmplot(x="% change in Access", y="% Change in Obesity", truncate=True, size=7, data=food_desert)
plt.show()


plt.savefig("Access_v_Obesity_chnge.png")
```


    <matplotlib.figure.Figure at 0x116e67f28>



![png](output_12_1.png)



```python
#Change in Obesity Versus Change in Access- Correlation Found
import plotly.plotly as py
import plotly.graph_objs as go


trace1 = go.Bar(
    x=food_desert['State'],
    y=food_desert['% change in Access'],
    name='Access'
)
trace2 = go.Bar(
    x=food_desert['State'],
    y=food_desert['% Change in Obesity'],
    name='Obesity'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
plt.savefig("accessvshealth.png")
py.iplot(fig, filename='grouped-bar')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~blakezara/2.embed" height="525px" width="100%"></iframe>




```python
#Change Fast Food and Change Grocery and Change Health
import plotly.plotly as py
import plotly.graph_objs as go


trace1 = go.Bar(
    x=food_desert['State'],
    y=food_desert['% change in grocery per capita'],
    name='Grocery'
)
trace2 = go.Bar(
    x=food_desert['State'],
    y=food_desert['% Change in Obesity'],
    name='Obesity'
)
trace3 = go.Bar(
    x=food_desert['State'],
    y=food_desert['% change FF Per Capita'],
    name='FF'
)

data = [trace1, trace2, trace3]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
plt.savefig("accessvshealth.png")
py.iplot(fig, filename='grouped-bar')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~blakezara/2.embed" height="525px" width="100%"></iframe>



### 2. Are there outliers to this theory? 

We found that there were outliers that did not follow the the expected correlation over the years that we looked at. A majority of these outliers were where limited acces decreased but health became worse therfore illustrating that limited access to food is not the only contributor to increased negative health. 


```python
#Identify Outliers
outlier_df = food_desert[['State', '% Change in Obesity', '% change in Access']]

```


```python
#Sort through data that has opposite signs to find outliers and create new column
outlier_df['Outlier'] = outlier_df.apply((lambda x: (x[2]> 0 and x[1]>0) or (x[2]< 0 and x[1]<0)  ), axis = 1)
outlier_df.head()

#Export Outliers List
outlier_df.to_csv("Outliers.csv")
```

    /Users/blakezpalder/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    



```python
#Pull Outliers
outlier_list=outlier_df[outlier_df['Outlier'] == False]
outlier_list
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>% Change in Obesity</th>
      <th>% change in Access</th>
      <th>Outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AK</td>
      <td>-3.868000</td>
      <td>13.310000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CT</td>
      <td>9.686667</td>
      <td>-1.623333</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DE</td>
      <td>2.875000</td>
      <td>-16.720000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>LA</td>
      <td>5.148409</td>
      <td>-0.480227</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MA</td>
      <td>0.770714</td>
      <td>-13.382143</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ME</td>
      <td>0.627500</td>
      <td>-5.960625</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MN</td>
      <td>3.579737</td>
      <td>-4.340658</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ND</td>
      <td>3.401961</td>
      <td>-10.590784</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>NE</td>
      <td>2.488539</td>
      <td>-7.723483</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NH</td>
      <td>8.821000</td>
      <td>-6.348000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>RI</td>
      <td>3.892000</td>
      <td>-15.734000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>41</th>
      <td>TN</td>
      <td>2.359043</td>
      <td>-0.652766</td>
      <td>False</td>
    </tr>
    <tr>
      <th>42</th>
      <td>TX</td>
      <td>-1.059705</td>
      <td>17.321646</td>
      <td>False</td>
    </tr>
    <tr>
      <th>45</th>
      <td>VT</td>
      <td>3.164615</td>
      <td>-3.643846</td>
      <td>False</td>
    </tr>
    <tr>
      <th>49</th>
      <td>WY</td>
      <td>3.260000</td>
      <td>-0.268571</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Index Numbers
outlier_list.index
#Pull outliers data from original DataFrame
outlier_data = food_desert.iloc[[0, 6, 7, 17, 18, 20, 22, 27, 28, 29, 38, 41, 42, 45, 49],:]
outlier_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Grocery 2009</th>
      <th>Convenience 2009</th>
      <th>Fast Food Restaurants 2009</th>
      <th>Adult Diabetes % 2009</th>
      <th>Adult Obsesity % 2009</th>
      <th>Median Household Income 2009</th>
      <th>Limited Access Population% 2009</th>
      <th>Limited Access Population 2009</th>
      <th>Population 2009</th>
      <th>...</th>
      <th>Population 2014</th>
      <th>FF per capita 2009</th>
      <th>FF per capita 2014</th>
      <th>% change FF Per Capita</th>
      <th>Grocery per capita 2009</th>
      <th>Grocery per capita 2014</th>
      <th>% change in grocery per capita</th>
      <th>% change in Access</th>
      <th>% Change in Obesity</th>
      <th>% change in income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AK</td>
      <td>9.400000</td>
      <td>6.000000</td>
      <td>7.300000</td>
      <td>7.880000</td>
      <td>31.760000</td>
      <td>50652.700000</td>
      <td>36.800000</td>
      <td>4066.000000</td>
      <td>9371.700000</td>
      <td>...</td>
      <td>9553.100000</td>
      <td>0.001179</td>
      <td>0.000469</td>
      <td>-18.909000</td>
      <td>0.001536</td>
      <td>0.001250</td>
      <td>-10.104000</td>
      <td>13.310000</td>
      <td>-3.868000</td>
      <td>14.522000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CT</td>
      <td>84.833333</td>
      <td>125.666667</td>
      <td>220.333333</td>
      <td>7.483333</td>
      <td>23.100000</td>
      <td>67162.500000</td>
      <td>26.833333</td>
      <td>116899.833333</td>
      <td>406260.833333</td>
      <td>...</td>
      <td>409714.333333</td>
      <td>0.000491</td>
      <td>0.000583</td>
      <td>20.293333</td>
      <td>0.000190</td>
      <td>0.000201</td>
      <td>4.110000</td>
      <td>-1.623333</td>
      <td>9.686667</td>
      <td>12.111667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DE</td>
      <td>32.500000</td>
      <td>77.500000</td>
      <td>112.000000</td>
      <td>11.750000</td>
      <td>31.050000</td>
      <td>50036.000000</td>
      <td>18.000000</td>
      <td>32606.500000</td>
      <td>179727.500000</td>
      <td>...</td>
      <td>191338.500000</td>
      <td>0.000613</td>
      <td>0.000590</td>
      <td>-2.575000</td>
      <td>0.000174</td>
      <td>0.000188</td>
      <td>15.355000</td>
      <td>-16.720000</td>
      <td>2.875000</td>
      <td>9.960000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>LA</td>
      <td>16.477273</td>
      <td>48.750000</td>
      <td>58.022727</td>
      <td>12.052273</td>
      <td>33.740909</td>
      <td>39219.681818</td>
      <td>23.750000</td>
      <td>23823.522727</td>
      <td>85195.795455</td>
      <td>...</td>
      <td>87089.159091</td>
      <td>0.000517</td>
      <td>0.000544</td>
      <td>5.187045</td>
      <td>0.000225</td>
      <td>0.000195</td>
      <td>-10.515682</td>
      <td>-0.480227</td>
      <td>5.148409</td>
      <td>9.418636</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MA</td>
      <td>61.571429</td>
      <td>115.785714</td>
      <td>203.714286</td>
      <td>9.150000</td>
      <td>24.621429</td>
      <td>56539.428571</td>
      <td>23.785714</td>
      <td>74951.071429</td>
      <td>282394.357143</td>
      <td>...</td>
      <td>290014.500000</td>
      <td>0.000850</td>
      <td>0.000895</td>
      <td>5.889286</td>
      <td>0.000273</td>
      <td>0.000281</td>
      <td>5.662143</td>
      <td>-13.382143</td>
      <td>0.770714</td>
      <td>12.657143</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
# Create List Omitting Outliers
standard =food_desert.drop(food_desert.index[[0, 6, 7, 17, 18, 20, 22, 27, 28, 29, 38, 41, 42, 45, 49]])
standard.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Grocery 2009</th>
      <th>Convenience 2009</th>
      <th>Fast Food Restaurants 2009</th>
      <th>Adult Diabetes % 2009</th>
      <th>Adult Obsesity % 2009</th>
      <th>Median Household Income 2009</th>
      <th>Limited Access Population% 2009</th>
      <th>Limited Access Population 2009</th>
      <th>Population 2009</th>
      <th>...</th>
      <th>Population 2014</th>
      <th>FF per capita 2009</th>
      <th>FF per capita 2014</th>
      <th>% change FF Per Capita</th>
      <th>Grocery per capita 2009</th>
      <th>Grocery per capita 2014</th>
      <th>% change in grocery per capita</th>
      <th>% change in Access</th>
      <th>% Change in Obesity</th>
      <th>% change in income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>11.590909</td>
      <td>46.954545</td>
      <td>49.606061</td>
      <td>13.906061</td>
      <td>35.119697</td>
      <td>36158.409091</td>
      <td>16.909091</td>
      <td>15479.530303</td>
      <td>71153.681818</td>
      <td>...</td>
      <td>72071.696970</td>
      <td>0.000570</td>
      <td>0.000584</td>
      <td>1.949394</td>
      <td>0.000209</td>
      <td>0.000186</td>
      <td>-6.459545</td>
      <td>29.911364</td>
      <td>3.277424</td>
      <td>9.576970</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AR</td>
      <td>10.112676</td>
      <td>32.042254</td>
      <td>41.661972</td>
      <td>12.401408</td>
      <td>33.661972</td>
      <td>35439.126761</td>
      <td>19.056338</td>
      <td>13986.169014</td>
      <td>57926.732394</td>
      <td>...</td>
      <td>58807.563380</td>
      <td>0.000568</td>
      <td>0.000562</td>
      <td>2.233521</td>
      <td>0.000221</td>
      <td>0.000181</td>
      <td>-12.675493</td>
      <td>41.280000</td>
      <td>7.626197</td>
      <td>10.319155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AZ</td>
      <td>64.692308</td>
      <td>144.153846</td>
      <td>309.615385</td>
      <td>9.984615</td>
      <td>28.069231</td>
      <td>41814.615385</td>
      <td>34.307692</td>
      <td>92083.769231</td>
      <td>486469.846154</td>
      <td>...</td>
      <td>511791.769231</td>
      <td>0.000517</td>
      <td>0.000512</td>
      <td>-1.579231</td>
      <td>0.000141</td>
      <td>0.000136</td>
      <td>-1.438462</td>
      <td>7.690769</td>
      <td>3.651538</td>
      <td>8.396154</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>75.590909</td>
      <td>90.750000</td>
      <td>241.022727</td>
      <td>8.279545</td>
      <td>25.579545</td>
      <td>48118.727273</td>
      <td>21.340909</td>
      <td>51613.681818</td>
      <td>362603.431818</td>
      <td>...</td>
      <td>376846.227273</td>
      <td>0.000623</td>
      <td>0.000629</td>
      <td>1.688182</td>
      <td>0.000278</td>
      <td>0.000281</td>
      <td>2.268864</td>
      <td>-3.377500</td>
      <td>-1.526136</td>
      <td>9.622727</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CO</td>
      <td>13.857143</td>
      <td>33.553571</td>
      <td>63.928571</td>
      <td>6.876786</td>
      <td>21.951786</td>
      <td>47755.446429</td>
      <td>28.482143</td>
      <td>18153.000000</td>
      <td>87508.214286</td>
      <td>...</td>
      <td>92247.821429</td>
      <td>0.000596</td>
      <td>0.000587</td>
      <td>1.272321</td>
      <td>0.000334</td>
      <td>0.000356</td>
      <td>5.220357</td>
      <td>0.107857</td>
      <td>2.454821</td>
      <td>10.479286</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
sns.set()


# Plot tip as a function of toal bill across days
g = sns.lmplot(x="% Change in Obesity", y="% change in Access", hue="Outlier",
               truncate=True, size=5, data=outlier_df)
plt.savefig("outvsstand2.png")
plt.show()
```


    <matplotlib.figure.Figure at 0x1170a3f28>



![png](output_22_1.png)



```python
#Create Map for Outliers
import plotly.plotly as py
import pandas as pd

for col in outlier_list.columns:
    outlier_list[col] = outlier_list[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = outlier_list['State'],
        z = outlier_list["% Change in Obesity"].astype(float),
        locationmode = 'USA-states',
       # text = texas_desert['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Obesity Percent Change")
        ) ]

layout = dict(
        title = 'Outliers Obesity Percent Change',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
plt.savefig("outliersmap.png")
py.iplot( fig, filename='d3-cloropleth-map' )
```

    /Users/blakezpalder/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    





<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~blakezara/6.embed" height="525px" width="100%"></iframe>



### 3. Are there other factors that impact health beyond access such as socio economics or region or culture?

The theory argues that low income populations are more impacted by change in grocery accessibility. Therfore the outlier states had higher median income and were less impacted by the change in grocery than those states that showed the correlation we expected. This leads us to belive that socio economics are a important factor to health and we beleive that cultural and regional norms also must have some impact to health as well. 


```python
#Full data obesity vs income comparrison
import plotly.plotly as py
import plotly.graph_objs as go


trace1 = go.Bar(
    x=['Outlier'],
    y=outlier_data['% change in income'].mean(),
    name='Outlier'
)
trace2 = go.Bar(
    x=['Standard'],
    y=standard['% change in income'].mean(),
    name='Standard'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
plt.savefig("MedianIncome.png")
py.iplot(fig, filename='grouped-bar')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~blakezara/2.embed" height="525px" width="100%"></iframe>




```python
#Outlier Change Acces Versus Change Health 
sns.set()

# Household income against Limited Access
w = sns.lmplot(x="% change in Access", y="% Change in Obesity", truncate=True, size=7, data=outlier_data)
plt.show()

plt.savefig("Access_v_Obesity_chnge.png")
```


    <matplotlib.figure.Figure at 0x1a1bbf1588>



![png](output_27_1.png)



```python
#States Change in Obesity Versus Change in Fast Food
x7_axis = standard['% Change in Obesity']

y7_axis = standard['% change FF Per Capita']

#low access vs. diabetes

plt.scatter(x7_axis, y7_axis, marker="o", facecolors="blue", edgecolors="black", alpha=0.5)
#plt.title("")
plt.xlabel("Change in Obesity")
plt.ylabel("Change in Fast Food")

#plot grid
plt.grid(True, color='w', linestyle='-', linewidth=.2)
plt.gca().patch.set_facecolor('0.85')

plt.savefig("standard_obesity_v_ff.png")
plt.show()
```


![png](output_28_0.png)



```python
#Outlier States Change in Obesity Versus Change in Fast Food
x7_axis = outlier_data['% Change in Obesity']

y7_axis = outlier_data['% change FF Per Capita']

#low access vs. diabetes

plt.scatter(x7_axis, y7_axis, marker="o", facecolors="blue", edgecolors="black", alpha=0.5)
#plt.title("Diabetes vs Limited Access")
plt.xlabel("Change in Obesity")
plt.ylabel("Change in Fast Food")

#plot grid
plt.grid(True, color='w', linestyle='-', linewidth=.2)
plt.gca().patch.set_facecolor('0.85')

plt.savefig("outlier_obesity_v_ff.png")
plt.show()
```


![png](output_29_0.png)



```python
#An exploration into outlier versus standard states differences
full_desert = pd.read_csv("Food Desert_Data_jj.csv")
GA_data = full_desert[full_desert['State']=='GA']
MA_data = full_desert[full_desert['State']=='MA']
MT_data = full_desert[full_desert['State']=='MT']
```


```python
#Select States Obesity vs Income 2009
fig, ax1 = plt.subplots()
x1 = MT_data['Adult Obsesity % 2009']
y1 = MT_data['Median Household Income 2009']
x2 = GA_data['Adult Obsesity % 2009']
y2 = GA_data['Median Household Income 2009']
x3 = MA_data['Adult Obsesity % 2009']
y3 = MA_data['Median Household Income 2009']

ax1.scatter(x1,y1, c= "blue", alpha=0.4, label = "MT")
ax1.scatter(x2,y2, c= "lime", alpha=0.4, label = "GA")
ax1.scatter(x3,y3, c= "red", alpha=0.4, label = "MA")
ax1.grid(True)

plt.title("Adult Obesity % vs. Median Household Income 2009")
plt.xlabel("Adult Obesity %")
plt.ylabel("Median Household Income")
plt.legend()
plt.savefig("select_state_health_versus_income_09.png")
plt.show()
```


![png](output_31_0.png)



```python
#% change in Obesity vs % change in Income
fig, ax1 = plt.subplots()
x1 = MT_data['% change in income']
y1 = MT_data['% Change in Obesity']
x2 = GA_data['% change in income']
y2 = GA_data['% Change in Obesity']
x3 = MA_data['% change in income']
y3 = MA_data['% Change in Obesity']

ax1.scatter(x1,y1, c= "blue", alpha=0.4, label = "MT")
ax1.scatter(x2,y2, c= "lime", alpha=0.4, label = "GA")
ax1.scatter(x3,y3, c= "red", alpha=0.4, label = "MA")
ax1.grid(True)

plt.title("% change in Income vs. % Change in Obesity")
plt.xlabel("% Change in Income ")
plt.ylabel("% Change in Obesity")
plt.xlim(-50,50)
plt.ylim(-50,50)
plt.legend()
plt.savefig("select_state_health_versus_income.png")
plt.show()
```


![png](output_32_0.png)

