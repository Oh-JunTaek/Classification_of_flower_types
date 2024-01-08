#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[2]:


print(iris.keys())


# In[ ]:


print(iris['data'])  # 데이터 확인
print(iris['target'])  # 타겟 값 확인
print(iris['target_names'])  # 타겟 이름 확인
print(iris['feature_names'])  # 특성 이름 확인


# In[3]:


import pandas as pd
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['target'] = iris['target']


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=42)


# In[5]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[6]:


from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# 데이터프레임 생성
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['target'] = iris['target']

# 산점도 행렬을 이용한 시각화
sns.pairplot(df, hue='target')
plt.show()


# In[8]:


# 특성 'sepal length (cm)'에 대한 히스토그램
sns.histplot(df['sepal length (cm)'], bins=10, kde=True)
plt.show()


# In[ ]:





# In[ ]:




