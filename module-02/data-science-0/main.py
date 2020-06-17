#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday


# ## Dataframe com uma resumo quantitativo dos dados do Dataframe black_friday 

# In[ ]:


df_info = pd.DataFrame({'names': black_friday.columns, 'types': black_friday.dtypes,
                             'NA #': black_friday.isna().sum(),
                             'NA %': (black_friday.isna().sum() / black_friday.shape[0]) * 100})


# In[84]:


df_info


# uma cópia do Dataframe black_friday para possíveis alterações

# In[ ]:


bf_copy = pd.DataFrame.copy(black_friday)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[7]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return (black_friday.shape)

q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[8]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday[ (black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35') ].shape[0]

q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[9]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].unique().shape[0]

q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[10]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return df_info['types'].unique().shape[0]

q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[11]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return df_info['NA %'].max() / 100

q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[12]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return df_info['NA #'].max()

q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[13]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return int(black_friday['Product_Category_3'].value_counts().idxmax())

q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[7]:


def q8():
    # Retorne aqui o resultado da questão 8.
    xmin = black_friday.Purchase.min()
    xmax = black_friday.Purchase.max()

    bf_copy['Purchase_normal'] = (black_friday.Purchase - xmin) / (xmax - xmin) 
    
    return round(bf_copy['Purchase_normal'].mean(), 3)

q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[8]:


def q9():
    # Retorne aqui o resultado da questão 9.
    P_mean = black_friday.Purchase.mean()
    P_std = black_friday.Purchase.std()
    
    bf_copy['Purchase_pad'] = (black_friday.Purchase - P_mean) / P_std
    
    return len(bf_copy[abs(bf_copy.Purchase_pad) <= 1])

q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[86]:


def q10():
    # Retorne aqui o resultado da questão 10.
    pc2 = list(bf_copy.Product_Category_2.isnull())
    pc3 = list(bf_copy.Product_Category_2.isnull())

    bf_copy['compare'] = [True if (pc2[x] == True) and (pc3[x] == True) else False for x in range(len(pc2)) ]
    
    return df_info['NA #']['Product_Category_2'] == bf_copy[bf_copy.compare == True].shape[0]


q10()


# In[ ]:




