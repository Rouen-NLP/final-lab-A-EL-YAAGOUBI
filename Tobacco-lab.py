
# coding: utf-8

# # Classification des documents du procès des groupes américains du tabac
# 
# ## Contexte
# 
# Le gouvernement américain a attaqué en justice cinq grands groupes américains du tabac pour avoir amassé d'importants bénéfices en mentant sur les dangers de la cigarette. Le cigarettiers  se sont entendus dès 1953, pour "mener ensemble une vaste campagne de relations publiques afin de contrer les preuves de plus en plus manifestes d'un lien entre la consommation de tabac et des maladies graves".
# 
# Dans ce procès 14 millions de documents ont été collectés et numérisés. Afin de faciliter l'exploitation de ces documents par les avocats, vous êtes en charge de mettre en place une classification automatique des types de documents.

# ## Import et chargement des données

# In[1]:



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)


# In[2]:


paths = pd.read_csv('./tobacco_data/Tobacco3482.csv', sep = ",")
print('Nombre de documents : ', len(paths))
paths.head()


# ## Analyse des données

# In[3]:


labels = {}
X = []
Y = []
for index, row in paths.iterrows():
    path, label = row[0].replace('jpg', 'txt'), row[1]
    if label in labels:
        labels[label] += 1
    else:
        labels[label] = 1
    with open('./tobacco_data/{}'.format(path), 'r') as f:
        text = f.read().replace('\n', ' ')
    X.append(text)
    Y.append(label)
labels


# In[4]:


s = pd.Series(
    list(labels.values()),
    index = list(labels.keys())
)

plt.figure(figsize=(12,5))

plt.title('Histogramme des differentes classes', fontsize=20)
plt.ylabel('Frequence', fontsize=15)

my_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']

s.plot(
    fontsize=15,
    kind='bar', 
    color=my_colors
)
plt.savefig('./output/histogrammeLabels.png', bbox_inches='tight')
plt.show()


# - On peut remarquer que nos données sont divisées en dix classes
# - Les classes Email, Form, Letter et Memo sont plus représentées que les autres classes

# ## Classification

# ### Split des données

# In[5]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# ### BOW et TF-IDF

# In[6]:


vectorizer = CountVectorizer(max_features=2000, max_df=0.3)
vectorizer.fit(x_train)

x_train_counts = vectorizer.transform(x_train)
x_test_counts  = vectorizer.transform(x_test)

tf_transformer = TfidfTransformer().fit(x_train_counts)
x_train_tf     = tf_transformer.transform(x_train_counts)
x_test_tf      = tf_transformer.transform(x_test_counts)

print('Entrainement : ', x_train_counts.shape)
print('Validation   :  ', x_test_counts.shape)


# ### MultinomialNB

# In[7]:


model = MultinomialNB(alpha=0.1)
model.fit(x_train_tf, y_train)

predictions = model.predict(x_test_tf)

report = classification_report(y_test, predictions, target_names=np.unique(Y))
cfsM   = confusion_matrix(y_test, predictions)


# ### Rapport pour le model MultinomialNB

# In[8]:


print(report)


# ### Matrice de confusion

# In[9]:


# Definition d'une fonction utile pour l'affichage des matrices de confustion
def plotConfusionMatrix(cfsM, labels, path):
    """
    Affichage de la matrice de confusion plus normalisation.
    """
    
    plt.figure(figsize=(17,8.5))

    plt.subplot(121)
    plt.imshow(cfsM)
    plt.title('Matrice de confusion', fontsize=20)
    plt.colorbar(shrink=0.75)
    plt.yticks(np.arange(len(labels.keys())), labels.keys(), fontsize=15)
    plt.xticks(np.arange(len(labels.keys())), labels.keys(), fontsize=15, 
               rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            s = str(cfsM[i, j])[:4]
            plt.text(j, i, s=s, ha="center", va="center", color="w")

    plt.subplot(122)
    plt.imshow(cfsM / cfsM.astype(np.float).sum(axis=1))
    plt.title('Matrice de confusion normalisée', fontsize=20)
    plt.colorbar(shrink=0.75)
    plt.yticks(np.arange(len(labels.keys())), labels.keys(), fontsize=15)
    plt.xticks(np.arange(len(labels.keys())), labels.keys(), fontsize=15, 
               rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            s = str((cfsM / cfsM.astype(np.float).sum(axis=1))[i, j])[:4]
            plt.text(j, i, s=s, ha="center", va="center", color="w")

    plt.savefig(path, bbox_inches='tight')
    plt.show()


path = './output/matriceConfusionMultinomialNB.png'
plotConfusionMatrix(cfsM=cfsM, labels=labels, path=path)


# ### Multi Layer Perceptron

# In[10]:


model = MLPClassifier()
model.fit(x_train_tf, y_train)

predictions  = model.predict(x_test_tf)

report = classification_report(y_test, predictions)
cfsM   = confusion_matrix(y_test, predictions)


# ### Rapport pour le model MLP

# In[11]:


print(report)


# ### Matrice de confusion

# In[12]:


path = './output/matriceConfusionMLP.png'
plotConfusionMatrix(cfsM=cfsM, labels=labels, path=path)

