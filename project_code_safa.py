#!/usr/bin/env python
# coding: utf-8

# In[2]:


import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


# In[3]:


mat = scipy.io.loadmat('mnist-original.mat') # lecture du fichier
data = np.transpose(mat['data'])
label = np.transpose(mat['label'])
X = [data[i].reshape(28,28) for i in range(len(data)) ] # on créé une nouvelle liste qui va contenir nos matrices 28x28


# In[4]:


def index_chiffre(n, y):  # elle nous renvoie un tableau contenant les indices représentants le chiffre n
    return [i for i in range(len(y)) if y[i]==n] 

def image_moyenne(n, X, y): # cette fonction renvoie la matrice moyenne (centroïde) pour le chiffre n
    index = index_chiffre(n, y) 
    sum_matrice = np.zeros((28,28)) # la matrice qui sera utilisée dans la somme
    for i in index:
        sum_matrice += X[i] # on somme toutes les matrices représentants le chiffre n
    return sum_matrice/len(index)

def distance(x,y,k): # fonction qui calcule la distance entre deux matrices 
    x = x.reshape(len(x)**2) # x est une matrice carré de "longueur" n donc sa représentation en vecteur sera de taille n*n
    y = y.reshape(len(y)**2)
    return np.linalg.norm(x-y,k)

def prediction(x,centroide, n, k):# on va chercher le centroïde le plus proche de la matrice X_n où n est l'indice de la matrice
    l=[distance(x[n], centroide[i],k) for i in range(10)]
    return (float(l.index(min(l))))

def calcul_pourcentage(n):# renvoie le pourcentage de reussite n représente l'ordre de la norme qu'on va choisir
    prediction_l = [ prediction(X_test,centroide_train, i, n) for i in range(len(X_test)) ]
    A =[i for i,j in zip(y_test,prediction_l) if i==j]
    return(len(A)*100/len(X_test))


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X,label,test_size=0.2,random_state=22)

centroide_train = [(image_moyenne(i, X_train, y_train)) for i in range(10)]# on calcul le centroïde pour chaque chiffre i entre 0 et 9 avec X_train et y_train


# In[6]:


fig, ax = plt.subplots(1,10,figsize=(20,2)) # affichage des centroïdes 
for i in range(10):
    ax[i].imshow(centroide_train[i],cmap='gray')
    ax[i].set_title("mean "+str(i))


# In[7]:


# On va calculer le pourcentage de réussite pour la norme 2
print("Le pourcentage de précision est {}:".format(calcul_pourcentage(2)))


# In[16]:


# On va maintenant utiliser plusieur norme L^p
pourcentage = [calcul_pourcentage(i) for i in range(0,8)]
plt.figure(figsize=(12,6)) 
plt.plot(pourcentage)
plt.show


# In[20]:


# on voit que le maximum est atteint pour la troisième valeur donc on a de meilleurs résultats pour une norme L^3
for i in range(0,7):
    print("Le pourcentage de précision pour la norme L^"+str(i)+" est égal à "+str(pourcentage[i]))


# #### Ici on utilise une méthode différente pour calcule de la distance entre deux matrices, d'abord je définis la norme matricielle (celle de Frobenius), ensuite je définis la distance entre deux matrices comme étant la norme de la différence, après ça j'utilise le même code que précédemment sauf que je remplace l'ancienne distance par la nouvelle, au final cette méthode est un peu plus rapide que la précédente (environ 4s).

# In[21]:


def norme_eucli_matrice(A): # calcul de la norme euclidienne de la matrice
    return np.trace(A@(A.transpose()))**(1/2)
def distance_norme(x,y):
    return norme_eucli_matrice(x-y)
def prediction_norme(x, n):
    l=[distance_norme(x[n], centroide_train[i]) for i in range(10)]
    return (float(l.index(min(l))))


# In[22]:


prediction_l = [ prediction_norme(X_test,i) for i in range(len(X_test)) ]
A =[i for i,j in zip(y_test,prediction_l) if i==j]
print("Le pourcentage de précision est {}:".format(len(A)*100/len(X_test)))


# ### La matrice de confusion:
# Elle permet de jauger directement les nombres les plus à même d'être confondus : si la prédiction était sans failles, l'on obtiendrait des nombres le long de la diagonale et des 0 partout ailleurs.

# In[23]:


cm = confusion_matrix(y_test,[prediction_norme(X_test,i) for i in range(len(y_test))])
cm


# In[24]:


plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de confusion')
plt.colorbar()
    
fmt = 'd'
thresh = cm.max()/2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i,format(cm[i,j], fmt), horizontalalignment="center",color="white" if cm[i,j]> thresh else "black")
        
plt.tight_layout()
plt.ylabel('Vraies valeurs')
plt.xlabel('Valeurs prédites')
plt.show()


# ### Matrice de confusion normalisée

# In[25]:


ncm = confusion_matrix(y_test,[prediction_norme(X_test,i) for i in range(len(y_test))], normalize = 'true')

plt.figure(figsize=(7,10))
plt.imshow(ncm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de confusion')
#ax = plt.axes()
#im = ax.imshow(np.linspace(0,100)
plt.colorbar(fraction= 0.045)
    
fmt = ".1%"
thresh = ncm.max()/2.
for i, j in itertools.product(range(ncm.shape[0]), range(ncm.shape[1])):
    plt.text(j,i,format(ncm[i,j], fmt), horizontalalignment="center",color="white" if ncm[i,j]> thresh else "black")
        
plt.tight_layout()
plt.ylabel('Vraies valeurs')
plt.xlabel('Valeurs prédites')
plt.show()


# ### Observations:
# On observe que le chiffre 1 est le mieux prédit par l'algorithme (96,5% de prédictions justes), tandis que le chiffre 5 est le moins bien prédit. Cette grille nous permet également de constater que les chiffres les plus confondus entre eux sont les 3 et 5, ainsi que les 4 et 9. Il peut être intéressant de s'attarder plus longuement sur les causes de ces confusions

# #### On vas essayer de faire un peu de data cleaning pour expliquer les résultats qu'on a eu 

# In[26]:


centroide_total = []
for i in range(10): # on calcul le centroïde pour chaque chiffre i entre 0 et 9 avec X et label 
    centroide_total.append(image_moyenne(i, X,label)) 
    
fig, ax = plt.subplots(1,10,figsize=(20,2)) # affichage des centroïdes 
for i in range(10):
    ax[i].imshow(centroide_total[i],cmap='gray')
    ax[i].set_title("mean "+str(i))


# In[27]:


def liste (x,y,n):# cette fonction Renvoie la liste des matrices de X qui Correspond au chiffre n
    index = index_chiffre(n, y) # j'utilise la fonction définie avant
    L=[]
    for i in range(len(index)):
        j= index[i]
        L.append(x[j])
    return L

def Liste_distance (x,y,n) :# cette fonction va nous renvoyer une liste des distances ente les matrices qui représente le chiffre n et le centroide total 
    Listedistance = []
    L = np.asarray( liste (x,y,n))

    for i in range(len(L)):
        Listedistance.append(distance(L[i],centroide_total[n],3))

    return Listedistance


# In[28]:


print("le nombre de data total qu'on a ="+str (len(X)))
for i in range(10):
    D = Liste_distance (X,label,i)
    print("le nombre de data total pour le chiffre {} est = {}".format(str(i),str(len(D))))
    print("la distance moyenne = {}".format(str(np.mean(D))))
    print("L'écart type est {}".format(np.std(D)))
    plt.figure(figsize=(12,6))
    plt.plot(range(len(D)),np.asarray(D))
    plt.show()


# In[29]:


def liste_propre (x,y,d):
    X_nv = []
    Y_nv = []
    X_fun= []
    Y_fun = []
    for i in range(10):
        
        D = Liste_distance (x,y,i)
        L = liste (x,y,i) 
        for j in range(len(D)):
            M = []
            if D[j] <= d:
                X_nv.append(L[j])
                M.append(i)
                Y_nv.append(M) 
            else :
                X_fun.append(L[j])
                M.append(i)
                Y_fun.append(M) 
    return [X_nv,Y_nv,X_fun,Y_fun] 


# In[30]:


d = 900
X_nv = liste_propre (X,label,d)[0]
Y_nv = np.asarray(liste_propre (X,label,d)[1])
print (len(X_nv))
for i in range(10):
    D_nv = Liste_distance (X_nv,Y_nv,i)
    D = Liste_distance (X,label,i)
    print("le nombre de data total Nettoyée pour le chiffre "+str(i)+ " qu'on a = "+str(len(D_nv)))
    print("le pourcentage de data en moins {}%".format((1-len(D_nv)/len(D))*100))
    print("la distance moyenne = "+str(np.mean(D_nv)))
    print(np.std(D_nv))
    plt.plot(range(len(D_nv)),np.asarray(D_nv))
    plt.show()


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X_nv,Y_nv,test_size=0.2,random_state=22)
centroide_train = []
for i in range(10): # on calcul le centroïde pour chaque chiffre i entre 0 et 9 avec X_train et y_train
   centroide_train.append(image_moyenne(i, X_train, y_train)) 
fig, ax = plt.subplots(1,10,figsize=(20,2)) # affichage des centroïdes 
for i in range(10):
   ax[i].imshow(centroide_train[i],cmap='gray')
   ax[i].set_title("mean "+str(i))


# In[32]:


print("Le pourcentage de précision est {}:".format(calcul_pourcentage(2)))


# In[33]:


# je veux voir si je lance sur les données "pas propres" le programme poura bien les prédire 
X_fun =liste_propre (X,label,d)[2]
Y_fun= np.asarray(liste_propre (X,label,d)[3])

k = 0 # elle va compte le nombre de fois où l'aglorithme a bien prédit
for i in range(len(X_fun)):
    if Y_fun[i]==prediction_norme(X_fun, i):
        k+=1
print("Le pourcentage de précision est "+'{:.2f}'.format(k*100/len(X_fun)))

centroide_fun = []
for i in range(10): # on calcul le centroïde pour chaque chiffre i entre 0 et 9 avec X_train et y_train
    centroide_fun.append(image_moyenne(i, X_fun, Y_fun)) 
fig, ax = plt.subplots(1,10,figsize=(20,2)) # affichage des centroïdes 
for i in range(10):
    ax[i].imshow(centroide_fun[i],cmap='gray')
    ax[i].set_title("mean "+str(i))


# # Attention ⚠️  La prochaine Ligne prends du temps pour s'executer 

# In[35]:


# je veux tracer mes pourcentage de precision a chaque fois que je modifie la distance d 
L_sale = []
L_prop = []

for d in range(650,1200,50):
        
    X_nv = liste_propre (X,label,d)[0]
    Y_nv = np.asarray(liste_propre (X,label,d)[1])
    X_fun =liste_propre (X,label,d)[2]
    Y_fun= np.asarray(liste_propre (X,label,d)[3])
        
    X_train, X_test, y_train, y_test = train_test_split(X_nv,Y_nv,test_size=0.2,train_size=0.8,random_state=22)
    
    centroide_train = [image_moyenne(i, X_train, y_train) for i in range(10) ]
        
    prediction_1 = [ prediction(X_fun,centroide_train, i, 2) for i in range(len(X_fun)) ]
    A =[i for i,j in zip(Y_fun,prediction_1) if i==j]
    L_sale.append((len(A)*100/len(X_fun)))
                  
    prediction_2 = [ prediction(X_test,centroide_train, i, 2) for i in range(len(X_test)) ]
    B =[i for i,j in zip(y_test,prediction_2) if i==j]
    L_prop.append((len(B)*100/len(X_test)))
                  
D=[i for i in range(650,1200,50)] 
plt.figure(figsize=(12,6))
plt.plot(D,L_prop)
plt.plot(D,L_sale)

plt.show()

