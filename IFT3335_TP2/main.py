# create function pretraitement
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import CountVectorizer
# import sklearn.model_selection
# # https://www.d.umn.edu/~tpederse/data.html contains the data of interest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Open the file with read only permit
with open('interest.acl94.txt', "r") as f:
  # Use the read() method to read the entire file
#   file_contents = f.read()
    # Use the readlines() method to read line by line
    file_contents = f.readlines()
    

# Store the contents of the file in a corpus without $$ and \n
corpus = []
for i in range(len(file_contents)):
    if file_contents[i] != '$$\n':
        corpus.append(file_contents[i].replace('\n', ''))


#####################################################################

# exemple of element in corpus
# [ yields/NNS ] on/IN [ money-market/JJ mutual/JJ funds/NNS ]
# continued/VBD to/TO slide/VB ,/, amid/IN [ signs/NNS ] that/IN [
# portfolio/NN managers/NNS ] expect/VBP [ further/JJ declines/NNS ]
# in/IN [ interest_6/NN rates/NNS ] ./.

#####################################################################

# slice each element of the corpus with the delimiter ' '
# pour chercher plus facilement les mots avant/après le mot "interest"
corpus_sliced = []

for i in range(len(corpus)):
  tmp = corpus[i].split(' ')
  corpus_sliced.append(tmp)

# enlever element inutiles
for i in range(1):
    for elemt in list(corpus_sliced[i]):
        if '[' in elemt:
            corpus_sliced[i].remove(elemt)
        if ']' in elemt:
            corpus_sliced[i].remove(elemt)
        if '{' in elemt:
            corpus_sliced[i].remove(elemt)
        if '}' in elemt:
            corpus_sliced[i].remove(elemt)
        if '======================================' in elemt:
            corpus_sliced[i].remove(elemt)
        # if elemt is empty
        if elemt == '':
            corpus_sliced[i].remove(elemt)


# contiendra les mots avant/apres interest
# pour chaque occurence de interest
sac_mots = [[] for i in range(2369)]

# contiendra les categories de chaque mot
# pour chaque occurence de interest
categorie = [[] for i in range(2369)]

sense = ['1', '2', '3', '4', '5', '6']
senceList = []

def preparationData(contextWindowSize = 2): 
    count = 0

    for i in range (len(corpus_sliced)):
        
        for j in range(len(corpus_sliced[i])):

            # si le mot est interest tmp != None
            tmp = re.search("interest(s*)_", corpus_sliced[i][j])
            

            if tmp:
                if "1" in corpus_sliced[i][j]:
                    senceList.append(1)
                elif "2" in corpus_sliced[i][j]:
                    senceList.append(2)
                elif "3" in corpus_sliced[i][j]:
                    senceList.append(3)
                elif "4" in corpus_sliced[i][j]:
                    senceList.append(4)
                elif "5" in corpus_sliced[i][j]:
                    senceList.append(5)
                elif "6" in corpus_sliced[i][j]:
                    senceList.append(6)


                isTmp = [False] * contextWindowSize #isTmp est un tableau qui contient des booleens pour chaque mot de contexte

                # on voit les mots avant interest
                for k in range(j - 1, 0, -1):
                    currentWord = corpus_sliced[i][k]
                    
                    if currentWord != ']' and currentWord != '[' and currentWord != "{" and currentWord != "}":
                        if currentWord != "======================================":
                            
                            # si deja trouve 2 mots avant interest, on arrete de chercher plus loin
                            if len(sac_mots[count]) == contextWindowSize:
                                break

                            # contien le mot a word_group[0] et ca categorie a word_group[1]
                            word_group = currentWord.split('/')
                            sac_mots[count].append(word_group[0])

                            # pour savoir si categorie a deja trouvee la categorie des mots necessaire
                            # aka, la categorie des 2 mots avant interest
            
                            lenCat = len(categorie[count])

                                
                            
                            for l in range(contextWindowSize):
                                if lenCat == l:
                                    if len(word_group) == 2:
                                        categorie[count].append("C-" + str(l+1) + " = " + word_group[1])
                                    else:
                                        continue
                                    break
                            # si on a trouvee les 2 mots avant interest, 
                            # on arrete de chercher plus loin
                            if len(categorie[count]) == contextWindowSize :
                                break
                   


                for k in range(contextWindowSize):
                    if len(sac_mots[count]) == k:
                        isTmp[k] = True  
                    
                # servira a s'assurer que ca commence a C+1
                # si on enleve ca, il y aura des trucs bizzare
                # exemple C+-4 = ...
                lenWord = len(sac_mots[count])


                #Maintenant pour les mots après interest
                # print ("Here corpus_sliced[i]", len(corpus_sliced[i]))
                for k in range(j + 1 , len(corpus_sliced[i])):
                    
                    currentWord = corpus_sliced[i][k]
                    
                    if currentWord != ']' and currentWord != '[' and currentWord != "{" and currentWord != "}":
                        if currentWord != "======================================":
                            
                            if len(sac_mots[count]) == contextWindowSize*2:
                                break
                            word_group = currentWord.split('/')
                            sac_mots[count].append(word_group[0])
                            lenCat = len(categorie[count])

                            for l in range(contextWindowSize):
                                if (lenCat - lenWord) == l:
                                    if len(word_group) == 2:
                                        categorie[count].append("C+" + str(l+1) + " = " + word_group[1])
                                    else:
                                        continue
                                    break
                            if len(categorie[count]) == contextWindowSize*2 :
                                break

                count += 1


nbelement = 4
preparationData(nbelement)

# on met les elements de la categorie dans le bon ordre
for i in range(len(categorie)):
    partA = []
    partB = []
    for j in range(len(categorie[i])):
        tmp = categorie[i][j].split(' = ')
        if '-' in tmp [0]:
            partA.insert(0, categorie[i][j])
        elif '+' in tmp[0]:
            partB.append(categorie[i][j])

    while len(partA) != nbelement:
        partA.insert(0, "") 
    
    while len(partB) != nbelement:
        partB.append("")
    
    categorie[i] = partA + partB


# remove last element of sac_mots and categorie
# because it is empty
sac_mots.pop()
categorie.pop()

##############################################################
##############################################################

def NB(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    gnb = MultinomialNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("Classification report:", classification_report(y_test, y_pred))

def DT(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("Classification report:", classification_report(y_test, y_pred))

def RF(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("Classification report:", classification_report(y_test, y_pred))

def SVM(X_train, X_test, y_train, y_test):
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("Classification report:", classification_report(y_test, y_pred))

def MLP(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("Classification report:", classification_report(y_test, y_pred))

##############################################################
##############################################################

# transorm arrays to a pandas dataframe
categorie = pd.DataFrame(categorie)
sac_mots = pd.DataFrame(sac_mots)


one_hot_encoder = OneHotEncoder()

# Decommenter pour utiliser algos sur categorie 
X_one_hot = one_hot_encoder.fit_transform(categorie)
X_train, X_test, y_train, y_test = train_test_split(X_one_hot, senceList, test_size=0.2, random_state=42)

# Decommenter pour utiliser algos sur sac_mots
# X_one_hot = one_hot_encoder.fit_transform(sac_mots)
# X_train, X_test, y_train, y_test = train_test_split(X_one_hot, senceList, test_size=0.2, random_state=42)


# Decommenter pour algo voulu
# NB(X_train, X_test, y_train, y_test)
# DT(X_train, X_test, y_train, y_test)
# RF(X_train, X_test, y_train, y_test)
# SVM(X_train, X_test, y_train, y_test)
# MLP(X_train, X_test, y_train, y_test)

