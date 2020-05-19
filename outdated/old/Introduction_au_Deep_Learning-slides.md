
# Formation au Deep-Learning

-----------------

## Sommaire
* Les matrices : quelques rappels
* Réseau de neurones
* Droites de régression
* Listes de matrices
* Calcul par CPU / GPU / TPU
* Calcul d’erreur

__<!-- -------------------------------------------------------------------------------------------------------------------------
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                                               Chapitre LES MATRICES : RAPPELS
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
-------------------------------------------------------------------------------------------------------------------------- -->__

## Les matrices : quelques rappels !

<img src="chapterHeader/calculMatrice.png" width="" height="350" align="" >

### L'addition de matrice

-----------------

### L'addition de matrice

<p>&nbsp;</p>

* Généralités :

  * L'addition des matrices est définie pour deux matrices de même type.  

* La somme de deux matrices de type (m, n), est obtenue en additionnant les éléments correspondants.

### L'addition de matrice

<p>&nbsp;</p>

#### Exemple avec :

<img src="formules/addition1.PNG" alt="addition1" width="" height="150" align="" />

### L'addition de matrice

<p>&nbsp;</p>

#### Étape 1 :

<p>&nbsp;</p>

<img src="formules/addition2.PNG" alt="addition2" width="" height="150" align="" />

### L'addition de matrice

<p>&nbsp;</p>

#### Étape 2 :

<p>&nbsp;</p>

<img src="formules/addition3.PNG" alt="addition3" width="" height="150" align="" />

### Le produit matriciel

-----------------

### Le produit matriciel
#### Généralités :

  * La multiplication des matrices n'est pas <u>commutative</u>, **c'est-à-dire que A&bull;B n'est pas égal à B&bull;A**.

À l’avenir, nous aurons seulement besoin de connaître le produit d’une matrice de type (1,<u>2</u>) et d’une matrice de type (<u>2</u>,2).

<p>&nbsp;</p>

<span style="color: #fb4141">**[!]** Produit matriciel &ne; multiplication de matrice</span>

### Le produit matriciel
#### Exemple avec :

<img src="formules/multi1.PNG" alt="multi1" width="" height="150" align="" />

### Le produit matriciel

<p>&nbsp;</p>

#### Étape 1 :

<p>&nbsp;</p>

<img src="formules/multi2.PNG" alt="multi2" width="" height="150" align="" />

### Le produit matriciel

<p>&nbsp;</p>

#### Étape 2 :

<p>&nbsp;</p>

<img src="formules/multi3.PNG" alt="multi3" width="" height="300" align="" />

### Le produit matriciel

<p>&nbsp;</p>

#### Étape 2 :

<p>&nbsp;</p>

<img src="formules/multi4.PNG" alt="multi4" width="" height="300" align="" />

### Le produit matriciel

<p>&nbsp;</p>

#### Étape 2 :

<p>&nbsp;</p>

<img src="formules/multi5.PNG" alt="multi5" width="" height="300" align="" />

### Le produit matriciel

<p>&nbsp;</p>

#### Étape 3 :

<p>&nbsp;</p>

<img src="formules/multi6.PNG" alt="multi6" width="" height="300" align="" />

### Le produit matriciel

<p>&nbsp;</p>

#### Étape 4 :

<p>&nbsp;</p>

<img src="formules/multi7.PNG" alt="multi7" width="" height="300" align="" />

### Le produit matriciel

<p>&nbsp;</p>

#### Étape 5 :

<p>&nbsp;</p>

<img src="formules/multi8.PNG" alt="multi8" width="" height="150" align="" />

### Le produit matriciel

<p>&nbsp;</p>

#### Étape 6 :

<p>&nbsp;</p>

<img src="formules/multi9.PNG" alt="multi9" width="" height="150" align="" />

__<!-- -------------------------------------------------------------------------------------------------------------------------
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                                               Chapitre RESEAU DE NEURONES
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
-------------------------------------------------------------------------------------------------------------------------- -->__

## Réseau de neurones

<img src="chapterHeader/neuralNetwork.png" width="" height="350" align="" >

### Le réseau _fully connected_

-----------------

### Le réseau _fully connected_

#### Définition :

  * Le neurone formel est conçu comme un automate doté d'une fonction de transfert qui transforme ses entrées en sortie selon des règles précises. Ces neurones sont par ailleurs associés en réseaux dont la topologie des connexions est variable : réseaux proactifs, récurrents, etc..

<img src="pictures/nerve-cell.jpg" width="" height="350" align="" >


### Le réseau _fully connected_

  * Un réseau de neurones est considéré « _fully connected_ » lorsque toute entrée est relié par une arête appelé « poids » et représenté par « _w<sub><font size=3>i,j</font></sub>_ » à l’intégralité des neurones présents dans les couches cachées.

<img src="pictures/neurons-network.jpg" width="" height="350" align="" >

### Le réseau _fully connected_

#### Représentation graphique :

<img src="neurones/graph_neurones.PNG" width="" height="400" align="" >

[ ! ] Les couches cachées sont appelé zone de « pré-activation » et l’ensemble des output zone d’ « activation ».

### Le réseau _fully connected_

#### Définition d'un biais :
  * Le biais est l'erreur provenant d’hypothèses erronées dans l'algorithme d'apprentissage. Un biais élevé peut être lié à un algorithme qui manque de relations pertinentes entre les données en entrée et les sorties prévues (sous-apprentissage).

### Le réseau _fully connected_

#### Définition d’une fonction d’activation :
  * La fonction d’activation (ou fonction de seuillage, ou encore fonction de transfert) sert à introduire une non-linéarité dans le fonctionnement du neurone.

  * Les fonctions de seuillage présentent généralement trois intervalles :
    * en dessous du seuil, le neurone est **non-actif**
    * aux alentours du seuil, une **phase de transition**
    * au-dessus du seuil, le neurone est **actif**

### Le réseau _fully connected_

#### Calcul de la valeur d'un neurone :

<img src="neurones/neuro1.PNG" width="" height="300" align="" >

Pour calculer la valeur d’un neurone, il faut effectuer la somme des connexions entrantes :
<span style="color: #fb4141">Neurone 1 = X<sub><font size=3>1</font></sub>W<sub>1,1</font></sub> + X<sub><font size=3>2</font></sub>W<sub><font size=3>2,1</font></sub> + B<sub><font size=3>1</font></sub></span>

### Le réseau _fully connected_

#### Calcul de la valeur d'un neurone :

<img src="neurones/neuro2.PNG" width="" height="300" align="" >

Pour calculer la valeur d’un neurone, il faut effectuer la somme des connexions entrantes :
Neurone 1 = X<sub><font size=3>1</font></sub>W<sub><font size=3>1,1</font></sub> + X<sub><font size=3>2</font></sub>W<sub><font size=3>2,1</font></sub> + B<sub><font size=3>1</font></sub>  
<span style="color: #fb4141">Neurone 2 = X<sub><font size=3>1</font></sub>W<sub><font size=3>1,2</font></sub> + X<sub><font size=3>2</font></sub>W<sub><font size=3>2,2</font></sub> + B<sub><font size=3>2</font></sub></span>

### Le réseau _fully connected_

#### Calcul de la valeur d'un neurone :

<img src="neurones/neuro3.PNG" width="" height="300" align="" >
<img src="neurones/Diapositive5.PNG" width="" height="300" align="" >

__<!-- -------------------------------------------------------------------------------------------------------------------------
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                                               Chapitre DROITES DE REGRESSION
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
-------------------------------------------------------------------------------------------------------------------------- -->__

## Droites de régression

<img src="chapterHeader/regression.gif" width="" height="350" align="" >

### Régression linéaire

-----------------

### Régression linéaire

#### Définition :

  * Désigne un modèle dans lesquels est la médiane conditionnelle de **« y »** sachant **« x »**.

  * Le modèle de régression linéaire est souvent estimé par la méthode des moindres carrés mais il existe aussi de nombreuses autres méthodes pour estimer ce modèle.

### Régression linéaire

#### Représentation graphique du réseau précèdent :

<img src=" neurones/linearite.png" width="" height="200" align="" >  
Voici une droite de régression linéaire. Par définition elle suit l'équation suivante :  
_<b>ax + b</b>_  
Afin de mieux comprendre et de l'appliquer à notre réseau de neurone, nous pouvons l'écrire de la façon suivante :  
_<b>xw + B</b>_  
<font size=3>_x : la valeur d'entrée_  
_w : le poids_  
_B : le biais_</font>

### Régression linéaire

#### Représentation graphique du réseau précèdent :

![Playground 2 neurones](pictures/playgrnd-reseau-simple.png)

Ce schéma représente ainsi la fonction d’activation

### La fonction d'activation

-----------------

### La fonction d'activation

#### Définition :

  * La fonction d’activation est une fonction mathématique appliquée à un signal en sortie d'un neurone artificiel. Soit dans notre cas à la droite de régression linéaire.

### La fonction d'activation

#### Graphiquement :

<img src="pictures/courbes.PNG" width="" height="400" align="" >

### La fonction d'activation

#### Cas pratique :

  * http://playground.tensorflow.org

<img src="pictures/exemple_relu.png" width="" height="350" align="" >

__<!-- -------------------------------------------------------------------------------------------------------------------------
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                                               Chapitre LISTES DE MATRICES
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
-------------------------------------------------------------------------------------------------------------------------- -->__

## Listes de matrices

<img src="chapterHeader/matrice4d.png" width="" height="350" align="" >

### Une matrice

-----------------

### Une matrice

#### Définition :

  * Une matrice est une liste de listes, une liste est une liste de vecteurs, un vecteur est une liste de chiffres.

<img src="pictures/matrix.jpg" width="" height="350" align="" >

### Une matrice

#### Prenons pour exemple, cette image :

<img src="pictures/dog.jpg" width="" height="350" align="" >

### Une matrice

#### Prenons pour exemple, cette image :

<TABLE BORDER=1>
  <TR>
    <TD><img src="pictures/dog.jpg" width="" height="150" align="" ></TD>
    <TD style="vertical-align: top;"><u>Détails image :</u>  
     **- Dimensions :**  
       1280 x 768  
     **- Caractéristiques :**  
       En couleurs (3 dimensions)</TD>
    <TD><img src="neurones/matrice.PNG" width="" height="150" align="" ></TD>
  </TR>
</TABLE>

__<!-- -------------------------------------------------------------------------------------------------------------------------
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                                               Chapitre CALCUL CPU/GPU/TPU
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
-------------------------------------------------------------------------------------------------------------------------- -->__

## Calculs par CPU / GPU / TPU

<img src="chapterHeader/CPU-GPU-TPU.jpg" width="" height="250" align="" >

### CPU versus GPU

-----------------

### CPU versus GPU

#### Le facteur nombre de cœurs :

![Comparaison CPU - GPU](pictures/cpu-vs-gpu.jpg)

### CPU versus GPU

  * Avantages :
    * Accélération via GPU des applications

![Schema CPU - GPU](pictures/accelleration-gpu.png)

### CPU versus GPU

#### Démonstration :

<img src="pictures/cpu.gif" width="" height="280" align="left" >
<img src="pictures/gpu.gif" width="" height="280" align="right" >

https://www.youtube.com/watch?v=-P28LKWTzrI

### TPU ? Késako ?

-----------------

### TPU ? Késako ?

#### Définition :

  * Le TPU (Tensor Processor Unit) est un module hardware dédié spécifiquement aux applications de Machine Learning

<img src="pictures/tpu.png" width="" height="350" align="" >

__<!-- -------------------------------------------------------------------------------------------------------------------------
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                                               Chapitre CALCUL D'ERREUR
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
-------------------------------------------------------------------------------------------------------------------------- -->__

## Calcul d'erreur

<img src="chapterHeader/errorLearning.jpg" width="" height="350" align="" >

### Notion d'erreur

-----------------

### Notion d'erreur

#### Définition :

  * A chaque itération, l'algorithme va calculer un indicateur de performance globale (l'erreur qu'il commet) en comparant la sortie attendue et la sortie prédite.

<img src="pictures/un-plus-un.jpg" width="" height="350" align="" >

### Le _batch_

-----------------

### Le _batch_

#### Définition :
Le _batch_ est un échantillon du panel d'images qui serviront à entraîner notre système.  
Dans le cas d'un entraînement de modèle, le batch est d'environ 10 à 20% du panel d'images, permettant lors de l’entraînement de venir tester l'efficacité du modèle.

### Le _gradient descent_

-----------------

### Le _gradient descent_

#### Définition :





### Le _learning rate_

-----------------

### Le _learning rate_

#### Définition :

  * Représente la taille du « pas » en avant effectué par le système pour atteindre le point d’apprentissage le plus efficient

<img src="pictures/LR1.png" width="" height="280" align="left" >
<img src="pictures/LR2.png" width="" height="280" align="right" >


### Le minimum local

-----------------

### Le minimum local

#### Définition :

  * Le minimum local est point dans une zone où le système établit qu’il obtient la meilleure précision mais ne l’est effectivement pas sur la courbe de précision de classification.

<img src="pictures/minim-local.jpg" width="" height="350" align="" >


### Notion de dérivée de sigmoïde

-----------------

### Notion de dérivée de sigmoïde

#### Définition :


### Notion de dérivée de sigmoïde

#### Définition (suite) :



### Le _vanishing gradients_

-----------------

<img src="pictures/vanish.png" width="" height="130" align="" >

### Le _vanishing gradients_

#### Définition :

  * Le vanishing gradients est une perte (ou fuite) de gradient, affectant les neurones plus profond et unités de saturations dans un réseau profond.

![TPU](pictures/long-network.png)

### L'_overfitting_

-----------------

### L'_overfitting_

#### Définition :

  * L’overfitting (ou surapprentissage) est une étape où le système est arrivé à reconnaitre quasi-seulement les images sur lesquelles il a été entrainé et une variation de lumière ou de milieu peut l’induire à ne pas reconnaitre l’objet.

![TPU](pictures/overfitting.jpg)

### Le _cross-validation_

#### Définition :

* La validation croisée (_cross-validation_) est une méthode d’estimation de fiabilité d'un modèle fondé sur une technique d'échantillonnage. Cela sert à comparer la pertinence d'un modèle par rapport à un autre.

__<!-- -------------------------------------------------------------------------------------------------------------------------
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                                               Chapitre LES CONVOLUTIONS
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
-------------------------------------------------------------------------------------------------------------------------- -->__

## Réseau neuronal à convolution

<img src="chapterHeader/convolution.gif" width="" height="350" align="" >


### Les convolutions

-----------------

### Les convolutions

#### Définition :

  * Les convolutions consistent en un empilage multicouche d'algorithme, dont le but est de pré-traiter de petites quantités d'informations.

### Les convolutions

#### Cas concret :

<img src="pictures/convolution.PNG" width="" height="350" align="" >

### Les blocs de construction

-----------------

### Les blocs de construction

#### Définition :

Une architecture de réseau de neurones convolutifs est formée par un empilement de couches de traitement (blocs de construction), il en existe 5 :
  * la couche de convolution (CONV)
  * la couche de pooling (POOL)
  * la couche de correction (ReLU)
  * la couche « entièrement connectée » (FC)
  * la couche de perte (LOSS)

### La couche de _pooling_

-----------------

### La couche de _pooling_

#### Définition :

  * Le _pooling_ (« mise en commun »), est une forme de sous-échantillonnage de l'image.
  * Le _pooling_ réduit la taille spatiale d'une image, réduisant ainsi la quantité de paramètres et de calcul dans le réseau. Il est donc fréquent d'insérer périodiquement une couche de _pooling_ entre deux couches convolutives successives d'une architecture de réseau de neurones convolutifs pour réduire le sur-apprentissage.

### La couche de _pooling_

[ ! ] Il existe plusieurs méthodes afin de réduire la taille spatiale d'une image concernant le _pooling_ :
  * _Average pooling_
  * _Max-pooling_
  * _L2-norm pooling_
  * _Stocastic pooling_

### Le _max-pooling_

-----------------

### Le _max-pooling_

#### Définition :

  * Le _max-pooling_ permet une réduction de la taille de la représentation en gardant seulement la plus grande valeur des tuiles dans le filtre.

<img src="pictures/maxPooling.png" width="" height="250" align="" >

### Le _max-pooling_

#### Détaillons :

<img src="pictures/maxPooling.png" width="" height="250" align="" >

  * Ici, nous avons un filtre de 2 x 2, avec un pas de 2  

<br/><font size=3>(il est possible d'avoir un filtre plus important, ou encore, de ne pas avoir de couche de pooling)</font>


__<!-- -------------------------------------------------------------------------------------------------------------------------
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                                               Chapitre CONTACT & RESSOURCES
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
-------------------------------------------------------------------------------------------------------------------------- -->__

## Contact

<p>&nbsp;</p>
<p>&nbsp;</p>

<img src="logo/mmtt.png" width="" height="40" align="" >

<p>Tom DESHAIRES - MomentTech SAS</p>
<p>@: <a href="mailto:tom.deshaires@mmtt.fr">
tom.deshaires@mmtt.fr</a></p>

<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>

<p><a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a></p>

### Ressources

* Wikipédia
* <a href="https://playground.tensorflow.org"> TensorFlow Playground </a>
* <a href="https://0x003e.github.io/TRS-deep-learning/"> EPITA </a>
* <a href="https://blogs.msdn.microsoft.com/big_data_france/2014/06/17/evaluer-un-modle-en-apprentissage-automatique/"> Microsoft </a>
* <a href="https://www.technologies-ebusiness.com/enjeux-et-tendances/le-deep-learning-pas-a-pas"> Le deep-learning pas à pas </a>
* <a href="https://cs231n.github.io/"> Convolutional Neural Networks for Visual Recognition </a>
* <a href="https://wingshore.wordpress.com/"> Wingshore </a>
* <a href="https://www.quora.com/How-does-the-ReLu-solve-the-vanishing-gradient-problem"> Quora </a>
