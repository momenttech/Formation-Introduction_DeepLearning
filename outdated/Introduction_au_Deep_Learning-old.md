
# Formation au Deep-Learning

## Sommaire
* Réseau de neurones
* Droites de régression
* Listes de matrices
* Calcul d’erreur

## Les matrices : quelques rappels !

### L'addition de matrice

#### Généralités :

  * L'addition des matrices est définie pour deux matrices de même type.
  * La somme de deux matrices de type (m, n), est obtenue en additionnant les éléments correspondants.


#### Exemple avec :

\[
\begin{Bmatrix}
   A_{1}  \\
   A_{2}
\end{Bmatrix}
\] + \[
\begin{Bmatrix}
   B_{1}  \\
   B_{2}
\end{Bmatrix}
\]

### Le produit matriciel

#### Généralités :

  * La multiplication des matrices n'est pas commutative, c'est-à-dire que AB n'est pas égal à BA.

À l’avenir, nous aurons seulement besoin de connaitre le produit d’une matrice de type (1,2) et d’une matrice de type (2,2).

#### Exemple avec :

\[
\begin{Bmatrix}
   A_{1}  \\
   A_{2}
\end{Bmatrix}
\] + \[
\begin{Bmatrix}
   B_{1} && B_{1,2}  \\
   B_{2,1} && B_{2,2}
\end{Bmatrix}
\]

## Réseau de neurones

### Le réseau _fully connected_

#### Définition :

  * Le neurone formel est conçu comme un automate doté d'une fonction de transfert qui transforme ses entrées en sortie selon des règles précises. Ces neurones sont par ailleurs associés en réseaux dont la topologie des connexions est variable : réseaux proactifs, récurrents, etc..

<img src="pictures/nerve-cell.jpg" width="500" height="350" align="" >

  * Un réseau de neurones est considéré « fully connected » lorsque toute entrée est relié par une arête appelé « poids » et représenté par «  » à l’intégralité des neurones présents dans les couches cachées.

<img src="pictures/neurons-network.jpg" width="500" height="350" align="" >

#### Représentation graphique :

![Schema]()

[ ! ] Les couches cachées sont appelé zone de « pré-activation » et l’ensemble des output zone d’ « activation ».

#### Définition d'un biais :
  * Le biais est l'erreur provenant d’hypothèses erronées dans l'algorithme d'apprentissage. Un biais élevé peut être lié à un algorithme qui manque de relations pertinentes entre les données en entrée et les sorties prévues (sous-apprentissage).

#### Définition d’une fonction d’activation :
  * La fonction d’activation (ou fonction de seuillage, ou encore fonction de transfert) sert à introduire une non-linéarité dans le fonctionnement du neurone.
  * Les fonctions de seuillage présentent généralement trois intervalles :
    * en dessous du seuil, le neurone est non-actif
    * aux alentours du seuil, une phase de transition
    * au-dessus du seuil, le neurone est actif

#### Calcul de la valeur d'un neurone :



## Droites de régression

### Régression linéaire

#### Définition :

  * Désigne un modèle dans lesquels est la médiane conditionnelle de « y » sachant « x ».
  * Le modèle de régression linéaire est souvent estimé par la méthode des moindres carrés mais il existe aussi de nombreuses autres méthodes pour estimer ce modèle.

#### Représentation graphique du réseau précèdent :

![Playground 2 neurones](pictures/playgrnd-reseau-simple.jpg)

Ce schéma représente ainsi la fonction d’activation

### La fonction d'activation

#### Définition :

  * La fonction d’activation est une fonction mathématique appliquée à un signal en sortie d'un neurone artificiel. Soit dans notre cas à la droite de régression linéaire.

#### Graphiquement :


#### Cas pratique :

  * playground.tensorflow.org

![Aperçu du playground tensorflow](pictures/playgrnd-tf.jpg)

## Listes de matrices

### Matrice : définition

#### Définition :

  * Les matrices sont des tableaux de nombres qui servent à interpréter en termes calculatoires et donc opérationnels les résultats théoriques de l'algèbre

![Matrice](pictures/matrix.jpg)

#### Prenons pour exemple, cette image :

![Image exemple](pictures/dog.jpg)

## Calculs par CPU / GPU / TPU

### CPU versus GPU

#### Le facteur nombre de cœurs :

![Comparaison CPU - GPU](pictures/cpu-vs-gpu.jpg)

  * Avantages :
    * Accélération via GPU des applications

![Schema CPU - GPU](pictures/accelleration-gpu.jpg)

#### Démonstration :

![Compa CPU - GPU](pictures/cpu.gif)
![Compa CPU - GPU](pictures/gpu.gif)

https://www.youtube.com/watch?v=-P28LKWTzrI

### TPU ? Késako ?

#### Définition :

  * Le TPU (Tensor Processor Unit) est un module hardware dédié spécifiquement aux applications de Machine Learning

![TPU](pictures/tpu.jpg)

## Calcul d'erreur

### Notion d'erreur

#### Définition :

  * A chaque itération, l'algorithme va calculer un indicateur de performance globale (l'erreur qu'il commet) en comparant la sortie attendue et la sortie prédite.

![1+1](pictures/un-plus-un.jpg)

### Le batch

#### Définition :


### Le minimum local

#### Définition :

  * Le minimum local est point dans une zone où le système établit qu’il ne peut semble pense avoir obtenu la meilleure précision mais ne l’est effectivement pas sur la courbe de précision de classification.

![Minimum local](pictures/minim-local.jpg)

### Le _learning rate_

#### Définition :

  * Représente la taille du « pas » en avant effectuer par le système pour atteindre le point d’apprentissage le plus efficient

![TPU](pictures/LR1.jpg)
![TPU](pictures/LR2.jpg)

### Le _vanishing gradients_

![TPU](pictures/vanish.png)

#### Définition :

  * Le vanishing gradients est une perte (ou fuite) de gradient, affectant les neurones plus profond et unités de saturations dans un réseau profond.

![TPU](pictures/long-network.jpg)

### L'_overfitting_

#### Définition :

  * L’overfitting (ou surapprentissage) est une étape où le système est arrivé à reconnaitre quasi-seulement les images sur lesquelles il a été entrainé et une variation de lumière ou de milieu peut l’induire à ne pas reconnaitre l’objet.

![TPU](pictures/overfitting.jpg)
