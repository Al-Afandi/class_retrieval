# Class retrieval for a detected adversarial attack

In this repository You can find our code which impelments a class retrieval, descird in the paper:

*Class retrieval for a detected adversarial attack*
Jala Al-afandi and András Horváth

Submitted to:
The journal of Applied science

### Prerequisites-Installing
To run our code You need to install Python, Pytorch, Torchvision, Torchattacks and cifar10_models (https://github.com/huyvnphan/PyTorch_CIFAR10).

### Running our code
The repository has  four scripts, three scripts to implement our class retrieval over three different datasets (mnist, cifar10 and imagenet) and the last script investigate the effect of number of classes parameter.

### Problem illustration
An example image displaying a two dimensional UMAP projection of the MNIST digits in the sklearn package with an additional 100 attacked samples which originally belonged to class 7 and were transformed to class 3 with the PGD algorithm. The classes are marked and circled on the figure where 'A' denoting the adversarially attacked samples.

<img src="https://github.com/Al-Afandi/class_retrieval/blob/main/umap_attacks_marked.png" width="500" height="300">

## Authors
**Jalal Al-afandi 
Andras Horvath** 
