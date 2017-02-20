# Attribute-Prediction

Attribute Predicton using Hilbert-Schmidt Independence Criterion (Cross-Covariance Operator) on aPascal/aYahoo dataset.

"Hilbert-Schmist Norm of the Cross-Covariance operator" is proposed as an independence criterion in reproducing kernel Hilbert spaces (RKHSs). The criterion is used to measure the dependence between the two multivariate random variables. Details can be found here.
http://www.wikicoursenote.com/wiki/Measuring_Statistical_Dependence_with_Hilbert-Schmidt_Norm

The project has three variations. 

1- apascal_train.lua: Fine-tune the Overfeat model. The pre-trained Overfeat model can be found from here.
https://github.com/jhjin/overfeat-torch

2- train_aPY_varII: This variation contains two branches after the second last fully connected in the overfeat model. First branch has binary cross entropy loss for attribute prediction and second branch has cross-entropy loss for classifiction.

3- train_aPY_varIII: The fine-tuning of variation-II model don't disentangle the attribute-oriented and category-centric factors. There is a risk that the category factor would leak into the hidden units of attribute-oriented factors and make it category-discriminative as well. The variation III uses HSIC to penalize the mutual information between attribute-oriented factors and category factors.

4- apascal_svm_train: The code extracts the 4096D features from pre-trained overfeat model and runs svm on top of it.

#Requirements
Torch7 (http://github.com/torch/torch7)
torch-svm (https://github.com/koraykv/torch-svm)
matio-ffi.torch (https://github.com/soumith/matio-ffi.torch)
Overfeat Model (https://github.com/jhjin/overfeat-torch)

