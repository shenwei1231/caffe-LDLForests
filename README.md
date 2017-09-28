<img align="center" src="http://wei-shen.weebly.com/uploads/2/3/8/2/23825939/illustration_orig.png" width="900">

#### Label Distribution Learning Forests

Label distribution learning (LDL) is a general learning framework, which assigns to an instance a distribution over a set of labels rather than a single label or multiple labels. Current LDL methods have either restricted assumptions on the expression form of the label distribution or limitations in representation learning, e.g., to learn deep features in an end-to-end manner. This paper presents label distribution learning forests (LDLFs) - a novel label distribution learning algorithm based on differentiable decision trees, which have several advantages: 1) Decision trees have the potential to model any general form of label distributions by a mixture of leaf node predictions. 2) The learning of differentiable decision trees can be combined with representation learning. We define a distribution-based loss function for a forest, enabling all the trees to be learned jointly, and show that an update function for leaf node predictions, which guarantees a strict decrease of the loss function, can be derived by variational bounding. The effectiveness of the proposed LDLFs is verified on several LDL tasks and a computer vision application, showing significant improvements to the state-of-the-art LDL methods. For detailed algorithm and experiment results please see our NIPS 2017 [paper](https://arxiv.org/abs/1702.06086).

#### Demo: 
A quick demo of using the proposed LDLFs for age estimation on the Morph datasetcan be found at "examples/morph/Alex_net". In this example, we adopt AlexNet model and simply replace the softmax loss layer with the proposed LDLFs.
To run the demo, do the following steps:
1. Download the Morph dataset. The Morph dataset is not free availabel, but you can request for it from [here](https://ebill.uncw.edu/C20231_ustores/web/store_main.jsp?STOREID=4).
2. Run morph2lmdb.py to convert the Morph dataset to LMDB format.
3. Download pre-trained AlexNet model [bvlc_alexnet.caffemodel](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) and put it into examples/morph.
4. Change the data source paths to yours in train_net.pt and test_net.pt
5. Run examples/morph/Alex_net/train_agenet.sh

#### Transplant:
If you have different Caffe version than this repo and would like to try out the proposed LDLFs layers, you can transplant the following code to your repo.

(util) 
 - include/caffe/util/sampling.hpp
 - src/caffe/util/sampling.cpp
 - include/caffe/util/neural_decision_util_functions.hpp
 - src/caffe/util/neural_decision_util_functions.cpp

(training) 
 - include/caffe/layers/neural_decision_distr_learning_forest_loss_layer.hpp 
 - src/caffe/layers/neural_decision_distr_learning_forest_loss_layer.cpp
 - src/caffe/layers/neural_decision_distr_learning_forest_loss_layer.cu

(testing) 
 - include/caffe/layers/neural_decision_forest_layer.hpp 
 - src/caffe/layers/neural_decision_forest_layer.cpp
 - src/caffe/layers/neural_decision_forest_layer.cu

Tips: Make sure that the names of the NeuralDecisionDLForestWithLoss layer and the NeuralDecisionForest layer in the train_net and test_net prototxts are the same, so that the learned leaf nodes can be loaded in the testing stage.

Please cite the following paper if it helps your research:

    @inproceedings{shen2017ldlforests,
      author = {Wei Shen and Kai Zhao and Yilu Guo and Alan Yuille},
      booktitle = {Proc. NIPS},
      title = {Label Distribution Learning Forests},
      year = {2017}
    }

If you have any issues using the code please email us at shenwei1231@gmail.com, zhaok1206@gmail.com