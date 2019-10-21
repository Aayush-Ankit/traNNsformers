# traNNsformers
TraNNsformer is an integrated MATLAB framework for training MLP and CNN networks using structured pruning approach to enable efficient mapping on memristive crossbar based neuromorphic architecures.

## System requirements

| Requirement | Version                    |
| ----------- | -------------------------- |
| MATLAB      | > 2016a                    |

## Network Topologies
TraNNsformer framework is implemented using MATLAB to transform neural networks of two topologies: 

1. Multi-layer perceptrons (MLP) models are specified and transformed using codes in [traNNsformers/NN](NN)

2. Convolutional Neural networks (CNN) models are specified and transformed using codes in:

    1. if GPU support is not available and for small sized networks [traNNsformers/CNN](CNN)
    2. if GPU support is available and for large sized networks [traNNsformers/CNN_wGPU](CNN_wGPU_AlexNet)

## Citation
Please cite the following paper if you find this work useful:

* A. Ankit, T. Ibrayev, A. Sengupta, K. Roy. [**TraNNsformer: Clustered Pruning on Crossbar-based Architectures for Energy Efficient Neural Networks**](https://ieeexplore.ieee.org/document/8865639). IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2019.

## Authors

Aayush Ankit, Timur Ibrayev
