# traNNsformers
TraNNsformer is an integrated MATLAB framework for training MLP and CNN networks using structured pruning approach to enable efficient mapping on memristive crossbar based neuromorphic architecures.

## System requirements

| Requirement | Version                    |
| ----------- | -------------------------- |
| MATLAB      | > 2016a                    |

## Network Topologies
TraNNsformer framework is implemented using MATLAB to transform neural networks of two topologies: 

1. Multi-layer perceptrons (MLP) are transformed using codes in [traNNsformers](NN)

2. Convolutional Neural networks (CNN) are transformed using codes in:
  a. if GPU support is not available and for small sized networks <CNN>
  b. if GPU support is available and for large sized networks <CNN_wGPU>


## Citation
Please cite the following paper if you find this work useful:

* A. Ankit, T. Ibrayev, A. Sengupta, K. Roy. **TraNNsformer: A Neural Network Transformation Approach for Memristive Crossbar based Neuromorphic System Design**. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2019. 

## Authors

Aayush Ankit, Timur Ibrayev
