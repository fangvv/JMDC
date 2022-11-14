## JMDC

This is the source code for our paper: **JMDC: A Joint Model and Data Compression System for Deep Neural Networks Collaborative Computing in Edge-Cloud Networks**. A brief introduction of this work is as follows:

> Deep Neural Networks (DNNs) have shown exceptional promise in providing Artificial Intelligence (AI) to many computer vision applications. Nevertheless, complex models, intensive computations, and resource constraints hinder the use of DNNs in edge computing scenarios. Existing studies have already focused on edge-cloud collaborative inference, where a DNN model is decoupled at an intermediate layer, and the two parts are sequentially executed at the edge device and the cloud server, respectively. In this work, we examine the status quo approaches of DNN execution, and find that it still takes a lot of time on edge device computation and edge-cloud data transmission. Using this insight, we propose a new edge-cloud DNN collaborative computing framework, JMDC, based on Joint Model and Data Compression. In JMDC, we adopt the attention mechanism to select important model channels for efficient inference computation, and important regions of the intermediate output for transferring to the cloud. We further use the quantization technique to reduce actual bits needed to be transferred. Depending on the specific application requirements on latency or accuracy, JMDC can adaptively determine the optimal partition point and compression strategy under different resource conditions. By extensive experiments based on the Raspberry Pi 4B device and the CIFAR10 dataset, we demonstrate the effectiveness of JMDC in enabling on-demand low-latency DNN inference, and its superiority over other baseline schemes.

> 通过模型压缩和数据压缩来实现高效的边云网络上深度神经网络的协同计算

This work will be published by JPDC (Journal of Parallel and Distributed Computing).

## Required software

PyTorch

## Contact

Weiwei Fang (fangvv@qq.com)

Mengran Liu (18800191663@163.com)
