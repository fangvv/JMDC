## JMDC

This is the source code for our paper: **JMDC: A Joint Model and Data Compression System for Deep Neural Networks Collaborative Computing in Edge-Cloud Networks**. A brief introduction of this work is as follows:

> Deep Neural Networks (DNNs) have shown exceptional promise in providing Artificial Intelligence (AI) to many computer vision applications. Nevertheless, complex models, intensive computations, and resource constraints hinder the use of DNNs in edge computing scenarios. Existing studies have already focused on edge-cloud collaborative inference, where a DNN model is decoupled at an intermediate layer, and the two parts are sequentially executed at the edge device and the cloud server, respectively. In this work, we examine the status quo approaches of DNN execution, and find that it still takes a lot of time on edge device computation and edge-cloud data transmission. Using this insight, we propose a new edge-cloud DNN collaborative computing framework, JMDC, based on Joint Model and Data Compression. In JMDC, we adopt the attention mechanism to select important model channels for efficient inference computation, and important regions of the intermediate output for transferring to the cloud. We further use the quantization technique to reduce actual bits needed to be transferred. Depending on the specific application requirements on latency or accuracy, JMDC can adaptively determine the optimal partition point and compression strategy under different resource conditions. By extensive experiments based on the Raspberry Pi 4B device and the CIFAR10 dataset, we demonstrate the effectiveness of JMDC in enabling on-demand low-latency DNN inference, and its superiority over other baseline schemes.

> 深度神经网络（DNNs）在为许多计算机视觉应用提供人工智能（AI）方面显示出非凡的潜力。然而，复杂的模型、密集的计算和资源限制阻碍了DNNs在边缘计算场景中的应用。现有研究已聚焦于边缘-云协同推理，其中DNN模型在一个中间层被解耦，两个部分分别在边缘设备和云服务器上顺序执行。在这项工作中，我们审视了DNN执行的现状方法，发现边缘设备计算和边缘-云数据传输仍需耗费大量时间。基于这一洞察，我们提出了一种新的边缘-云DNN协同计算框架JMDC，基于联合模型和数据压缩。在JMDC中，我们采用注意力机制选择重要的模型通道以实现高效推理计算，并选择中间输出的重要区域传输到云端。我们进一步使用量化技术减少需要传输的实际位数。根据特定应用对延迟或精度的要求，JMDC能够在不同资源条件下自适应地确定最佳分割点和压缩策略。通过基于Raspberry Pi 4B设备和CIFAR10数据集的广泛实验，我们证明了JMDC在实现按需低延迟DNN推理方面的有效性及其相较于其他基准方案的优越性。

This work will be published by JPDC (Journal of Parallel and Distributed Computing). Click [here](https://doi.org/10.1016/j.jpdc.2022.11.008) for our paper.

## Required software

PyTorch

## Citation
    @article{ding2023jmdc,
    title={JMDC: A joint model and data compression system for deep neural networks collaborative computing in edge-cloud networks},
    author={Ding, Yi and Fang, Weiwei and Liu, Mengran and Wang, Meng and Cheng, Yusong and Xiong, Naixue},
    journal={Journal of Parallel and Distributed Computing},
    volume={173},
     pages={83--93},
    year={2023},
    publisher={Elsevier}
    }

## Contact

Mengran Liu (18800191663@163.com)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
