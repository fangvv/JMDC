# JMDC
This is the source code for our paper: **JMDC: A Joint Model and Data Compression System for Deep Neural Networks Collaborative Computing in Edge-Cloud Networks**. A brief introduction of this work is as follows:
> Deep Neural Networks (DNNs) have shown exceptional promise in providing Artificial Intelligence (AI) to many computer vision applications. Nevertheless, complex models, intensive computations, and resource constraints hinder the use of DNNs in edge computing scenarios. Existing studies have already focused on edge-cloud collaborative inference, where a DNN model is decoupled at an intermediate layer, and the two parts are sequentially executed at the edge device and the cloud server, respectively. In this work, we examine the status quo approaches of DNN execution, and find that it still takes a lot of time on edge device computation and edge-cloud data transmission. Using this insight, we propose a new edge-cloud DNN collaborative computing framework, JMDC, based on Joint Model and Data Compression. In JMDC, we adopt the attention mechanism to select important model channels for efficient inference computation, and important regions of the intermediate output for transferring to the cloud. We further use the quantization technique to reduce actual bits needed to be transferred. Depending on the specific application requirements on latency or accuracy, JMDC can adaptively determine the optimal partition point and compression strategy under different resource conditions. By extensive experiments based on the Raspberry Pi 4B device and the CIFAR10 dataset, we demonstrate the effectiveness of JMDC in enabling on-demand low-latency DNN inference, and its superiority over other baseline schemes.
> 深度神经网络（DNNs）在为许多计算机视觉应用提供人工智能（AI）方面显示出非凡的潜力。然而，复杂的模型、密集的计算和资源限制阻碍了DNNs在边缘计算场景中的应用。现有研究已聚焦于边缘-云协同推理，其中DNN模型在一个中间层被解耦，两个部分分别在边缘设备和云服务器上顺序执行。在这项工作中，我们审视了DNN执行的现状方法，发现边缘设备计算和边缘-云数据传输仍需耗费大量时间。基于这一洞察，我们提出了一种新的边缘-云DNN协同计算框架JMDC，基于联合模型和数据压缩。在JMDC中，我们采用注意力机制选择重要的模型通道以实现高效推理计算，并选择中间输出的重要区域传输到云端。我们进一步使用量化技术减少需要传输的实际位数。根据特定应用对延迟或精度的要求，JMDC能够在不同资源条件下自适应地确定最佳分割点和压缩策略。通过基于Raspberry Pi 4B设备和CIFAR10数据集的广泛实验，我们证明了JMDC在实现按需低延迟DNN推理方面的有效性及其相较于其他基准方案的优越性。

This work has been published by JPDC (Journal of Parallel and Distributed Computing). Click [here](https://doi.org/10.1016/j.jpdc.2022.11.008) for our paper.
## Required software
- PyTorch
- NumPy
- torchvision
## Project Structure
```
JMDC/
├── cloud/                              # Cloud-side server code
│   ├── model/                         # Pre-trained model files (.pkl)
│   │   ├── alexnetlayermodel.pkl
│   │   └── googlelenet.pkl
│   ├── initCloud.py                   # Cloud server: receives offloaded tasks via TCP
│   ├── initMobile.py                  # Mobile simulator: client that connects to cloud server
│   ├── init.py                        # Alternative mobile client entry point
│   ├── alexnetlayer.py                # AlexNet with layer-wise partition support
│   ├── vgg16layer.py                  # VGG16 with layer-wise partition support
│   ├── data.py                        # CIFAR-10 dataset loader
│   ├── Quantification.py              # Quantization module for data compression
│   └── predict.py                     # Standalone model accuracy evaluation
├── mobile/                             # Mobile/edge-side code (same structure as cloud/)
│   ├── model/
│   ├── initMobile.py                  # Edge device: runs local layers, offloads to cloud
│   ├── alexnetlayer.py
│   ├── vgg16layer.py
│   ├── data.py
│   ├── Quantification.py
│   └── predict.py
├── image/
│   ├── cloud.jpg
│   └── mobile.jpg
└── README.md
```
## Core Modules
### Layer-wise DNN Models (`alexnetlayer.py` / `vgg16layer.py`)
Each model supports per-layer execution via a custom `forward(x, startLayer, endLayer, isTrain)` method, enabling flexible DNN partitioning between the edge device and the cloud server:
- When `isTrain=True`: full forward pass (training mode).
- When `isTrain=False`: execute only layers from `startLayer` to `endLayer` (inference partitioning mode).

**Supported models:**
| Model | `features` layers | `classifier` layers | Total layers |
|-------|-------------------|---------------------|--------------|
| AlexNet | 10 | 3 | 13 |
| VGG16 | 31 | 3 | 34 |
### Cloud Server (`initCloud.py`)
The cloud-side server that receives partitioned DNN inference tasks from the edge device via TCP socket communication (`IP: 192.168.123.10, PORT: 8081`). Key workflow:
1. Load a pre-trained model from `model/` directory.
2. Listen for incoming TCP connections from edge devices.
3. Receive serialized `Data(inputData, startLayer, endLayer)` objects.
4. Execute the specified DNN layers on the cloud side.
5. Return the intermediate output back to the edge device.
### Edge Client (`initMobile.py`)
The edge/mobile client that performs collaborative DNN inference with the cloud server. Key workflow:
1. Load the pre-trained model and CIFAR-10 test data.
2. Connect to the cloud server via TCP.
3. Based on the partition vector `x` (a binary array where `1` = edge, `0` = cloud), determine which layers run locally and which are offloaded.
4. Run local layers on the edge device, then send intermediate features to the cloud.
5. Receive results from the cloud and compute final accuracy and latency.


### Quantification (`Quantification.py`)
A quantization module that compresses intermediate feature data before transmission to reduce communication overhead. Supports configurable bit-width quantization:
| Function | Description |
|----------|-------------|
| `calcScaleZeroPoint(min_val, max_val, num_bits)` | Compute scale and zero-point for quantization |
| `quantize_tensor(x, num_bits, min_val, max_val)` | Quantize a float tensor to integer representation |
| `dequantize_tensor(scale, x, zero_point)` | Dequantize back to float tensor |
### Data Loader (`data.py`)
Loads the CIFAR-10 dataset from the `datasets/` directory. Supports both training and test set loading with `get_data_set(name="train"/"test")`.
### Accuracy Evaluation (`predict.py`)
Standalone script that loads a pre-trained model and evaluates its accuracy on the full CIFAR-10 test set.
## Usage
### Prerequisites
1. Prepare the CIFAR-10 dataset in `datasets/cifar_10/` directory.
2. Place pre-trained model files (`.pkl`) in the `model/` directory.
### Run Cloud Server
```bash
# Start the cloud server
cd cloud
python initCloud.py --refine <path_to_checkpoint> --arch <alexnetlayer|vgg16layer> --dataset cifar10 --depth <model_depth>
```
### Run Edge Client
```bash
# Start the edge/mobile client (connect to cloud server)
cd cloud  # or cd mobile
python initMobile.py --refine <path_to_checkpoint> --arch <alexnetlayer|vgg16layer> --dataset cifar10 --depth <model_depth>
```
### Evaluate Accuracy (standalone)
```bash
cd cloud  # or cd mobile
python predict.py
```
## Citation
If you find JMDC useful or relevant to your project and research, please kindly cite our paper:
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