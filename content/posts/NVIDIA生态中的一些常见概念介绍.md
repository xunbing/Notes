# NVIDIA生态中的一些常见概念介绍

## 1. 背景

对NVIDIA全系产品常见概念进行简单梳理，以便了解各架构发展历程、显卡算力及引擎开发可能涉及的库。

## 2. 概念区分

- 概念区分
    - 加速卡，Product级别概念，如V100卡，3080Ti卡
    - GPU，Chip级别概念，如V100卡搭载的是GV100的GPU芯片
- CUDA
    - 英伟达的并行计算平台和应用程序编程接口（API）模型。
- CUDA-X
    - CUDA-X AI 是软件加速库的集合。这些库建立在 CUDA® （NVIDIA 的开创性并行编程模型）之上，提供对于深度学习、机器学习和高性能计算 (HPC) 必不可少的优化功能。这些库包括 cuDNN（用于加速深度学习基元）、cuML（用于加速数据科学工作流程和机器学习算法）、NVIDIA® TensorRT™（用于优化受训模型的推理性能）、cuDF（用于访问 pandas 之类的数据科学 API）、cuGraph（用于在图形上执行高性能分析），以及超过 13 个的其他库。
- CUDA Core
    - nvidia GPU的处理器核心或叫像素管道。相同架构下的GPU，越多的CUDA Core代表着越高的性能。什么是架构，有哪些架构，本文有详细说明。GPU架构与CUDA Core的数量一样重要，更先进的架构的加速卡，不一定比用了老架构的卡更快，比如T4就不如V100。
- Stream Processor
    - AMD家的“CUDA Core”。不能将 CUDA Core和Stream Processer等同起来，两者功能相似，但是不可比。
- Tensor Core
    - Nvidia在其Volta架构中引入了Tensor Core这一特殊功能单元，使得Tesla V100的峰值吞吐率可以达到Tesla P100 32位浮点吞吐率的12倍，开发者也可以利用混合精度在不牺牲精度的情况下达到更高的吞吐率。
    - 自Volta架构首次推出 Tensor Core 技术到目前（07）的Hoper架构以来，NVIDIA GPU 的峰值性能提高了 60 倍。
- TFFLOPS
    - 一个TFLOPS（teraFLOPS）等于每秒一万亿（=10^12）次的浮点运算。
- NVLINK
    - 高速互联技术，2016年随Pascal GP100 GPU和Tesla P100加速器一起推出。
    - 与PCIe互联技术相比，可以提供GPU之间GPU和CPU之间更高速的传输性能。
- GDDR6
    - 第六版图形用双倍资料传输率（Graphics Double Data Rate, version 6，简称GDDR6）是一种高带宽的显示存储器标准，用于显卡、游戏终端以及高性能运算上。
- 各个Tensor Core架构支持的精度类型

## 3. 有哪些架构

- NVIDIA Kepler架构
    - 对应GPU芯片如：GK180
    - 对应加速器卡如：Tesla K40
- NVIDIA Maxwell架构
    - 对应GPU芯片如：GM200
    - 对应加速器卡如：Tesla M40
- NVIDIA Pascal™架构
    - 对应GPU芯片如：GP100
    - 对应加速器卡如：Tesla P100
- Volta架构
    - 对应GPU芯片如：GV100
    - 对应加速器卡如：Tesla V100
    - NVIDIA® CUDA®内核和 Tensor 内核搭配使用
    - NVIDIA Volta™ 是第一代tensor core架构。专为深度学习而设计，通过 FP16 和 FP32 下的混合精度矩阵乘法提供了突破性的性能 – 与 NVIDIA Pascal 相比，用于训练的峰值 teraFLOPS (TFLOPS) 性能提升了高达 12 倍，用于推理的峰值 TFLOPS 性能提升了高达 6 倍。这项关键功能使 Volta 提供了比 Pascal 高 3 倍的训练和推理性能。
- Turing架构
    - 对应GPU芯片如：TU104
    - 对应加速器卡如：Tesla T4
    - NVIDIA Turing™架构是第二代Tensor Core架构，能进行多精度计算，可实现高效的 AI 推理。Turing Tensor Core 提供了一系列用于深度学习训练和推理的精度（从 FP32 到 FP16 再到 INT8 和 INT4），性能大大超过 NVIDIA Pascal™ GPU。
- NVIDIA Ampere架构
    - 对应GPU芯片如：GA100
    - 对应加速器卡如：A100 PCIe
    - NVIDIA Ampere 架构 是第三代Tensor Core架构。 基于先前的创新成果而构建，通过使用新的精度（TF32 和 FP64）来加速和简化 AI 采用，并将 Tensor Core 的强大功能扩展至 HPC。这些第三代 Tensor Core 支持 BFloat16、INT8 和 INT4，可为 AI 训练和推理创建高度通用的加速器。
- NVIDIA Hopper™ 架构
    - 对应GPU芯片如：GH100
    - 对应加速器卡如：H100 PCIe
    - H100架构全称NVIDIA Hopper™ 架构，是第四代Tensor Core架构，NVIDIA Hopper™ 架构利用 Transformer 引擎改进第四代 Tensor Core，该引擎使用新的 8 位浮点精度 (FP8)，可为万亿参数模型训练提供比 FP16 高 6 倍的性能。Hopper Tensor Core 使用 TF32、FP64、FP16 和 INT8 精度，将性能提升 3 倍，能够加速处理各种工作负载。
- 下一代架构 Lovelace
    - 预计将在2022年RTX系列新卡正式推出
- 架构更迭时间线
    - 2010-2016 Fermi
    - 2010-2013 VLIW Vec4
    - 2010-2016 Fermi 2.0
    - 2012-2018 Kepler
    - 2013-2015 Kepler 2.0
    - 2014-2017 Maxwell
    - 2014-2019 Maxwell 2.0
    - 2016-2021 Pascal
    - 2017-2020 Volta
    - 2018-2022 Turing
    - 2020-2022 Ampere
    - 2022 Hopper
    - 2022-2023 Lovelace

## 4. 有哪些产品

根据[NVIDIA官网](https://developer.nvidia.com/cuda-gpus#collapseOne)的信息和自己的调研，个人从两个角度进行了分类，仅做了解用，不专业地方见谅。

- 按照工作场景分类
    - 工作站
        - Tesla K40、Tesla K80
    - 数据中心
        - Tesla K、M、P、T、A系列
    - 移动设备
        - 不一一列举，详见
            
            [官网](https://developer.nvidia.com/cuda-gpus#collapseOne)
            
    - 桌面
        - 不一一列举，详见
            
            [官网](https://developer.nvidia.com/cuda-gpus#collapseOne)
            
    - 笔记本
        - 不一一列举，详见
            
            [官网](https://developer.nvidia.com/cuda-gpus#collapseOne)
            
    - 按照使用场景分类
        - 计算卡
            - Tesla(特斯拉)：主要用于服务器高性能电脑运算，Tesla一般是不设计外接接口，主要是辅助CPU去计算所需应用，常应用于研究物理、生化和深度学习等领域；Tesla与Nvidia Grid系列产品全部由NVIDIA原厂设计和生产，产品品质和服务都更有保障，毕竟应用于高科技领域，Nvidia肯定是将技术核心掌握在手中。
        - 图形卡
            - Quadro系列：是面向各种3D设计软件、CAD软件、工业模型设计软件而推出。授权生产的厂商非常有限，一般比GeForce的显存更大，可靠性更高。欧美地区由美国的必恩威(PNY)负责，而台湾的丽台(Leadtek)负责欧亚太地区的销售。艾尔莎日本(Elsa Japan)则仅仅在日本有销售的权利。这三家公司互不进入对方所在的市场，所以我们见到的全新Quadro显卡都是属于丽台生产的。
            - Quadro NVS：属于Quadro产品线中的一个系列，用于多显示商用显卡，可用多个虚拟桌面显示，协助企业安装部署的多种预设设定。
        - 消费级显卡
            - GeForce系列：面向消费者的图形处理产品，主要用于台式电脑和笔记本电脑；主要由第三方厂商生产，而且还区分为采用原厂设计的公版型号和厂商自行设计的非公版型号，其产品的稳定性可能也因不同厂商的设计和工艺水平存在差异。
        - 其他
            - Nvidia Grid：Nvidia用于图形虚拟化的一套硬件和服务，可以根据用户需求分配使用量，这意味着，多名用户可以共享单一 GPU。
            - Tegra：用于移动设备的芯片系列，常应用于手机及平板电脑等移动端使用；Tegra系列产品由NVIDIA与微软合作设计和生产。
        - 题外话：矿卡
            - 主流矿卡也与时俱进，一般是中高端消费级显卡。
            - 24小时不间断的运行。通常矿卡可以使用半年以上，具体要看用的矿卡的型号和用法。矿卡使用6个月的性能降低是比较小的，但使用九个月以后它的显存频率就会降低，之后再继续使用就容易坏，如果性能稍强的矿卡能用时间就会更久，但一般不会超过一年半。

## 5. 几张常见卡的算力比较

## 6. 更多介绍

### CUDA-X：

- NVIDIA TensorRT
    - NVIDIA® TensorRT™ 是一款高性能推理平台，在充分发挥 NVIDIA Tensor Core GPU 的强大功能方面发挥着关键作用。与仅使用 CPU 的平台相比，TensorRT 可使吞吐量提升高达 40 倍，同时还可更大限度地降低延迟。使用 TensorRT，您可以从任何框架入手，并在生产环境中快速优化、验证和部署经过训练的神经网络。TensorRT 还可在 NVIDIA NGC 目录上使用。
- NVIDIA TRITON
    - NVIDIA Triton 推理服务器（以前称为 TensorRT 推理服务器）是一款开源软件，可简化深度学习模型在生产环境中的部署。借助 Triton 推理服务器，团队可以通过任何框架（TensorFlow、PyTorch、TensorRT Plan、Caffe、MXNet 或自定义框架），在任何基于 GPU 或 CPU 的基础设施上从本地存储、Google 云端平台或 AWS S3 部署经过训练的 AI 模型。它可在单个 GPU 上同时运行多个模型，以更大限度地提高利用率，并可与 Kubernetes 集成以用于编排、指标和自动扩展。
- cuDNN
    - NVIDIA CUDA® 深度神经网络库 (cuDNN) 是一个 GPU 加速的深度神经网络基元库，能够以高度优化的方式实现标准例程（如前向和反向卷积、池化层、归一化和激活层）。
- NCCL
    - NCCL是Nvidia Collective multi-GPU Communication Library的简称，它是一个实现多GPU的collective communication通信（all-gather, reduce, broadcast）库，Nvidia做了很多优化，以在PCIe、Nvlink、InfiniBand上实现较高的通信速度。
- Cublas
    - cuBLAS是CUDA专门用来解决线性代数运算的库，分为三个级别：Lev1向量乘向量、Lev2矩阵乘向量、Lev3矩阵乘矩阵。CUDA自带cublas，无需额外安装。
    - cuDnn是CUDA专门用来解决神经网络加速的库，速度快、内存开销低。cuDNN可以方便的集成到更高级别的机器学习框架中(PyTorch、Tensorflow)，支持常用神经网络组件，需要独立安装。
- 更多详见：
    
    [NVIDIA CUDA-X | NVIDIA Developer](https://developer.nvidia.com/gpu-accelerated-libraries)
    

### CUDA-X以外：

- OpenCL
    - 是由苹果（Apple）公司发起，业界众多著名厂商共同制作的面向异构系统通用目的并行编程的开放式、免费标准，也是一个统一的编程环境。CUDA和OpenCL的关系并不是冲突关系，而是包容关系。CUDA C是一种高级语言，那些对硬件了解不多的非专业人士也能轻松上手；而OpenCL则是针对硬件的应用程序开发接口，它能给程序员更多对硬件的控制权，相应的上手及开发会比较难一些。

## 7. 参考链接

- 
    
    [NVIDIA 技术和 GPU 架构 英伟达](https://www.nvidia.cn/technologies/)
    
- 
    
    [CUDA GPUs算力一览](https://developer.nvidia.com/cuda-gpus#collapseOne)
    
- 
    
    [CUDA Cores vs Stream Processors Explained](https://graphicscardhub.com/cuda-cores-vs-stream-processors/)
    
- 
    
    [Nvidia Turing vs Volta v Pascal GPU Architecture Comparison |夏天的风的博客](http://xiaqunfeng.cc/2019/12/06/turing-vs-volta-v-pascal/)
    
- 
    
    [NVIDIA GM200 GPU Specs | TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/nvidia-gm200.g772)
    
- 
    
    [GPU Database | TechPowerUp](https://www.techpowerup.com/gpu-specs/)
    
- 
    
    [GeForce和Quadro图形卡有什么区别？ - 产品评测](http://www.hp168.com/news_view.aspx?id=706&nid=2&typeid=50027&IsActiveTarget=True)
    
- 
    
    [NVIDIA CUDA-X | NVIDIA Developer](https://developer.nvidia.com/gpu-accelerated-libraries)
    
- 
    
    [NVIDIA显卡GeForce、Tesla、Quadro、Tegra、Tesla、等详细介绍](https://blog.csdn.net/weixin_39883440/article/details/111694586)