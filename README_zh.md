<!-- ## **HunyuanVideo** -->

[English Version](./README.md)
[日本語はこちら](./README_ja.md)

<p align="center">
  <img src="./assets/logo.png"  height=100>
</p>

# **HunyuanVideo-I2V** 🌅

<div align="center">
  <a href="https://github.com/Tencent/HunyuanVideo-I2V"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-I2V 代码&message=Github&color=blue"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=项目主页&message=Web&color=green"></a> &ensp;
  <a href="https://video.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=在线体验&message=Web&color=green"></a>
</div>
<div align="center">
  <a href="https://arxiv.org/abs/2412.03603"><img src="https://img.shields.io/static/v1?label=技术报告&message=Arxiv&color=red"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com/hunyuanvideo.pdf"><img src="https://img.shields.io/static/v1?label=技术报告&message=高清版本 (~350M)&color=red"></a>
</div>
<div align="center">
  <a href="https://huggingface.co/tencent/HunyuanVideo-I2V"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-I2V&message=HuggingFace&color=yellow"></a> &ensp;
</div>

<p align="center">
    👋 加入我们的<a href="assets/WECHAT.md" target="_blank">微信社区</a>和<a href="https://discord.gg/tv7FkG4Nwf" target="_blank">Discord</a> 
</p>

-----

继我们成功开源[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)后，我们很高兴推出[HunyuanVideo-I2V](https://github.com/Tencent/HunyuanVideo-I2V)，一个新的图像到视频生成框架，加速开源社区的探索！

本仓库包含官方PyTorch模型定义、预训练权重及推理/采样代码。更多可视化效果请访问[项目主页](https://aivideo.hunyuan.tencent.com)。同时，我们发布了LoRA训练代码，用于定制化特效生成，可创建更有趣的视频效果。

> [**HunyuanVideo: A Systematic Framework For Large Video Generation Model**](https://arxiv.org/abs/2412.03603)

## 🔥🔥🔥 最新动态
* 2025年3月6日: 👋 发布HunyuanVideo-I2V的推理代码和模型权重。[下载地址](https://github.com/Tencent/HunyuanVideo-I2V/blob/main/ckpts/README.md)

## 🎥 演示
### I2V 示例
<div align="center">
  <video src="https://github.com/user-attachments/assets/442afb73-3092-454f-bc46-02361c285930" width="80%" poster="./assets/video_poster.jpg"> </video>
  <p>联合创作 @D-aiY 导演 丁一</p>
</div>

### 定制化I2V LoRA效果演示

| 特效类型       |  参考图像  | 生成视频  |
|:---------------:|:--------------------------------:|:----------------:|
|   头发生长   |        <img src="./assets/demo/i2v_lora/imgs/hair_growth.png" width="40%">         |       <video src="https://github.com/user-attachments/assets/06b998ae-bbde-4c1f-96cb-a25a9197d5cb" width="100%"> </video>        |
|     拥抱     |      <img src="./assets/demo/i2v_lora/imgs/embrace.png" width="40%">          |       <video src="https://github.com/user-attachments/assets/f8c99eb1-2a43-489a-ba02-6bd50a6dd260" width="100%" > </video>        |

## 📑 开源计划
- HunyuanVideo-I2V（图像到视频模型）
  - [x] LoRA训练脚本
  - [x] 推理代码
  - [x] 模型权重
  - [x] ComfyUI支持
  - [ ] 多GPU序列并行推理（提升多卡推理速度）
  - [ ] Diffusers集成 
  - [ ] FP8量化权重

## 目录
- [**HunyuanVideo-I2V** 🌅](#hunyuanvideo-i2v-)
  - [🔥🔥🔥 最新动态](#-最新动态)
  - [🎥 演示](#-演示)
    - [I2V 示例](#i2v-示例)
    - [定制化I2V LoRA效果演示](#定制化i2v-lora效果演示)
  - [📑 开源计划](#-开源计划)
  - [目录](#目录)
  - [**HunyuanVideo-I2V 整体架构**](#hunyuanvideo-i2v-整体架构)
  - [📜 运行要求](#-运行要求)
  - [🛠️ 依赖安装](#️-依赖安装)
    - [Linux 安装指引](#linux-安装指引)
  - [🧱 下载预训练模型](#-下载预训练模型)
  - [🔑 单 GPU 推理](#-单-gpu-推理)
    - [使用图生视频模型的建议](#使用图生视频模型的建议)
    - [使用命令行](#使用命令行)
    - [更多配置](#更多配置)
  - [🎉自定义 I2V LoRA 效果训练](#自定义-i2v-lora-效果训练)
    - [要求](#要求)
    - [训练环境](#训练环境)
    - [训练数据构建](#训练数据构建)
    - [开始训练](#开始训练)
    - [推理](#推理)
  - [🔗 BibTeX](#-bibtex)
  - [致谢](#致谢)

---

## **HunyuanVideo-I2V 整体架构**
基于[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)强大的视频生成能力，我们将其扩展至图像到视频生成任务。为此，我们采用图像隐空间拼接技术，有效重构并融合参考图像信息至视频生成流程中。

由于我们使用预训练的Decoder-Only架构多模态大语言模型（MLLM）作为文本编码器，可用于显著增强模型对输入图像语义内容的理解能力，并实现图像与文本描述信息的深度融合。具体而言，输入图像经MLLM处理后生成语义图像tokens，这些tokens与视频隐空间tokens拼接，实现跨模态的全注意力计算。

我们的系统架构旨在最大化图像与文本模态的协同效应，确保从静态图像生成连贯的视频内容。该集成不仅提升了生成视频的保真度，还增强了模型对复杂多模态输入的解析能力。整体架构如下图所示：
<p align="center">
  <img src="./assets/backbone.png"  height=300>
</p>

## 📜 运行要求

下表展示了运行HunyuanVideo-I2V模型（batch size=1）生成视频的硬件要求：

|      模型       | 分辨率  | GPU显存峰值 |
|:---------------:|:-------:|:-----------:|
| HunyuanVideo-I2V |  720p   |    60GB     |

* 需配备支持CUDA的NVIDIA GPU
  * 测试环境为单卡80G GPU
  * **最低要求**: 720p分辨率需至少60GB显存
  * **推荐配置**: 建议使用80GB显存GPU以获得更佳生成质量
* 测试操作系统：Linux

## 🛠️ 依赖安装

首先克隆仓库：
```shell
git clone https://github.com/tencent/HunyuanVideo-I2V
cd HunyuanVideo-I2V
```

### Linux 安装指引

我们推荐使用 CUDA 12.4 或 11.8 的版本。

Conda 的安装指南可以参考[这里](https://docs.anaconda.com/free/miniconda/index.html)。

```shell
# 1. 创建conda环境
conda create -n HunyuanVideo-I2V python==3.11.9

# 2. 激活环境
conda activate HunyuanVideo-I2V

# 3. 通过conda安装PyTorch等依赖
# CUDA 12.4版本
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. 安装pip依赖
python -m pip install -r requirements.txt

# 5. 安装flash attention v2加速（需CUDA 11.8及以上）
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

如果在特定 GPU 型号上遭遇 float point exception(core dump) 问题，可尝试以下方案修复：

```shell
# 确保已安装CUDA 12.4、CUBLAS>=12.4.5.8和CUDNN>=9.00（或直接使用我们的CUDA 12 docker镜像）
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/
```

另外，我们提供了一个预构建的 Docker 镜像，可以使用如下命令进行拉取和运行。
```shell
# CUDA 12.4镜像（避免浮点异常）
docker pull hunyuanvideo/hunyuanvideo-i2v:cuda_12
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo-i2v --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo-i2v:cuda_12
```

## 🧱 下载预训练模型

下载预训练模型的详细信息请参见 [here](ckpts/README.md)。

## 🔑 单 GPU 推理

类似于 [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)，HunyuanVideo-I2V 支持高分辨率视频生成，分辨率最高可达 720P，视频长度最高可达 129 帧（5 秒）。
### 使用图生视频模型的建议

- **使用简短的提示**：为了有效地引导模型的生成，请保持提示简短且直截了当。
- **包含关键元素**：一个结构良好的提示应包括：
  - **主体**：指定视频的主要焦点。
  - **动作**：描述正在发生的运动或活动。
  - **背景（可选）**：设置视频的场景。
  - **镜头（可选）**：指示视角或视点。
- **避免过于详细的提示**：冗长或高度详细的提示可能会导致视频输出中出现不必要的转场。

### 使用命令行

```bash
cd HunyuanVideo-I2V

python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --prompt "A man with short gray hair plays a red electric guitar." \
    --i2v-mode \
    --i2v-image-path ./assets/demo/i2v/imgs/0.png \
    --i2v-resolution 720p \
    --video-length 129 \
    --infer-steps 50 \
    --flow-reverse \
    --flow-shift 17.0 \
    --seed 0 \
    --use-cpu-offload \
    --save-path ./results 
```
<!-- # ### 运行gradio服务
# ```bash
# python3 gradio_server.py --flow-reverse

# # set SERVER_NAME and SERVER_PORT manually
# # SERVER_NAME=0.0.0.0 SERVER_PORT=8081 python3 gradio_server.py --flow-reverse
# ``` -->

### 更多配置

我们列出了一些常用的配置以方便使用：

|        参数        |            默认            |                          描述                          |
|:----------------------:|:-----------------------------:|:------------------------------------------------------------:|
|       `--prompt`       |             None              |           用于视频生成的文本提示。               |
|       `--model`        |      HYVideo-T/2-cfgdistill   | 这里我们使用 HYVideo-T/2 用于 I2V，HYVideo-T/2-cfgdistill 用于 T2V 模式。 |
|     `--i2v-mode`       |            False              |                是否开启 I2V 模式。                      |
|  `--i2v-image-path`    | ./assets/demo/i2v/imgs/0.png  |        用于视频生成的参考图像。              |
|  `--i2v-resolution`    |            720p               |        生成视频的分辨率。                |
|    `--video-length`    |             129               |         生成视频的长度。                    |
|    `--infer-steps`     |              50               |         采样步骤的数量。                     |
|     `--flow-shift`     |             7.0               |     流匹配调度器的偏移因子。               |
|   `--flow-reverse`     |            False              | 如果反转，从 t=1 学习/采样到 t=0。                |
|        `--seed`        |             None              | 生成视频的随机种子，如果为 None，则初始化一个随机种子。 |
|  `--use-cpu-offload`   |            False              | 使用 CPU 卸载模型加载以节省更多内存，对于高分辨率视频生成是必要的。 |
|     `--save-path`      |         ./results             |         保存生成视频的路径。                     |


## 🎉自定义 I2V LoRA 效果训练

###  要求

下表显示了训练 HunyuanVideo-I2V lora 模型（批量大小 = 1）以生成视频的要求：

|      模型       | 分辨率 | GPU 峰值内存 |
|:----------------:|:----------:|:---------------:|
| HunyuanVideo-I2V |    360p    |      79GB       |

* 需要支持 CUDA 的 NVIDIA GPU。
  * 该模型在单个 80G GPU 上进行了测试。
  * **最低要求**: 生成 360p 视频所需的最小 GPU 内存为 79GB。
  * **推荐**: 建议使用 80GB 内存的 GPU 以获得更好的生成质量。
* 测试操作系统: Linux
* 注意: 您可以使用 360p 数据进行训练，并直接推断 540p 视频

### 训练环境
```
pip install -r requirements.txt
```

### 训练数据构建
提示描述：触发词直接写在视频说明中。建议使用短语或简短句子。

例如，AI 头发生长效果（触发词）：rapid_hair_growth, The hair of the characters in the video is growing rapidly. + 原始提示

准备好训练视频和提示对后，参考 [这里](hyvideo/hyvae_extract/README.md) 进行训练数据构建。


### 开始训练
```
sh scripts/run_train_image2video_lora.sh
```
我们列出了一些训练特定配置以方便使用：

|     参数     |                            默认                            |                         描述                         |
|:----------------:|:-------------------------------------------------------------:|:-----------------------------------------------------------:|
|   `SAVE_BASE`    |                               .                               |         保存实验结果的根路径。          |
|    `EXP_NAME`    |                           i2v_lora                            |        保存实验结果的路径后缀。         |
| `DATA_JSONS_DIR` | ./assets/demo/i2v_lora/train_dataset/processed_data/json_path | 由 hyvideo/hyvae_extract/start.sh 生成的数据 jsons 目录。 |
|    `CHIEF_IP`    |                            0.0.0.0                            |            主节点 IP 地址。                   |

### 推理
```bash
python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --prompt "Two people hugged tightly, In the video, two people are standing apart from each other. They then move closer to each other and begin to hug tightly. The hug is very affectionate, with the two people holding each other tightly and looking into each other's eyes. The interaction is very emotional and heartwarming, with the two people expressing their love and affection for each other." \
    --i2v-mode \
    --i2v-image-path ./assets/demo/i2v_lora/imgs/embrace.png \
    --i2v-resolution 540p \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 5.0 \
    --seed 0 \
    --use-cpu-offload \
    --save-path ./results \
    --use-lora \
    --lora-scale 1.0 \
    --lora-path ./ckpts/hunyuan-video-i2v-720p/lora/embrace_kohaya_weights.safetensors \
```
我们列出了一些 LoRA 特定配置以方便使用：

|      参数       | 默认 |         描述          |
|:-------------------:|:-------:|:----------------------------:|
|    `--use-lora`     |  None   |  是否开启 LoRA 模式。  |
|   `--lora-scale`    |   1.0   | LoRA 模型的融合比例。 |
|   `--lora-path`     |   ""    |  LoRA 模型的权重路径。 |

## 🔗 BibTeX

如果您发现 [HunyuanVideo](https://arxiv.org/abs/2412.03603) 对您的研究和应用有所帮助，请使用以下 BibTeX 引用：

```BibTeX
@misc{kong2024hunyuanvideo,
      title={HunyuanVideo: A Systematic Framework For Large Video Generative Models}, 
      author={Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Junkun Yuan, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yanxin Long, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, and Jie Jiang, along with Caesar Zhong},
      year={2024},
      archivePrefix={arXiv preprint arXiv:2412.03603},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.03603}, 
}
```

## 致谢

HunyuanVideo 的开源离不开诸多开源工作，这里我们特别感谢 [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) 的开源工作和探索。另外，我们也感谢腾讯混元多模态团队对 HunyuanVideo 适配多种文本编码器的支持。



<!-- ## Star 趋势

<a href="https://star-history.com/#Tencent/HunyuanVideo&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
 </picture>
</a> -->


<!-- # I2V + lora

## 训练环境
```
pip install -r requirements.txt
```

## 训练数据构造
prompt说明: trigger词直接写在video caption里面，建议用短语或短句, 比如

比如ai生发特效：rapid_hair_growth, The hair of the characters in the video is growing rapidly. + 原始prompt

有了训练视频和prompt对后，训练数据构造参考[这里](hyvideo/hyvae_extract/README.md)。


## 启动训练
```
sh scripts/run_train_image2video_lora.sh
# 重要参数
# --data-jsons-path 训练数据路径
# --model  训练底模
# --output-dir lora存放位置
```

## 推理
```
sh scripts/run_sample_image2video.sh
# 重要参数
# --prompt 推理prompt
# --i2v-image-path 输入图片位置
# --lora-path 待加载lora位置
# --lora-scale lora加载权重
``` -->

