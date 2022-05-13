# 扑克策略强化学习（Douzero）
我们构造了置信度和危险率网络用来进行红十游戏中智能体间竞争-合作模式的辨识。

我们利用强化学习对多个智能体间竞争-合作关系模式进行辨识,在关系确定的情况下，我们利用Douzero模型来决定每个智能体的出牌策略
在Douzero模型中，将深度蒙特卡洛（Deep Monte Carlo, DMC）与动作编码和并行actor（Parallel Actors）相结合的方法，为3人斗地主提供了一个简单而有效的解决方案。在这里我们将Douzero模型扩展到4个智能体，用于确定在红十游戏中关系确定的情况下各个智能体出牌策略的选择。

在douzero/env中我们构造了红十游戏环境：rlcard_red10
## 安装
训练部分的代码是基于GPU设计的，因此如果想要训练模型，您需要先安装CUDA。安装步骤可以参考[官网教程](https://docs.nvidia.com/cuda/index.html#installation-guides)。对于评估部分，CUDA是可选项，您可以使用CPU进行评估。

首先，克隆本仓库（如果您访问Github较慢，国内用户可以使用[Gitee镜像](https://gitee.com/daochenzha/DouZero)）：
```
git clone https://github.com/kwai/DouZero.git
```

确保您已经安装好Python 3.6及以上版本，然后安装依赖：
```
cd douzero
pip3 install -r requirements.txt
```
我们推荐通过以下命令安装稳定版本的Douzero：
```
pip3 install douzero
```
如果您访问较慢，国内用户可以通过清华镜像源安装：
```
pip3 install douzero -i https://pypi.tuna.tsinghua.edu.cn/simple
```
或是安装最新版本（可能不稳定）：
```
pip3 install -e .
```
注意，Windows用户只能用CPU来模拟。关于为什么GPU会出问题，详见[Windows下的问题](README.zh-CN.md#Windows下的问题)。

同时我们需要按照douzero/env.rlcard_red10中文件对红十游戏环境进行初始化

## 训练Douzero 模型 和置信度、危险率网络
假定您至少拥有一块可用的GPU，运行
```
python3 train.py
```
这会使用一块GPU训练DouZero。如果需要用多个GPU训练Douzero，使用以下参数：
*   `--gpu_devices`: 用作训练的GPU设备名
*   `--num_actor_devices`: 被用来进行模拟（如自我对弈）的GPU数量
*   `--num_actors`: 每个设备的演员进程数
*   `--training_device`: 用来进行模型训练的设备

例如，如果我们拥有4块GPU，我们想用前3个GPU进行模拟，每个GPU拥有15个演员，而使用第四个GPU进行训练，我们可以运行以下命令：
```
python3 train.py --gpu_devices 0,1,2,3 --num_actor_devices 3 --num_actors 15 --training_device 3
```
如果用CPU进行训练和模拟（Windows用户只能用CPU进行模拟），用以下参数：
*   `--training_device cpu`: 用CPU来训练
*   `--actor_device_cpu`: 用CPU来模拟

例如，用以下命令完全在CPU上运行：
```
python3 train.py --actor_device_cpu --training_device cpu
```
以下命令仅仅用CPU来跑模拟：
```
python3 train.py --actor_device_cpu
```

其他定制化的训练配置可以参考以下可选项：
```
--xpid XPID           实验id（默认值：douzero）
--save_interval SAVE_INTERVAL
                      保存模型的时间间隔（以分钟为单位）
--objective {adp,wp}  使用ADP或者WP作为奖励（默认值：ADP）
--actor_device_cpu    用CPU进行模拟
--gpu_devices GPU_DEVICES
                      用作训练的GPU设备名
--num_actor_devices NUM_ACTOR_DEVICES
                      被用来进行模拟（如自我对弈）的GPU数量
--num_actors NUM_ACTORS
                      每个设备的演员进程数
--training_device TRAINING_DEVICE
                      用来进行模型训练的设备。`cpu`表示用CPU训练
--load_model          读取已有的模型
--disable_checkpoint  禁用保存检查点
--savedir SAVEDIR     实验数据存储跟路径
--total_frames TOTAL_FRAMES
                      Total environment frames to train for
--exp_epsilon EXP_EPSILON
                      探索概率
--batch_size BATCH_SIZE
                      训练批尺寸
--unroll_length UNROLL_LENGTH
                      展开长度（时间维度）
--num_buffers NUM_BUFFERS
                      共享内存缓冲区的数量
--num_threads NUM_THREADS
                      学习者线程数
--max_grad_norm MAX_GRAD_NORM
                      最大梯度范数
--learning_rate LEARNING_RATE
                      学习率
--alpha ALPHA         RMSProp平滑常数
--momentum MOMENTUM   RMSProp momentum
--epsilon EPSILON     RMSProp epsilon
```
训练置信度和危险率网络，可以运行train_RD.py，即将上面指令中的train.py 更换为train_RD.py
## Douzero评估
评估可以使用GPU或CPU进行（GPU效率会高得多）。
### 第1步：生成评估数据
```
python3 generate_eval_data.py
```
这里会为4个智能体初始化最初的手牌
以下为一些重要的超参数。
*   `--output`: pickle数据存储路径
*   `--num_games`: 生成数据的游戏局数，默认值 10000

## 第2步：自我对弈
```
python3 evaluate.py
```
以下为一些重要的超参数。
*   `--landlord`: 主智能体，为预训练Douzero模型的路径
*   `--landlord_up`: 主智能体上家的智能体，预训练模型的路径
*   `--landlord_down`: 主智能体下家的智能体，预训练模型的路径
*   `--landlord_front`: 主智能体对家的智能体，可预训练模型的路径
*   `--eval_data`: 包含评估数据的pickle文件
*   `--num_workers`: 用多少个进程进行模拟
*   `--gpu_device`: 用哪个GPU设备进行模拟。默认用CPU

我们训练好的4组不同阵营模式的Douzero模型权重放置在文件夹douzero_checkpoints下
## 置信度和危险率网络评估
评估可以使用GPU或CPU进行（GPU效率会高得多）。
我们利用evaluate_red_10.py 设计了一些程序用来分析置信度和危险率网络的表现
我们训练好的置信度和危险率网络权重放置在文件夹R_D_checkpoints下