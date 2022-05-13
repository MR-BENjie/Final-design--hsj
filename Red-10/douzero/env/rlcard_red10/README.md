# 红十游戏环境Red_10
<img width="500" src="https://dczha.com/files/rlcard/logo.jpg" alt="Logo" />
RLCard是一款卡牌游戏强化学习 (Reinforcement Learning, RL) 的工具包。我们利用RLCard构造红十游戏环境rlcard_red10


## 安装
确保您已安装**Python 3.6+**和**pip**。我们推荐您使用`pip`安装稳定版本`rlcard`：

```
pip3 install rlcard
```
默认安装方式只包括卡牌环境。如果想使用PyTorch实现的训练算法，运行
```
pip3 install rlcard[torch]
```
如果您访问较慢，国内用户可以通过清华镜像源安装：
```
pip3 install rlcard -i https://pypi.tuna.tsinghua.edu.cn/simple
```
或者，您可以克隆最新版本（如果您访问Github较慢，国内用户可以使用[Gitee镜像](https://gitee.com/daochenzha/rlcard)）：
```
git clone https://github.com/datamllab/rlcard.git
```
或使只克隆一个分支以使其更快
```
git clone -b master --single-branch --depth=1 https://github.com/datamllab/rlcard.git
```
运行以下命令进行安装
```
pip3 install -e .
pip3 install -e .[torch]
```