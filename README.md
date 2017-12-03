# What 这是什么
A template for most tensorflow projects.  

是一个文件组织模板。

# Why 为什么要做这个项目
I am a beginner and found that there is a far distance from the tensorflow example to mature, community-recognized tensorflow code.    
In order to maximize the extent of model reuse, the community gradually formed a set of unwritten standards. As a beginner, I try to discover these standards and start learning in an orderly manner.  

我是一个初学者，发现从tensorflow官网的例子，到成熟的，社区公认的tensorflow代码有不小的距离。  
为了最大化模型复用程度，社区逐渐形成了一套不成文的标准。我作为初学者，尝试去发现这些标准并有条理地开始学习。

# Zen of Deep Learning codes 机器学习代码之禅
- Experimental steps should write on the .sh for better debugging  
实验步骤放在bash里便于调试
- Training steps should write with TensorFlow-API to improve performance
训练放在tf里便于提高性能
- With the new model, you can not change more than two files, model.py and architecture.json
有新模型的时候，不要改超过两个文件，model.py 和 architecture.json 
- Let anyone run with one command
要让复现的人一个命令就能跑通
- Let anyone who want to improve the model focus on one file, model.py  
让想要改进的人只需要集中改一个文件 model.py
- Decouples the model, data, and code (those that are independent of the model and the data)  
把模型，数据和代码（跟模型和数据无关的那些代码）解耦

# Goal 目标
Maximizing code reuse, and limit the frequently user-modified parts in a few files.  

最大化代码复用率，并把要频繁要自己写的部分都限定在单独的几个文件中。

# TODO 未竟
- [ ] Move data prepare procedure to download.sh and prepare_features.py  
把数据准备代码移动到download.sh和prepare_features.py中
- [ ] Add Minist_example.sh for these procedure:data_prepare - train - eval  
增加Minist_example.sh并包含以下步骤data_prepare - train - eval
- [ ] Add eval.py to show some summary informations and some visualization code  
增加eval.py用于展示模型验证信息和可视化信息
- [ ] Compatible with common styles like [this](https://github.com/wiseodd/generative-models)   
兼容常见模型定义风格，比如[这个](https://github.com/wiseodd/generative-models)
- [ ] Add module examples ( convolution layers, I think)  
增加模块的例子(如卷积模块)
- [ ] Add .json examples for network architecture definition  
增加使用.json文件定义模型结构的例子

Any advices and contributions are welcome!  
Please click [Here](https://github.com/HudsonHuang/tensorflow-template/issues/new) to give your comments. 

欢迎各种建议和共建！
请点[此处](https://github.com/HudsonHuang/tensorflow-template/issues/new)以提出您的建议。



