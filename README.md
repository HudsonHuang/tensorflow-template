Tensorflow-template
===================

| **`Linux CPU`** | 
|-----------------|
| ![Build Status](https://travis-ci.org/HudsonHuang/tensorflow-template.svg?branch=master) | 

Template for tensorflow projects, maximizing code reuse.
 
一个文件组织模板。

I am a beginner and found that there is a far distance from the tensorflow example to mature, community-recognized tensorflow code.    
In order to maximize the extent of model reuse, and limit the frequently user-modified parts in a few files, the community gradually formed a set of unwritten standards. As a beginner, I try to discover these standards and start learning in an orderly manner.  

我是一个初学者，发现从tensorflow官网的例子，到成熟的，社区公认的tensorflow代码有不小的距离。  
为了最大化模型复用程度，并把要频繁要自己写的部分都限定在单独的几个文件中，社区逐渐形成了一套不成文的标准。我作为初学者，尝试去发现这些标准并有条理地开始学习。

# Key concepts 核心概念
- module.py和model.py是子图和主图  
module.py and model.py is sub-graph and main-graph
- dataset_prepare.py是数据准备器  
dataset_prepare.py is the data pre-processor
- main是图的运行器（以指定方式，通过给图注入数据进行运行）  
main.py is the runner of the graph(feed the prepared data into graph in given manner)


# Zen of Deep Learning codes 深度学习代码之禅
- Let anyone run with one command  
要让复现的人一个命令就能跑通
- Let anyone who want to improve the model focus on one file, model.py  
让想要改进的人只需要集中改一个文件 model.py
- Decouples the model, data, and code (those that are independent of the model and the data)  
把模型，数据和代码（跟模型和数据无关的那些代码）解耦
- Experimental steps should write on the .sh files for better debugging  
实验步骤应该放在.sh文件中里便于调试
- Use less TensorFlow for plain style, to the contrary where performance is important
要简洁的地方少用TensorFlow，需要性能的地方反之


# Usage 用法
- Default run 默认用法:  
Just run
`
bash all_default.sh
`
And wait everything done.    
只需运行bash all_default.sh即可

- To write a new model:
  - Prepare dataset(download,extract,and simply pre-processing) with dataset_prepare.py
  - Define network in ./models
  - Define modules in ./module
  - Define params in hprams.py
  - Define placeholder to feed in main.py, as the input of algorithm
  - [optional]Define dataset preprocessing mapper function in preprocessing_util.py
  - [optional]Define experiment with a new .sh file in ./experiment

# Sign 符号
- x: input data  
x：输入数据
- inputs: input of the model  
inputs：模型的输入
- y: traget data (As label in supervised learning)  
y：目标数据（在监督学习中，就是label）
- y_hat or y_: prediction of y  
y_hat或者y_：y的预测值

# TODO 待办
- [ ] Compatible with common styles like [this](https://github.com/wiseodd/generative-models)   
兼容常见模型定义风格，比如[这个](https://github.com/wiseodd/generative-models)


# Give up 舍弃
- [ ] Try tensorflow Estimator API to decouples Algorithm(net+loss+optim+...) and meta-Algorithm(experiment procedure: train, test,etc.)  
使用tensorflow Estimator API对main.py中的算法(net+loss+optim+...)和实验步骤(train, test,etc.)（实验步骤也可以作为元算法的组件，比如GAN）解耦

Any advices and contributions are welcome! Please click [Here](https://github.com/HudsonHuang/tensorflow-template/issues/new) to give your comments.   
欢迎各种建议和共建！
请点[此处](https://github.com/HudsonHuang/tensorflow-template/issues/new)以提出您的建议。



