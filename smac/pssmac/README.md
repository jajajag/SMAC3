# pssmac模块

pssmac使用PS-Lite作为参数服务器对SMAC算法进行扩展，使SMAC可以在异步并行的环境中运行，从而对串行算法进行加速。

## 概述

SMBO全称为Sequential Model-Based 
Optimization，是一个串行算法。但对于大数据集，模型的训练为速控步，会导致串行算法的时间代价较大。如果我们引入并行性，可以一定程度上的提升算法的效率。因此我们使用参数服务器的架构，在Server
端仅运行SMBO循环并保存runhistory，将计算密集的部分放在Worker端进行实现，Worker通过随机产生不同的初始点，由各个方向逼近最优值点，从而实现异步并行。

我们使用PS-Lite作为参数服务器，处理Server端和Worker端的通讯。SMBO和模型在Python端运行，通讯则由以C++
为基础的PS-Lite处理，Python端和PS-Lite端之间使用PIPE进行通信。

其中，每个Worker和Server，Scheduler先创建连接。然后由每个Worker端以work_id随机生成一组超参Configuration
，运行之后返回loss给SMBO循环中的Server。Server接收超参的Configuration和loss之后，构建经验模型EMP，再由SMBO
算法选出新的备选超参Challengers给Worker。每个Worker收到之后立即开始训练模型。因为SMBO是不等待所有Worker
运行完毕的，因此该算法是异步并行的。

代码主体由四部分组成：

* Server(Python)<br>
* Server(PS-Lite)<br>
* Worker(PS-Lite)<br>
* Worker(Python)<br>

具体的调用流程为：

![Flow Diagram](utils/Flow%20Diagram.png)

## 结构

pssmac模块主要包含三个目录，facade，ps和tae。

### 类继承关系

![Flow Diagram](utils/UML%20Graph.png)

### facade/

facade目录包含一个abstract_facade.py文件用于储存facade的基类，它派生的三个类分别存储于
scheduler_facade.py，server_facade.py和worker_facade.py中。这三个类分别用于处理PS-Lite
的三种结点。

* abstract_facade.py <br>
存储了facade的基类AbstractFacade，在AbstractFacade中定义了三种PS-Lite类公有的init和run
方法。派生出三个子类。

* scheduler_facade.py <br>
定义了scheduler的SchedulerFacade，在这个类中，使用Popen打开一个sheduler进程，进行等待。
scheduler负责协调server和worker之间的通讯。

* server_facade.py <br>
定义了server的ServerFacade，Server的实例、SMAC的参数及SMAC的主要过程都在这个类中运行。
ServerFacade负责处理SMBO的主要过程，包括构建经验模型EPM，预测最优点出现的位置和选点。

* worker_facade.py <br>
定义了worker的WorkerFacade，实例化了Intensification类并调用了Worker的子进程，同时定义了，
临时目录，stats，traj_logger等乱七八糟的东西。

### ps/

ps目录下的代码主要用于创建PS-Lite的C++进程，并使用管道在PS-Lite的程序和Python端程序之间进行
通讯和交互。PS-Lite主要用于Server和Worker之间信息的传递，具体的算法由Python端进行处理。

* abstract_ps.py <br>
定义了AbstractPS和ConfigHistory两个类。 <br>
前者是Server和Worker的基类，定义了push和pull两个
方法和push_parser，pull_parser两个虚方法用于覆写。其中push和pull广义地看做发送和拉取，和
PS-Lite中的定义有微妙的不同。 <br>
后者是存储一个超参组及其对应的历史记录的类，用于数据的存储和转化。

* compile.sh <br>
ps_smac.cc的编译命令，需要事先把wormhole中的PS-Lite进行安装，然后把PS-Lite的目录保存到环境
变量PS_LITE中(结尾不带/)，使用compile.sh ps_smac.cc进行编译。

* ps_smac.cc <br>
PS-Lite的C++端实现，主要负责处理由标准输入输出接受Python端父进程，由管道传来的信息，并格式化为
vector形式，再传到Server/Worker端。

* server_ps.py <br>
Server的类，主要覆写了AbstractPS中的push_parser和pull_parser两个函数，用来处理SMBO
传来数据，序列化后传到Worker端。这个Server使用Popen打开了对应的PS-Lite的Server端。

* worker_ps.py <br>
同理，是Worker类，内部维护一个loop用来接收超参数/训练模型/传回给Server。

### tae/

* abstract_tae.py
ta的基类，需要继承这个AbstractTAE类来写新的ta函数。__call__已经写好，需要覆写两个set
函数来设置ConfigurationSpace(超参空间)和model的创建。如有需要，可以对___call__
函数也进行覆写。

### smbo

* smac/optimizer/smbo.py<br>
SMBO的主过程，负责创建initial_design，然后调用Server并将超参数传到Worker端，之后从
Worker端接收计算出的loss，构建EPM，计算出新的Challengers，再次传给Worker端，循环直到找到最优解。

## 使用

1. 首先安装必要的环境，包括smac的所有依赖，包括swig，ConfigSpace，使用pip等Python
包管理器进行安装。以及[wormhole](https://github.com/dmlc/wormhole)中的PS-Lite，
需要注意的是，PS-Lite需要单独make，并将编译后的路径存入环境变量PS_LITE中(结尾不带/)。

2. 使用ps目录下的compile.sh文件，对ps_smac.cc进行编译，获得ps_samc二进制文件。
使用如./compile.sh ps_smac.cc的命令。

3. 如有需要，可对tae目录下的abstract_tae.py中的AbstractTAE基类进行扩展，对其中的set_model
和set_config_space方法进行覆写，分别设置模型和搜索用的ConfigurationSpace。调用方法已经在AbstractTAE
中的__call__方法进行实现，如有需要可做进一步修改。可以参考tae目录下的logistic_regression文件。

4. 最适合修改的是test_ps.py文件，这是个基于fabric的脚本文件，将服务器信息修改即可在任何机子上调用。
里面包含了服务器的配置信息，包括ssh连接方式，输入输出文件目录，各个节点的信息。这个脚本会打开复数个ssh
连接，然后执行pssmac并由nohup挂载，实际上是将指令格式化，传输给服务器端的run_facade.py文件。

5. 如果需要对pssmac的facade进行操作，或者修改读写数据的方式，可以编辑run_facade.py文件。
我默认对.csv结尾的文件按照csv方式读取，其他文件则全部按照libsvm格式读取。数据按照seed
为1的情况分为2：1形式。如果需要修改，可以对这个文件进行编辑。
