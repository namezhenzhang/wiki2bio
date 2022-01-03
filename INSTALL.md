## INSTALL DOCUMENTATION 安装说明

#### Runtime Envrionment 运行时环境

**Python >= 3.7** 

**w/o GPU support (but if with GPU support, you should make sure you have CUDA and CUDNN version matched)**

Python 最低为3.7 版本

有无GPU支持均可，但如果有GPU支持，你需要确保CUDA、CUDNN和GPU驱动版本相符

**Packages: jittor, tqdm, numpy, tensorflow-gpu==1.15.0(for tensowflow-gpu based version)**

**Extra packages: XML-DOM, ROUGE**

**Interactive Demo: PyQt5, pyqt-tools**

相关依赖包：jittor, tqdm, numpy, tensorflow-gpu==1.15.0(对于tensorflow的gpu支持版本)

额外依赖报：XML-DOM, ROUGE

交互DEMO依赖：PyQt5, pyqt-tools

#### How To Install and Run 如何安装和运行

首先安装对应的包和依赖。

注意：在安装XML-DOM的时候可能需要很多相关权限依赖，可能需要root权限。这样方便安装。

按照 About Dataset 准备好相关数据集，放在要运行的对应版本的子目录下。

对于训练，测试不同部分有不同的运行指令。

Experiment results will be stored in the ```results/res``` directory.

**Train**

For training, turn the "mode" in ```main.py``` to ```train``` or use ``--mode train`` :

Then run ```main.py```:
```
python main.py 
```
In the training stage, the model will report BLEU and ROUGE scores on the valid set and store the model parameters after certain training steps.
The detailed results will be stored in the  ```results/res/CUR_MODEL_TIME_STAMP/log.txt```.

**Test**

For testing, turn the "mode" in ```main.py``` to ```test``` and the "load" to the selected model directory or use ``--mode test`` and ``--resume`` or ``--load``:

Then test your model by running:
```
python main.py
```

要运行测试程序，先调整为 ``test`` 模式，然后运行 ``python main.py`` 即可。

##### Demo

To run the demo, first you should install pyqt, and run (note that you should keep the main.py in test mode)

```
python gui_main.py
```

and you shoule be able to see the interactive window.

要运行 demo 程序，先安装 pyqt，然后运行指令 ``python gui_main.py`` 即可。

#### About Dataset 关于数据集

##### Intro 介绍

**The dataset for evaluation is** [WIKIBIO](https://github.com/DavidGrangier/wikipedia-biography-dataset) **from** [Lebret et al. 2016](https://arxiv.org/abs/1603.07771). **The author preprocess the dataset in a easy-to-use way.**

**The** ```original_data``` **we proprocessed can be downloaded via** [Google Drive](https://drive.google.com/file/d/15AV8LeWY3nzCKb8RRbM8kwHAp_DUZ5gf/view?usp=sharing) **or** [Baidu Yunpan](https://pan.baidu.com/s/1c324Vs8).

原数据集被Lebret 等首次采用，作者自己对数据集进行了预处理，我们需要在获取到预处理数据集以后进行二次预处理。一次预处理版本可以从 Google Drive 或 Baidu Yunpan 链接中获取。

```
original_data
training set: train.box; train.summary
testing set:  test.box; test.summary
valid set:    valid.box; valid.summary
vocabularies: word_vocab.txt; field_vocab.txt
```

```*.box``` **in the** ```original_data``` **is the infoboxes from Wikipedia. One infobox per line.**

```*.summary``` **in the** ```original_data``` **is the biographies corresponding to the infoboxes in** ```*.box```. **One biography per line.**

```word_vocab.txt``` **and** ```field_vocab.txt``` **are vocabularies for words (20000 words) and field types (1480 types), respectively.** 

**The whole dataset is divided into training set (582,659 instances, 80%), valid set (72,831 instances, 10%) and testing set (72,831 instances, 10%).**

```*.box``` 文件是从 Wikipedia 获取的 infobox，每行一个 infobox。同样，summary文件是对应infobox的summary，同样是每个人的summary一行。word_vocab和field_vocab文件是根据数据集产生的对应的词表和域类型表。

数据集按照8：1：1的比例分为了训练集，验证集和测试集。

##### Process 预处理

**The author extract words, field types and position information from the original infoboxes** ```*.box```.
**After that, we idlize the extracted words and field type according to the word vocabulary** ```word_vocab.txt``` **and field vocabulary** ```field_vocab.txt```. 

作者将词汇，域的类型和位置信息从源box文件中抽取出来，建构成word_vocab和field_vocab，并基于这两个文件进行对应的tokenize过程。

```
python preprocess.py
```

**After preprocessing, the directory structure looks like follows:**

预处理后，仓库结构应该如下所示

```
-original_data
-processed_data
  |-train
    |-train.box.pos
    |-train.box.rpos
    |-train.box.val
    |-train.box.lab
    |-train.summary.id
    |-train.box.val.id
    |-train.box.lab.id
  |-test
    |-...
  |-valid
    |-...
-results
  |-evaluation
  |-res
```

```*.box.pos```, ```*.box.rpos```, ```*.box.val```, ```*.box.lab``` **represents the word position p+, word position p-, field content and field types, respectively.**

```*.box.pos```, ```*.box.rpos```, ```*.box.val```, ```*.box.lab``` 等四个文件分别记录了每个词的左数位置，右数位置，域的内容和域的类型，具体可以参考源论文。

