## 基于BiLSTM-CRF模型的中文分词

### 运行环境

python==3.x

torch==0.4

pickle

tensorflow(optional, for tensorboard logging)

### 使用方法

训练数据与测试数据均在./data目录下

首先执行data.py预处理数据，生成vocab_tag.pkl、train_corpus.pkl和test_corpus.pkl

然后就可以直接执行train.py进行模型的训练

或者使用test.py加载预训练的模型进行分词的预测