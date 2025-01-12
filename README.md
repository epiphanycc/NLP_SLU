### 运行测试

在根目录下运行

`python scripts/lora_bert.py --testing （用于测试model_lora_8391.bin)`

`python scripts/slu_bert_train.py --testing --encoder_cell RNN（用于测试model_bert_RNN.bin）`

其余测试方式类似，注意修改python文件中load的model文件的名称

### 代码说明

+ `scripts/slu_bert_train.py`:主程序脚本,包括bert全放开（放开部分）+全连接的训练代码以及bert冻结+FNN的训练代码
+ `scripts/slu_bert_optuna.py`:optuna实现
+ `scripts/lora_bert.py`:lora实现以及训练脚本
+  `scripts/slu_bert_multihead.py` 实现bert_multihead 训练 （基于optuna）
+ `models/slu_bert_withnet.py`: 不同FNN网络实现
