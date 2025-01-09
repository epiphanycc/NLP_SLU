import torch
import torch.nn as nn
from transformers import BertModel, BertForTokenClassification
class CustomBertWithGRUForTokenClassification(BertForTokenClassification):
    def __init__(self, config, num_labels, type):
        super().__init__(config)
        default_network_config = {
            'input_size': config.hidden_size,
            'hidden_size': 256,
            'num_layers': 1,
            'batch_first': True,
            'bidirectional': False,
        }
        
        # 使用 BERT 的预训练部分
        self.bert = BertModel(config)
        
        # 冻结 BERT 的所有参数
        for param in self.bert.parameters():
            param.requires_grad = False
        
        #根据需求
        if type == 'GRU':
            self.net = nn.GRU(**default_network_config)
        elif type == 'LSTM':
            self.net = nn.LSTM(**default_network_config)
        elif type == 'RNN':
            self.net = nn.RNN(**default_network_config)

        # 自定义分类层
        self.custom_classifier = nn.Linear(default_network_config['hidden_size'], num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # 获取 BERT 的输出
        with torch.no_grad():  # 确保 BERT 在前向传播中不计算梯度
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # BERT 的最后隐藏状态
        sequence_output = outputs[0]  # 形状 [batch_size, seq_length, hidden_size]
        
        # 将 BERT 的输出传递到 GRU 层
        output, _ = self.net(sequence_output)  # gru_output 形状 [batch_size, seq_length, gru_hidden_size]
        
        # 使用 GRU 的输出作为输入到自定义分类层
        logits = self.custom_classifier(output)  # logits 形状 [batch_size, seq_length, num_labels]
        
        # 如果有标签，计算损失
        loss = None
        if labels is not None:
            # 计算损失（通常是交叉熵损失）
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, logits) if loss is not None else (logits,)

class CustomBertMultiHeadForTokenClassification(BertForTokenClassification):
    def __init__(self, config, num_labels_list, type):
        super().__init__(config)
        default_network_config = {
            'input_size': config.hidden_size,
            'hidden_size': 256,
            'num_layers': 1,
            'batch_first': True,
            'bidirectional': False,
        }
        
        # 使用 BERT 的预训练部分
        self.bert = BertModel(config)
        
        # 冻结 BERT 的所有参数
        for param in self.bert.parameters():
            param.requires_grad = False
        
        #根据需求
        if type == 'GRU':
            self.net = nn.GRU(**default_network_config)
        elif type == 'LSTM':
            self.net = nn.LSTM(**default_network_config)
        elif type == 'RNN':
            self.net = nn.RNN(**default_network_config)

        # 自定义分类层
        self.custom_classifier_0 = nn.Linear(default_network_config['hidden_size'], num_labels_list[0])
        self.custom_classifier_1 = nn.Linear(default_network_config['hidden_size'], num_labels_list[1])
        self.custom_classifier_2 = nn.Linear(default_network_config['hidden_size'], num_labels_list[2])
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # Get BERT's output with no gradient computation
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # The last hidden state from BERT
        sequence_output = outputs[0]  # Shape [batch_size, seq_length, hidden_size]
        
        # Pass BERT's output to the GRU/LSTM/RNN layer
        output, _ = self.net(sequence_output)  # Shape [batch_size, seq_length, hidden_size]
        
        # Initialize an empty list to store logits for each head
        """logits_list = []
        
        # Compute logits for each head
        for classifier in self.custom_classifier_list:
            logits = classifier(output)  # Shape [batch_size, seq_length, num_labels_for_this_head]
            logits_list.append(logits)"""
        logits_list = [
            self.custom_classifier_0(output),
            self.custom_classifier_1(output),
            self.custom_classifier_2(output)
        ]

        # Initialize the loss to None
        loss = None
        if labels is not None:
            # Compute the loss for each head
            loss_fct = nn.CrossEntropyLoss()
            loss = 0  # Initialize loss to zero
            for i, logits in enumerate(logits_list):
                # Extract labels for the i-th head
                head_labels = labels[:, :, i]  # Shape [batch_size, seq_length]
                # Calculate loss and accumulate it
                loss += loss_fct(logits.view(-1, logits.shape[-1]), head_labels.view(-1))
        
        # Return loss and logits for each head
        return (loss, logits_list) if loss is not None else (logits_list,)