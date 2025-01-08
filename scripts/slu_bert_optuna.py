import sys, os, time, gc, json
from torch.optim import AdamW
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.vocab import PAD
from torch.utils.tensorboard import SummaryWriter
import optuna
from transformers import get_scheduler
# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))



    

def encode_batch(dataset, tokenizer, max_len):
    input_ids, attention_masks, labels = [], [], []
    for example in dataset:
        tokens = tokenizer(example.utt, truncation=True, padding='max_length', max_length=max_len, return_tensors="pt")
        label_ids = [Example.label_vocab.convert_tag_to_idx(tag) for tag in example.tags]
        label_ids = label_ids[:max_len] + [Example.label_vocab.convert_tag_to_idx(PAD)] * (max_len - len(label_ids))
        input_ids.append(tokens["input_ids"][0])
        attention_masks.append(tokens["attention_mask"][0])
        labels.append(torch.tensor(label_ids, dtype=torch.long))
    return pad_sequence(input_ids, batch_first=True), pad_sequence(attention_masks, batch_first=True), pad_sequence(labels, batch_first=True)

def decode(choice,model,tokenizer):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            input_ids, attention_masks, label_ids = encode_batch(cur_dataset, tokenizer, args.max_seq_len)
            input_ids, attention_masks, label_ids = input_ids.to(device), attention_masks.to(device), label_ids.to(device)

            outputs = model(input_ids, attention_mask=attention_masks, labels=label_ids)
            if not args.eztrain:
                loss,logits = outputs.loss,outputs.logits
            else:
                loss,logits = outputs[0] ,outputs[1]
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(label_ids.cpu().tolist())
            total_loss += loss.item()
            count += 1

        # 将预测的标签 ID 转换为 act-slot-value 三元组（value并非准确值，但不影响计算准确性和分数）
        converted_predictions = []
        for pred in predictions:
            triplets = []
            current_slot = None
            current_value = []
            for idx in pred:
                label = Example.label_vocab.convert_idx_to_tag(idx)
                if label.startswith("B-"):
                    if current_slot is not None:
                        triplet_str = f"{current_slot}-{''.join(current_value)}"
                        triplets.append(triplet_str)
                    current_slot = label[2:]
                    current_value = [label.split("-")[-1]]
                elif label.startswith("I-"):
                    current_value.append(label.split("-")[-1])
                elif label == "O" and current_slot is not None:
                    triplet_str = f"{current_slot}-{''.join(current_value)}"
                    triplets.append(triplet_str)
                    current_slot = None
                    current_value = []
            if current_slot is not None:
                triplet_str = f"{current_slot}-{''.join(current_value)}"
                triplets.append(triplet_str)

            converted_predictions.append(triplets)

        # 将真实标签 label_ids 也转换为 act-slot-value 三元组
        converted_labels = []
        for label in labels:
            triplets = []
            current_slot = None
            current_value = []
            for idx in label:
                label_name = Example.label_vocab.convert_idx_to_tag(idx)
                if label_name.startswith("B-"):
                    if current_slot is not None:
                        triplet_str = f"{current_slot}-{''.join(current_value)}"
                        triplets.append(triplet_str)
                    current_slot = label_name[2:]
                    current_value = [label_name.split("-")[-1]]
                elif label_name.startswith("I-"):
                    current_value.append(label_name.split("-")[-1])
                elif label_name == "O" and current_slot is not None:
                    triplet_str = f"{current_slot}-{''.join(current_value)}"
                    triplets.append(triplet_str)
                    current_slot = None
                    current_value = []
            if current_slot is not None:
                triplet_str = f"{current_slot}-{''.join(current_value)}"
                triplets.append(triplet_str)

            converted_labels.append(triplets)

        metrics = Example.evaluator.acc(converted_predictions, converted_labels)

    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count

def predict(model,tokenizer):
    
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = {}
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            input_ids, attention_masks, labels = encode_batch(cur_dataset, tokenizer, args.max_seq_len)
            input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)

            logits = model(input_ids, attention_mask=attention_masks).logits
            pred = torch.argmax(logits, dim=-1).cpu().tolist()
            
            input_ids = input_ids.cpu().tolist()  # 转换 input_ids 为 CPU 列表
            for pi, (p, input_id) in enumerate(zip(pred, input_ids)):
                did = cur_dataset[pi].did
                tokens = tokenizer.convert_ids_to_tokens(input_id)  # 将 ID 转换为 token
                tokens = [token for token in tokens if token != "[PAD]"]  # 去掉 [PAD] token
                # 去掉 [CLS] 和 [SEP]
                if "[CLS]" in tokens:
                    tokens = tokens[1:]
                if "[SEP]" in tokens:
                    tokens = tokens[:-1]

                label_seq = [Example.label_vocab.convert_idx_to_tag(idx) for idx in p[:len(tokens)]]

                # 将预测的标签序列转换为 act-slot-value
                triplets = []
                current_act = None
                current_slot = None
                current_value = []

                for idx, label in enumerate(label_seq):
                    if label.startswith("B-"):
                        if current_act and current_slot:
                            triplets.append([current_act, current_slot, ''.join(current_value)])
                        parts = label[2:].split('-')
                        if len(parts) == 2:
                            current_act, current_slot = parts
                            current_value = [tokens[idx]]
                    elif label.startswith("I-") and current_act and current_slot:  # 继续当前的 act-slot
                        current_value.append(tokens[idx])
                    elif label == "O" and current_act and current_slot:  # 结束当前 act-slot
                        triplets.append([current_act, current_slot, ''.join(current_value)])
                        current_act, current_slot = None, None
                        current_value = []

                # 确保最后一个 act-slot-value 被处理
                if current_act and current_slot:
                    triplets.append([current_act, current_slot, ''.join(current_value)])

                predictions[did] = triplets

    # 将预测结果保存到测试数据中
    test_json = json.load(open(test_path, 'r', encoding='utf-8'))
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt_id = f"{ei}-{ui}"
            if utt_id in predictions:
                utt['pred'] = predictions[utt_id]

    output_path = os.path.join(args.dataroot, 'prediction.json')
    json.dump(test_json, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

def freeze_bert_layers(model, freeze_layers=10):
    """
    冻结 BERT 模型的前 `freeze_layers` 层的参数
    """
    # 遍历 BERT 编码器的所有层
    for name, param in model.bert.encoder.layer.named_parameters():
        layer_idx = int(name.split('.')[0])  # 获取层编号
        if layer_idx < freeze_layers:
            param.requires_grad = False  # 冻结参数
        else:
            param.requires_grad = True   # 保留可训练参数

    # 冻结 BERT embedding 层（通常不需要微调）
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

def set_optimizer(model, lr, weight_decay, ratio):
    import torch.optim as optim
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = AdamW(grouped_params, lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(ratio * ((len(train_dataset) // args.batch_size) * args.max_epoch)),
        num_training_steps=((len(train_dataset) // args.batch_size) * args.max_epoch),
    )
    return optimizer, scheduler

# 目标函数：定义需要优化的超参数和模型训练过程
def objective(trial):
    """
    目标函数：定义需要优化的超参数和模型训练过程
    """
    lr = trial.suggest_loguniform('lr', 5e-6, 3e-2)  # 学习率范围
    # freeze_layers = trial.suggest_int('freeze_layers', 8, 8)  # 冻结的 BERT 层数
    batch_size = trial.suggest_int('batch_size', 8, 48, step=8)  # 批次大小
    max_epoch = trial.suggest_int('max_epoch', 10, 25)  # 最大训练 epoch 数
    # step_size = trial.suggest_int('step_size', 5, 20)  # 调整 step_size 的值
    # gamma = trial.suggest_uniform('gamma', 0.1, 0.9)  # 调整 gamma 的值
    ratio = trial.suggest_uniform('gamma', 0, 0.2)  # 调整 ratio 的值
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    
    
    # Load pre-trained BERT model and tokenizer
    bert_model_path = os.path.abspath(os.path.join(install_path, 'bert-base-chinese'))
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    if args.eztrain == True:
        config = BertConfig.from_pretrained("bert-base-chinese")
        from model.slu_bert_withnet import CustomBertWithGRUForTokenClassification
        model = CustomBertWithGRUForTokenClassification(config, num_labels = Example.label_vocab.num_tags, type = args.encoder_cell).to(device)
    else:
        model = BertForTokenClassification.from_pretrained(
        bert_model_path, num_labels=Example.label_vocab.num_tags).to(device)
        
    # 根据 Optuna 中的超参数冻结层
    freeze_bert_layers(model, freeze_layers=8)

    writer = SummaryWriter(log_dir="runs/slu_experiment_optuna")
    nsamples, step_size = len(train_dataset), batch_size
    best_result = {'dev_acc': 0., 'dev_f1': 0.}

    # 训练模型
    if not args.testing:

        writer = SummaryWriter(log_dir="runs/slu_experiment")
        num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
        ###
        print('Total training steps: %d' % (num_training_steps))
        optimizer, scheduler = set_optimizer(model, lr ,weight_decay,ratio)
        nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
        train_index, step_size = np.arange(nsamples), args.batch_size
        print('Start training ......')
        loss_cnt = 0
        for i in range(max_epoch):
            start_time = time.time()
            epoch_loss = 0
            np.random.shuffle(train_index)
            model.train()
            count = 0
            for j in range(0, nsamples, step_size):
                cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
                input_ids, attention_masks, label_ids = encode_batch(cur_dataset, tokenizer, args.max_seq_len)
                input_ids, attention_masks, label_ids = input_ids.to(device), attention_masks.to(device), label_ids.to(device)
                
                outputs = model(input_ids, attention_mask=attention_masks, labels=label_ids)
                if not args.eztrain:
                    loss = outputs.loss
                else:
                    loss = outputs[0]
                
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                writer.add_scalar('Loss/train', loss , loss_cnt)
                loss_cnt += 1
                optimizer.step()
                count += 1
            print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
            torch.cuda.empty_cache()
            gc.collect()

            start_time = time.time()
            metrics, dev_loss = decode('dev',model,tokenizer)
            scheduler.step()
            dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
            writer.add_scalar('Loss/dev', dev_loss, i)
            writer.add_scalar('Accuracy/dev', dev_acc, i)
            writer.add_scalar('F1/dev', dev_fscore['fscore'], i)
            print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)\tDev loss: %.4f' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore'],dev_loss))
            if dev_acc > best_result['dev_acc']:
                best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
                torch.save({
                    'epoch': i, 'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, open('model.bin', 'wb'))
                print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

        print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    return best_result['dev_acc']


# 创建 Optuna study
study = optuna.create_study(direction='maximize')

# 运行优化
study.optimize(objective, n_trials=10)  # 试验次数（根据需要调整）

# 输出优化结果
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value}")
print(f"Best parameters: {study.best_params}")


optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_slice(study).show()
