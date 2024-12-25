# coding: utf-8

import sys, os, time, gc, json
import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertForTokenClassification
from transformers import get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD

# Initialization params, output path, logger, random seed, and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

# Load datasets
start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

# Load pre-trained BERT model and tokenizer
bert_model_path = os.path.abspath(os.path.join(install_path, 'bert-base-chinese'))
tokenizer = BertTokenizer.from_pretrained(bert_model_path)
model = BertForTokenClassification.from_pretrained(
    bert_model_path,
    num_labels=Example.label_vocab.num_tags,
).to(device)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=8,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

if args.testing:
    checkpoint = torch.load(open('model_lora.bin', 'rb'), map_location=device)
    model.load_state_dict(checkpoint['model'])
    print("Loaded saved model from root path")

def set_optimizer(model, args):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * ((len(train_dataset) // args.batch_size) * args.max_epoch)),
        num_training_steps=((len(train_dataset) // args.batch_size) * args.max_epoch),
    )
    return optimizer, scheduler

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
def decode(choice):
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
            loss, logits = outputs.loss, outputs.logits
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

def predict():
    
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
if not args.testing:
    writer = SummaryWriter(log_dir="runs/slu_experiment_lora")
    optimizer, scheduler = set_optimizer(model, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training with LoRA ......')
    for i in range(args.max_epoch):
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
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            count += 1

        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        writer.add_scalar('Loss/dev', dev_loss, i)
        writer.add_scalar('Accuracy/dev', dev_acc, i)
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)\tDev loss: %.4f' % (
            i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore'], dev_loss))

        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('model.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))