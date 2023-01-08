# This script is modified from https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch/blob/master/train_eval.py
# to enable distributed training in the parameter server architecture.
# We also deploy SK-Gradient to perform gradient compression.

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm
from datetime import timedelta
from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

import ray
from SKGradient import SkGradient

PAD, CLS = '[PAD]', '[CLS]'
num_workers = 2
batch_num = 16
ZIP_RATE  = 32

def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

class Config(object):
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 5e-5
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.eval_interval = 2

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out

@ray.remote(num_gpus=0.8)
class DataWorker(object):
    def __init__(self, id) -> None:
        self.id = id
        dataset = 'THUCNews'
        self.config = Config(dataset)
        train_data, dev_data, test_data = build_dataset(self.config)
        self.train_iter = build_iterator(train_data, self.config)
        self.dev_iter = build_iterator(dev_data, self.config)
        self.test_iter = build_iterator(test_data, self.config)
        self.model = Model(self.config)
        self.compressor = SkGradient()
        self.compressor.init_compressor(self.model, ZIP_RATE=ZIP_RATE, shareHTable=True)

        self.param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        
        self.optimizer = BertAdam(self.optimizer_grouped_parameters,
                             lr=self.config.learning_rate,
                             warmup=0.05,
                             t_total=len(self.train_iter) * self.config.num_epochs)
    
    def update_sketches(self, sketches):
        self.compressor.import_sketches(sketches)

    def update_model(self, weights):
        self.model.load_state_dict(weights)

    def handout_sketches(self):
        return self.compressor.export_sketches()

    def handout_model(self):
        return self.model.state_dict()

    def update_grad(self, agg_weight):
        cur = self.model.state_dict()
        for key in agg_weight.keys():
            cur[key].data += agg_weight[key].data
        
    def train(self, last_id, current_gradient):
        if current_gradient != {}:
            # update the aggregated gradient
            agg_gradient = self.compressor.receive(current_gradient)
            self.update_grad(agg_gradient)
        
        self.model.to(self.config.device)

        # insert the old model (-1) into the sketches
        self.compressor.accumulate(self.model.state_dict(), -1)

        total_batch = 0
        self.model.train()
        last_id = self.id if last_id == -1 else (last_id + num_workers)
        flag_end = True

        for i, (trains, labels) in enumerate(self.train_iter, last_id):
            if (i - self.id) % num_workers == 0 :
                self.model.train()
                outputs = self.model(trains)
                self.model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_batch += 1
                ret_id = i
                if total_batch == batch_num:
                    flag_end = False
                    break
        
        # insert the new model (+1) into the sketches
        self.compressor.accumulate(self.model.state_dict(), 1)

        # generate compressed gradient for sending
        updated_weight = self.compressor.send()
        return updated_weight, ret_id, flag_end

    def test(self, outputmsg = True):
        self.model.eval()
        start_time = time.time()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(self.test_iter, test=True)
        if outputmsg:
            msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
            print(msg.format(test_loss, test_acc))
            print("Precision, Recall and F1-Score...")
            print(test_report)
            print("Confusion Matrix...")
            print(test_confusion)
            time_dif = get_time_dif(start_time)
            print("Time usage:", time_dif)
        return test_acc, test_loss

    def evaluate(self, data_iter, test=False):
        self.model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for texts, labels in data_iter:
                outputs = self.model(texts)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            report = metrics.classification_report(labels_all, predict_all, target_names=self.config.class_list, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(data_iter), report, confusion
        return acc, loss_total / len(data_iter)

@ray.remote(num_gpus=0.2)
class ParameterServer(object):
    def __init__(self):
        self.old_model = None
        self.config = Config('THUCNews')
        self.model = Model(self.config)
        self.compressor = SkGradient()

    def aggregate(self, *sketches):
        return self.compressor.aggregate(*sketches)


if __name__ == '__main__':
    ray.init()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    config = Config('./THUCNews')

    Workers = [DataWorker.remote(i) for i in range(num_workers)]
    PS = ParameterServer.remote()

    # synchronize sketch parameters/models across workers
    global_sketches = Workers[0].handout_sketches.remote()
    global_model = Workers[0].handout_model.remote()
    for worker in Workers[1:]:
        worker.update_sketches.remote(global_sketches)
        worker.update_model.remote(global_model)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        aggr_cnt = 0
        last_id = [-1 for _ in range(num_workers)]
        current_global_gradient = {}

        while True:
            aggr_cnt = aggr_cnt + 1
            aggr_interval_start_time = time.time()
            
            flag_end = True
            models = []
            jobs = []
            for i in range(num_workers):
                jobs.append(Workers[i].train.remote(last_id[i], current_global_gradient))
            for i in range(num_workers):
                m, id, flg = ray.get(jobs[i])
                models.append(m)
                last_id[i] = id
                flag_end = flag_end and flg
            current_global_gradient = ray.get(PS.aggregate.remote(*models))

            val_acc, val_loss = ray.get(Workers[0].test.remote(outputmsg = False))
            elapsed = time.time() - aggr_interval_start_time

            print_msg = f'epoch: {epoch:3d} | turn: {aggr_cnt:3d} | time: {elapsed:5.2f}s | valid loss: {val_loss:5.2f} | valid acc: {val_acc:5.4f}'
            print(print_msg)

            if flag_end:
                break
        
        val_acc, val_loss = ray.get(Workers[0].test.remote())
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f}')
        print('-' * 89)

    ray.shutdown()
