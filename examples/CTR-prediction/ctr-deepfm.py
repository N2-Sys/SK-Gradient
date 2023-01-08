import pandas as pd
import numpy as np
import time
import ray
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from SKGradient import SkGradient
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
target = ['label']

def load_data(path):
    data = pd.read_csv(path)

    data[sparse_features] = data[sparse_features].fillna('-1',)
    data[dense_features] = data[dense_features].fillna('0',)

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    return data

@ray.remote(num_gpus=0.7)
class DataWorker(object):

    def __init__(self, id) -> None:
        self.train_data = load_data('./data/train.csv')
        self.valid_data = load_data('./data/valid.csv')
        self.test_data = load_data('./data/test.csv')
        self.id = id
        self.batch_size = 128
        self.batch_num = 8
        self.start_point = 0
        self.device = 'cuda'

        fixlen_feature_columns = [SparseFeat(feat, (self.train_data[feat].nunique()+16), embedding_dim=1000)
                for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]
        
        self.feature_names = get_feature_names(fixlen_feature_columns)
        self.model = DeepFM(linear_feature_columns=fixlen_feature_columns, dnn_feature_columns=fixlen_feature_columns,
                   task='binary', l2_reg_embedding=1e-5, device=self.device)
        self.model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])

        self.train_loader = self.prepare_data(self.train_data)
        self.valid_loader = self.prepare_data(self.valid_data)
        self.optim = torch.optim.Adagrad(self.model.parameters())

        self.compressor = SkGradient()
        self.compressor.init_compressor(self.model, 32, threshold=2**15, device='cuda')
        self.model = self.model.cuda()

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.model.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def prepare_data(self, original_data):
        x = {name: original_data[name] for name in self.feature_names}
        x = [x[feature] for feature in self.model.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        y = original_data[target].values

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)), torch.from_numpy(y))
        train_loader = Data.DataLoader(dataset=train_tensor_data, 
            shuffle=True, batch_size=self.batch_size)
        
        return train_loader

    def train(self, weight):
        loss_func = F.binary_cross_entropy
        
        # receive and decompress the aggregated gradient
        if weight != {}:
            agg_weight = self.compressor.receive(weight)
            self.update_model(agg_weight)
        
        # insert the old model (-1) into the sketches
        self.compressor.accumulate(self.model.state_dict(), -1)
        
        self.model = self.model.train()
        print("Start training from batch #"+str(self.start_point)+"...")
        batch_cnt = 0

        train_start = time.time()
        for _, (x_train, y_train) in enumerate(self.train_loader, start=self.start_point):
            x = x_train.to(self.device).float()
            y = y_train.to(self.device).float() 

            y_pred = self.model(x).squeeze()
                    
            self.optim.zero_grad()

            loss = loss_func(y_pred, y.squeeze(), reduction='sum')
            reg_loss = self.get_regularization_loss()
            total_loss = loss + reg_loss + self.model.aux_loss

            total_loss.backward()
            self.optim.step()
                
            batch_cnt += 1
            if batch_cnt == self.batch_num:
                self.start_point = (self.start_point + self.batch_num) % len(self.train_loader)
                break
        train_end = time.time()

        print("[Log] Training time (per Batch) =", (train_end-train_start)/self.batch_num)

        # insert the new model (+1) into the sketches
        self.compressor.accumulate(self.model.state_dict(), 1)

        # generate compressed gradient for sending
        local_weight = self.compressor.send()
        return local_weight
        
    def evaluate(self, eval_loader=None):
        self.model = self.model.eval()

        if eval_loader == None:
            eval_loader = self.valid_loader
        
        auc = 0
        for _, (x_val, y_val) in enumerate(eval_loader):
            x_val = x_val.to(self.device).float()
            y_val = y_val.to(self.device).float()
            y_pred = self.model(x_val).squeeze()
            auc += roc_auc_score(y_val.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64"))
        auc = auc / len(eval_loader)
        print("Evaluate completed! [AUC] =", auc)

        return auc

    def export_origin_model(self):
        return self.model

    def export_model(self):
        return self.model.state_dict()

    def import_model(self, model):
        self.model.load_state_dict(model)
    
    def update_model(self, agg_weight):
        cur = self.model.state_dict()
        for key in agg_weight.keys():
            if cur[key].dtype == torch.int64: continue
            cur[key].data += agg_weight[key].data
    
    def export_sketch(self):
        return self.compressor.export_sketches()

    def import_sketch(self, sketches):
        return self.compressor.import_sketches(sketches)

@ray.remote(num_gpus=0.2)
class ParameterServer(object):
    def __init__(self):
        # decompression and recompression are no longer required
        self.compressor = SkGradient()
        return

    def aggregate(self, *sketches):
        # exploit the linearity property of FGC sketch
        return self.compressor.aggregate(*sketches)

if __name__ == '__main__':

    batch_num = 8
    num_iterations = int(2048/batch_num)
    num_workers = 8

    ray.init()
    ps = ParameterServer.remote()
    workers = [DataWorker.remote(i) for i in range(num_workers)]

    global_model = ray.get(workers[0].export_model.remote())
    for i in range(1, num_workers):
        ray.get(workers[i].import_model.remote(global_model))

    # synchronize sketch parameters across workers
    global_sketch = ray.get(workers[0].export_sketch.remote())
    for i in range(1, num_workers):
        ray.get(workers[i].import_sketch.remote(global_sketch))

    agg_weight = {}
    for i in range(num_iterations):
        models = [worker.train.remote(agg_weight) for worker in workers]
        agg_weight = ray.get(ps.aggregate.remote(*models))
        auc = ray.get(workers[0].evaluate.remote())

    ray.shutdown()
