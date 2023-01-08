import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time
import ray

from SKGradient import SkGradient
from model.vgg import VGG

# log paprameters
PRINT_INTERVAL = 1

# algo parameters
ZIP_RATE   = 32
TOTAL_ITER = 16384
BATCH_NUM  = 32
WORKER_NUM = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# configs
criterion = nn.CrossEntropyLoss()

def get_data_loader():

    cifar10_transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    cifar10_transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=cifar10_transform_train,
        ), 
        batch_size=128,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "./data", 
            train=False,
            transform=cifar10_transform_test
        ),
        batch_size=100,
        shuffle=False
    )

    return train_loader, test_loader
    
def evaluate(model, test_loader):

    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            loss += criterion(outputs, target).item()
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if total >= 3200:
                break
    
    return loss/total, correct/total

@ray.remote(num_gpus=0.7)
class DataWorker(object):

    def __init__(self, i):
        self.id = i
        self.iter_cnt = 0
        self.batch_num = BATCH_NUM
        self.model = VGG('VGG19')
        self.compressor = SkGradient()
        self.compressor.init_compressor(self.model, ZIP_RATE=ZIP_RATE, shareHTable=True)
        self.data_loader = get_data_loader()[0]
        self.test_loader = get_data_loader()[1]
        self.lst = list(batch for batch in self.data_loader)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def update_sketches(self, sketches):
        self.compressor.import_sketches(sketches)

    def update_model(self, weights):
        self.model.set_weights(weights)

    def handout_sketches(self):
        return self.compressor.export_sketches()

    def handout_model(self):
        return self.model.get_weights()

    def eval(self):
        return evaluate(self.model, self.test_loader)

    def update_grad(self, agg_weight):
        cur = self.model.state_dict()
        for key in agg_weight.keys():
            if cur[key].dtype == torch.int64: continue
            cur[key].data += agg_weight[key].data
    
    def compute(self, current_gradient):
        if current_gradient != {}:
            # update the aggregated gradient
            agg_gradient = self.compressor.receive(current_gradient)
            self.update_grad(agg_gradient)
            
        self.model = self.model.to(device)

        # insert the old model (-1) into the sketches
        self.compressor.accumulate(self.model.state_dict(), -1)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.model.train()

        start_point = int(((len(self.data_loader))//WORKER_NUM)*self.id +self.iter_cnt*self.batch_num)%len(self.data_loader)
        for batch_idx, (data, target) in enumerate(self.lst[start_point:]+self.lst[:start_point]):
            data = data.to(device)
            target = target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx >= self.batch_num - 1: break

        self.iter_cnt += 1

        # insert the new model (+1) into the sketches
        self.compressor.accumulate(self.model.state_dict(), 1)
        
        # generate compressed gradient for sending
        updated_weight = self.compressor.send()
        return updated_weight

@ray.remote(num_gpus=0.2)
class ParameterServer:
    def __init__(self):
        self.compressor = SkGradient()
    def aggregate(self, *sketches):
        return self.compressor.aggregate(*sketches)

if __name__ == '__main__':

    iterations = int(TOTAL_ITER / BATCH_NUM)

    ray.init()
    ps = ParameterServer.remote()
    workers = [DataWorker.remote(i) for i in range(WORKER_NUM)]

    print("Running synchronous parameter server training.")
    current_weight = {}

    # synchronize sketch parameters/models across workers
    global_sketches = workers[0].handout_sketches.remote()
    global_model = workers[0].handout_model.remote()
    for worker in workers[1:]:
        worker.update_sketches.remote(global_sketches)
        worker.update_model.remote(global_model)

    for i in range(iterations):
        print('Synchronization:', i)
        st = time.time()

        sketches = [worker.compute.remote(current_weight) for worker in workers]
        current_weight = ray.get(ps.aggregate.remote(*sketches))

        if i % PRINT_INTERVAL == 0:
            loss, accuracy = ray.get(workers[0].eval.remote())
            et = time.time()
            print("iteration: {}, acc: {:.4f}%, loss: {:.4f}, time: {:.4f}".format(int(i*BATCH_NUM), accuracy*100, loss, (et-st)/BATCH_NUM))

    ray.shutdown()