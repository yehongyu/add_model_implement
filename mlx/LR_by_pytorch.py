import os
import subprocess
import torch
import torchvision.datasets as d
import torchvision.transforms as t

from torch.autograd import Variable
from torch.nn import functional as F
from petastorm import make_batch_reader, make_reader
from petastorm.pytorch import DataLoader

from mlx.operators.operator import Operator



class LG(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LG, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class LogisticRegression(Operator):

    def run(self):
        input_dim = int(self.ctx.config['feature_dim'])
        output_dim = int(self.ctx.config['class_num'])
        train_data = self.ctx.inputs['train'].strip()
        test_data = self.ctx.inputs['test'].strip()
        print('train data: {}'.format(train_data))
        model = LG(input_dim, output_dim)

        loss = torch.nn.CrossEntropyLoss()
        lr = 0.001
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        iter = 0
        for epoch in range(int(self.ctx.config.get('epoch', 1))):
            with make_batch_reader(train_data, num_epochs=1, hdfs_driver='libhdfs') as reader:
                with DataLoader(reader) as train_loader:
                    for i, (data) in enumerate(train_loader):
                        # print(i)
                        images = Variable(data['features'].float())
                        # print(len(labels))
                        labels = Variable(data['label'].long())
                        optimizer.zero_grad()
                        outputs = model(images)
                        l = loss(outputs, labels)
                        l.backward()
                        optimizer.step()
                        iter += 1
                        total = 0
                        correct = 0
                        if iter % 500 == 0:
                            with make_batch_reader(test_data, num_epochs=1, hdfs_driver='libhdfs') as test_reader:
                                with DataLoader(test_reader) as test_loader:
                                    for data in test_loader:
                                        images = Variable(data['features'].float())
                                        outputs = model(images)
                                        _, predicted = torch.max(outputs.data, 1)
                                        total += labels.size(0)
                                        correct += (predicted == data['label']).sum()

                                    accuracy = 100 * correct / total
                                    print("Epoch: {}, iter {}, accuracy: {}%, loss: {}".format(epoch, iter, accuracy, l.item()))
                        if iter == 5000:
                            break
        model_name = 'model_{}'.format(os.environ.get('MLX_EXP_RUN_ID', 0))
        saved_model_path = self.ctx.config['saved_model_path']
        print('Save model and upload to hdfs: {}'.format(saved_model_path))
        torch.save(model.state_dict(), './{}'.format(model_name))
        subprocess.check_output('/opt/tiger/yarn_deploy/hadoop/bin/hadoop fs -put -f {} {}'.format(model_name, saved_model_path), shell=True)
        # saved_model = '{}/{}'.format(saved_model_path, model_name)
        self.ctx.outputs['model'] = saved_model_path

if __name__ == '__main__':
    op = LogisticRegression()
    op.run()
    op.finalize()
