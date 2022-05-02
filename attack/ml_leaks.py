import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import time

torch.manual_seed(0)
np.random.seed(0)

class SoftmaxModel(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_out)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        return x

class DefenceSoftmaxModel(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_out)


    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        return x


def train_attack_model(trainloader, n_in, n_out, epochs=500, lr=0.01, momentum=0.9, weight_decay=1e-7):
    net = DefenceSoftmaxModel(n_in, n_out)
    net = net.cuda()
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0.0
        for batch_idx, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.long().cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct += (torch.argmax(outputs, axis=1) == labels).float().mean()
        accuracy = 100 * correct / len(trainloader)
        if epoch % 50 == 0:
            print('[epoch {}/{}] accuracy {}'.format(epoch, epochs, accuracy))

    print('Finished Training')
    return net

def eval_attack_model(attack_model, testloader, prefix='Overall'):
    """
    Evaluate the model on the target member and non-member set
    """
    attack_model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs = inputs.cuda()
            pred = attack_model(inputs)
            predictions.append(torch.argmax(pred, axis=1))
            ground_truth.append(labels)

        predictions = torch.cat(predictions).cpu().numpy()
        ground_truth = torch.cat(ground_truth).cpu().numpy()
        print(predictions.shape, ground_truth.shape)
        print('{} Testing accuracy: {}'.format(prefix, accuracy_score(predictions, ground_truth)))

class Mydataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.labels)


def generate_attack_loader(model, member_loader, non_member_loader, w_label, limit, max_output_shape=10):
    """
    """
    attack_feat = []
    attack_labels = []

    i = 0
    with torch.no_grad():
        for (inputs, labels) in member_loader:
            inputs = inputs.cuda()
            outputs = model(inputs).cpu().numpy()
            # take only top10 (for more classes, the attack model gets too confused)
            if outputs.shape[1] > max_output_shape:
                top10 = np.flip(np.sort(outputs, axis=1)[:,-max_output_shape:], axis=1)
                outputs = top10
            if w_label:
                labels = labels.numpy()
                reshaped = labels.reshape((labels.shape[0], 1))
                outputs = np.concatenate((outputs, reshaped), axis=1)
            attack_feat.append(outputs)
            attack_labels.append(np.ones(labels.shape[0]))

        for (inputs, labels) in non_member_loader:
            if i >= limit:
                break
            i += inputs.shape[0]
            inputs = inputs.cuda()
            outputs = model(inputs).cpu().numpy()
            # take only top10 (for more classes, the attack model gets too confused)

            if outputs.shape[1] > max_output_shape:
                top10 = np.flip(np.sort(outputs, axis=1)[:,-max_output_shape:], axis=1)
                outputs = top10
            if w_label:
                labels = labels.numpy()
                reshaped = labels.reshape((labels.shape[0], 1))
                outputs = np.concatenate((outputs, reshaped), axis=1)
            attack_feat.append(outputs)
            attack_labels.append(np.zeros(labels.shape[0]))

    attack_feat = np.vstack(attack_feat)
    attack_labels = np.concatenate(attack_labels)
    attack_feat = attack_feat.astype('float32')
    attack_labels = attack_labels.astype('int32')
    print('attack_feat shape {}, attack_labels shape {}'.format(attack_feat.shape, attack_labels.shape))
    return attack_feat, attack_labels

def generate_attack_loader_gpu(model, member_loader, non_member_loader, w_label, limit,  max_output_shape=10):
    """
    """
    attack_feat = []
    attack_labels = []

    i = 0
    i_train = 0
    with torch.no_grad():
        for (inputs, labels) in member_loader:
            if i_train >= limit:
                break
            i_train += inputs.shape[0]
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            # take only top10 (for more classes, the attack model gets too confused)
            if outputs.shape[1] > max_output_shape:
                top10_values, top10_indices = torch.sort(outputs, descending=True, dim=1)
                outputs = top10_values[:,:max_output_shape]
            if w_label:
                reshaped = torch.reshape(labels, (labels.shape[0], 1))
                outputs = torch.cat((outputs, reshaped), axis=1)
            attack_feat.append(outputs)
            attack_labels.append(torch.ones(labels.shape[0]))

        for (inputs, labels) in non_member_loader:
            if i >= limit:
                break
            i += inputs.shape[0]
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            if outputs.shape[1] > max_output_shape:
                top10_values, top10_indices = torch.sort(outputs, descending=True, dim=1)
                outputs = top10_values[:,:max_output_shape]
            if w_label:
                reshaped = torch.reshape(labels, (labels.shape[0], 1))
                outputs = torch.cat((outputs, reshaped), axis=1)
            attack_feat.append(outputs)
            attack_labels.append(torch.zeros(labels.shape[0]))

    attack_feat = torch.vstack(attack_feat)
    attack_labels = torch.cat(attack_labels)
    #attack_feat = attack_feat.cpu().numpy().astype('float32')
    #attack_labels = attack_labels.cpu().numpy().astype('int32')
    return attack_feat, attack_labels

def attack(target_model, shadow_model, member_shadow_loader, non_member_shadow_loader,
           member_target_loader, non_member_target_loader, w_label, test_sz, attack_model_name, max_output_shape=10):
    """
    """
    # Need to create the trainX, trainY

    start_time = time.time()
    attack_feat, attack_labels = generate_attack_loader_gpu(shadow_model, member_shadow_loader,
                                                            non_member_shadow_loader, w_label, test_sz, max_output_shape)
    print("Attack with top %d prediction score"%(max_output_shape))
    print('[time] Create attack features and labels: {} sec'.format(time.time() - start_time))
    n_in = attack_feat.shape[1]
    n_out = len(np.unique(attack_labels))

    print(n_in, n_out)
    start_time = time.time()
    trainloader = DataLoader(Mydataset(attack_feat, attack_labels), batch_size=100, shuffle=True,)
    print('[time] Create dataloader for attack features and labels: {} sec'.format(time.time() - start_time))

    start_time = time.time()
    attack_model = train_attack_model(trainloader, n_in, n_out)
    print('[time] Training: {} sec'.format(time.time() - start_time))
    torch.save(attack_model, attack_model_name)

    start_time = time.time()
    attack_feat, attack_labels = generate_attack_loader_gpu(target_model, member_target_loader,
                                                            non_member_target_loader, w_label, test_sz, max_output_shape)
    print('[time] Create eval attack features and labels: {} sec'.format(time.time() - start_time))

    start_time = time.time()
    testloader = DataLoader(Mydataset(attack_feat, attack_labels), shuffle=False, batch_size=100)
    print('[time] Create test dataloader for attack features and labels: {} sec'.format(time.time() - start_time))

    start_time = time.time()
    eval_attack_model(attack_model, testloader)
    print('[time] Testing: {} sec'.format(time.time() - start_time))

    start_time = time.time()
    # Evaluate the attack model on members only
    mem_attack_feat, mem_attack_labels = generate_attack_loader_gpu(target_model, member_target_loader,
                                                                    [], w_label, test_sz, max_output_shape)
    mem_testloader = DataLoader(Mydataset(mem_attack_feat, mem_attack_labels), shuffle=False)
    eval_attack_model(attack_model, mem_testloader, prefix='Member')
    print('[time] Evaluate model on members: {} sec'.format(time.time() - start_time))

    start_time = time.time()
    nm_attack_feat, nm_attack_labels = generate_attack_loader_gpu(target_model, [],
                                                                  non_member_target_loader, w_label,
                                                                  test_sz, max_output_shape)
    nm_testloader = DataLoader(Mydataset(nm_attack_feat, nm_attack_labels), shuffle=False)
    eval_attack_model(attack_model, nm_testloader, prefix='Non-member')
    print('[time] Create non-members for evaluation: {} sec'.format(time.time() - start_time))

