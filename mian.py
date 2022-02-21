import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import torch
import time

import sys
sys.path.append("E:\HSI_Classification\WFCG\GNN")
sys.path.append("E:\HSI_Classification\WFCG\Load")
import WFCG
import utils
import data_reader
import dr_slic
import create_graph
import split_data_graph as split_data


# load data
def load_data():
    data = data_reader.IndianRaw().normal_cube
    data_gt = data_reader.IndianRaw().truth
    data_gt = data_gt.astype('int')
    return data, data_gt

data, data_gt = load_data()
class_num = np.max(data_gt)
height, width, bands = data.shape
gt_reshape = np.reshape(data_gt, [-1])
# cmap = cm.get_cmap('jet', class_num + 1)
# plt.set_cmap(cmap) 
samples_type = ['ratio', 'same_num'][0]
train_ratio = 0.01  
val_ratio = 0.01  
train_num = 10   
val_num = class_num
learning_rate = 1e-3  
max_epoch = 300   
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
superpixel_scale = 100
dataset_name = "indian_" 
path_model = r"E:\HSI_Classification\WFCG\model\\"
path_data = None
height, width, bands = data.shape  

# split data
train_index, val_index, test_index = \
                    split_data.split_data(gt_reshape, class_num, train_ratio, 
                    val_ratio, train_num, val_num, samples_type)
train_samples_gt, test_samples_gt, val_samples_gt = \
                    create_graph.get_label(gt_reshape, train_index, val_index, test_index)

train_gt = np.reshape(train_samples_gt,[height,width])
test_gt = np.reshape(test_samples_gt,[height,width])
val_gt = np.reshape(val_samples_gt,[height,width])

# label transfer to one-hot encode
train_samples_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
test_samples_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
val_samples_gt_onehot = create_graph.label_to_one_hot(val_gt, class_num)

train_samples_gt_onehot = np.reshape(train_samples_gt_onehot,[-1,class_num]).astype(int)
test_samples_gt_onehot = np.reshape(test_samples_gt_onehot,[-1,class_num]).astype(int)
val_samples_gt_onehot = np.reshape(val_samples_gt_onehot,[-1,class_num]).astype(int)

train_label_mask, test_label_mask, val_label_mask = \
                    create_graph.get_label_mask(train_samples_gt, test_samples_gt, val_samples_gt, data_gt, class_num)

ls = dr_slic.LDA_SLIC(data, np.reshape(train_samples_gt,[height,width]), class_num-1)
tic0=time.time()
Q, S ,A, Seg= ls.simple_superpixel(scale=superpixel_scale)
toc0 = time.time()
LDA_SLIC_Time=toc0-tic0
# print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))

# S. superpixel set
# Q. association matrix
# A. Adjacent matrix
# S. segments

Q=torch.from_numpy(Q).to(device)
A=torch.from_numpy(A).to(device)

train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

net_input=np.array(data, np.float32)
net_input=torch.from_numpy(net_input.astype(np.float32)).to(device)
net = WFCG.WFCG(height, width, bands, class_num, Q, A).to(device)

# 训练
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate, weight_decay=0.001)  
zeros = torch.zeros([height * width]).to(device).float()
best_loss=99999
net.train()
tic1 = time.time()
for i in range(max_epoch+1):
    optimizer.zero_grad()  # zero the gradient buffers
    output= net(net_input)
    loss = utils.compute_loss(output,train_samples_gt_onehot,train_label_mask)
    loss.backward(retain_graph=False)
    optimizer.step()  # Does the update
    
    # if i%10==0:
    with torch.no_grad():
        net.eval()
        output= net(net_input)
        trainloss = utils.compute_loss(output, train_samples_gt_onehot, train_label_mask)
        trainOA = utils.evaluate_performance(output, train_samples_gt, train_samples_gt_onehot, zeros)
        valloss = utils.compute_loss(output, val_samples_gt_onehot, val_label_mask)
        valOA = utils.evaluate_performance(output, val_samples_gt, val_samples_gt_onehot, zeros)
        if valloss < best_loss :
            best_loss = valloss
            torch.save(net.state_dict(), path_model + r"model.pt")
    torch.cuda.empty_cache()
    net.train()

    if i%10==0:
        print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss, trainOA, valloss, valOA))
toc1 = time.time()

print("\n\n====================training done. starting evaluation...========================\n")
torch.cuda.empty_cache()
with torch.no_grad():
    net.load_state_dict(torch.load(path_model + r"model.pt"))
    net.eval()
    tic2 = time.time()
    output = net(net_input)
    toc2 = time.time()
    testloss = utils.compute_loss(output, test_samples_gt_onehot, test_label_mask)
    testOA = utils.evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, zeros)
    print("{}\ttest loss={:.4f}\t test OA={:.4f}".format(str(i + 1), testloss, testOA))
    
torch.cuda.empty_cache()
del net

LDA_SLIC_Time=toc0-tic0
print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
training_time = toc1 - tic1 + LDA_SLIC_Time #分割耗时需要算进去
testing_time = toc2 - tic2 + LDA_SLIC_Time #分割耗时需要算进去
training_time, testing_time

test_label_mask_cpu = test_label_mask.cpu().numpy()[:,0].astype('bool')
test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64')
predict = torch.argmax(output, 1).cpu().numpy()
np.unique(predict), np.unique(test_samples_gt_cpu)

classification = classification_report(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu]+1, digits=4)
kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu]+1)
print(classification)
print("kappa", kappa)
