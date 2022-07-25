import torch

def compute_loss(network_output: torch.Tensor, train_samples_gt_onehot: torch.Tensor, train_label_mask: torch.Tensor):
    real_labels = train_samples_gt_onehot
    we = -torch.mul(real_labels,torch.log(network_output))
    we = torch.mul(we, train_label_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy

def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, zeros):
    with torch.no_grad():
        available_label_idx = (train_samples_gt!=0).float()        # 有效标签的坐标,用于排除背景
        available_label_count = available_label_idx.sum()          # 有效标签的个数
        correct_prediction = torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1), available_label_idx, zeros).sum()
        OA= correct_prediction.cpu() / available_label_count
        return OA
