import torch
import os
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed):
    """
    设置随机种子，确保可复现性
    """
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_acc(label, pred):
    """
    计算二分类准确率:
      - label: [batch_size, 1]
      - pred:  [batch_size, 1]
    """
    label_tag = torch.round(label)
    correct = (label_tag == pred).sum().float()
    return correct / pred.shape[0]


def normalize_bbox(dataset, width=1920, height=1080):
    """
    对每条轨迹的 [x1,y1,x2,y2] 做归一化
    输入:
      - dataset: list of sequences, 每个 sequence 是 list 或 np.array, shape [T_i, 4]
    输出:
      - normalized_set: list of np.array, 归一化后相同 shape
    """
    normalized_set = []
    for seq in dataset:
        if not seq:
            continue
        norm_seq = []
        # seq 可能是 list，也可能是 np.ndarray
        for bbox in seq:
            # bbox 必须可索引为 [x1,y1,x2,y2]
            x1, y1, x2, y2 = bbox
            norm_seq.append([
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height
            ])
        normalized_set.append(np.array(norm_seq, dtype=float))
    return normalized_set


def prepare_label(dataset):
    """
    将 activities 转为 0/1 标签
    输入:
      - dataset: list of activities, 每个元素为 [[label], ...]
    输出:
      - labels: np.array of shape [N], dtype int64
    """
    labels = np.zeros(len(dataset), dtype='int64')
    for i, act in enumerate(dataset):
        if act:
            labels[i] = act[0][0]
    return labels


def train(model, train_loader, valid_loader, class_criterion, reg_criterion, optimizer, checkpoint_filepath, writer, args):
    """
    训练循环，仅使用 bbox 输入进行二分类
    - train_loader 返回 (bbox, lengths, labels)
    - criterion: BCE Loss
    """
    best_valid_loss = float('inf')
    epochs = args.epochs if not args.learn else 5

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        nb_batches = len(train_loader)

        print(f'Epoch {epoch}/{epochs} - Training...')
        for bbox, lengths, labels in train_loader:
            # 数据移动到 GPU
            bbox = bbox.to(device)
            lengths = lengths.to(device)
            labels = labels.view(-1, 1).to(device).float()

            # 前向 + 损失 + 反向
            preds = model(bbox, lengths).view(-1, 1).view(-1, 1)
            loss = class_criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积
            total_loss += loss.item()
            total_acc  += binary_acc(labels, torch.round(preds)).item()

        avg_train_loss = total_loss / nb_batches
        avg_train_acc  = total_acc  / nb_batches
        writer.add_scalar('training cls_loss', avg_train_loss, epoch)
        writer.add_scalar('training Acc', avg_train_acc, epoch)
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")

        # 验证
        valid_loss, valid_acc = evaluate(model, valid_loader, class_criterion, reg_criterion)
        writer.add_scalar('validation cls_loss', valid_loss, epoch)
        writer.add_scalar('validation Acc', valid_acc, epoch)

        # 保存最优模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': best_valid_loss
            }, checkpoint_filepath)
            print('Model improved. Checkpoint saved.')
        else:
            print('No improvement.')

    print('Training completed.')


def evaluate(model, valid_loader, class_criterion, reg_criterion):
    """
    验证函数，仅计算分类损失与准确率
    - valid_loader 返回 (bbox, lengths, labels)
    """
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    nb_batches = len(valid_loader)

    with torch.no_grad():
        print('Evaluating...')
        for bbox, lengths, labels in valid_loader:
            bbox   = bbox.to(device)
            lengths= lengths.to(device)
            labels = labels.view(-1,1).to(device).float()

            preds = model(bbox, lengths)
            # 只计算分类损失，reg_criterion 未使用
            loss  = class_criterion(preds, labels)

            total_loss += loss.item()
            total_acc  += binary_acc(labels, torch.round(preds)).item()

    avg_loss = total_loss / nb_batches
    avg_acc  = total_acc  / nb_batches
    print(f"Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.4f}")
    return avg_loss, avg_acc


def test(model, test_loader):
    """
    测试函数，返回 preds 和 labels
    - test_loader 返回 (bbox, lengths, labels)
    """
    model.eval()
    preds_list  = []
    labels_list = []

    with torch.no_grad():
        print('Testing...')
        for bbox, lengths, labels in test_loader:
            bbox    = bbox.to(device)
            lengths = lengths.to(device)
            labels  = labels.view(-1,1).to(device).float()

            preds = model(bbox, lengths)
            preds_list.append(preds.cpu())
            labels_list.append(labels.cpu())

    preds  = torch.cat(preds_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return preds, labels
