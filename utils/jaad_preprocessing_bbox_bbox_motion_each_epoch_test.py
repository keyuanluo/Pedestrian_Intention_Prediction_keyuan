import torch
import os
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_acc(label, pred):
    label_tag = torch.round(label)
    correct_results_sum = (label_tag == pred).sum().float()
    acc = correct_results_sum / pred.shape[0]
    return acc


def end_point_loss(reg_criterion, pred, end_point):
    for i in range(4):
        if i == 0 or i == 2:
            pred[:, i] = pred[:, i] * 1920
            end_point[:, i] = end_point[:, i] * 1920
        else:
            pred[:, i] = pred[:, i] * 1080
            end_point[:, i] = end_point[:, i] * 1080
    return reg_criterion(pred, end_point)



import os
import pandas as pd
from datetime import datetime

def train(model,
          train_loader,
          valid_loader,
          test_loader,
          class_criterion,
          reg_criterion,
          optimizer,
          checkpoint_filepath,
          writer,
          args):
    # 1) 准备输出目录和记录容器
    excel_dir = '/media/robert/4TB-SSD/checkpoints/each_epoch_test'
    os.makedirs(excel_dir, exist_ok=True)
    excel_path = os.path.join(excel_dir, 'test_metrics_per_epoch_01.xlsx')
    records = []

    best_valid_loss = float('inf')
    no_improve = 0
    max_no_improve = 300
    epochs = 5 if args.learn else args.epochs
    time_crop = args.time_crop

    for epoch in range(1, epochs + 1):
        # —— A) 训练一个 epoch （省略） —— #
        model.train()
        # for bbox, bbox_motion, label, vel, traj in train_loader:
        #     # ... 完整的训练步骤 ...

        for bbox, bbox_motion, label, vel, traj in train_loader:
            label = label.view(-1, 1).to(device).float()
            bbox = bbox.to(device)
            bbox_motion = bbox_motion.to(device)
            vel = vel.to(device)
            end_point = traj.to(device)[:, -1, :4]

            # 可选的时间裁剪
            if np.random.randint(10) >= 5 and time_crop:
                crop_size = np.random.randint(args.sta_f, args.end_f)
                bbox = bbox[:, -crop_size:, :]
                vel = vel[:, -crop_size:, :]

            # 前向
            preds, points, s_cls, s_reg = model(bbox, bbox_motion, vel)  # 如果 forward 接收 decode-tra j 作为第四个参数
            # 损失
            cls_loss = class_criterion(preds, label)
            reg_loss = reg_criterion(points, end_point)
            full_loss = cls_loss / (s_cls ** 2) + reg_loss / (s_reg ** 2) + torch.log(s_cls) + torch.log(s_reg)

            # 反向更新
            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()


        # —— B) 验证集评估，决定 checkpoint —— #
        metrics_val = evaluate(
            model, valid_loader, class_criterion, reg_criterion, split_name='Valid'
        )
        if metrics_val['cls_loss'] < best_valid_loss:
            best_valid_loss = metrics_val['cls_loss']
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_filepath)
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                print("Early stopping")
                break

        # —— C) 测试集评估 —— #
        metrics_test = evaluate(
            model, test_loader, class_criterion, reg_criterion, split_name='Test'
        )

        # —— D) 把 test 的指标 append 到 records —— #
        records.append({
            'epoch':       epoch,
            'test_loss':   metrics_test['full_loss'],
            'test_acc':    metrics_test['accuracy'],
            'test_f1':     metrics_test['f1'],
            'test_prec':   metrics_test['precision'],
            'test_rec':    metrics_test['recall'],
            'test_auc':    metrics_test['auc'],
            'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })


        print(
            f"Epoch {epoch}: "
            f"loss={metrics_test['full_loss']:.4f}  "
            f"acc={metrics_test['accuracy']:.4f}  "
            f"f1={metrics_test['f1']:.4f}  "
            f"prec={metrics_test['precision']:.4f}  "
            f"rec={metrics_test['recall']:.4f}  "
            f"auc={metrics_test['auc']:.4f}"
        )

    # —— E) 全部 epoch 完毕后，一次性写入 Excel —— #
    df = pd.DataFrame(records)
    df.to_excel(excel_path, index=False)
    print(f"\n训练结束，测试集指标已保存到 {excel_path}")





def evaluate(model,
             data_loader,
             class_criterion,
             reg_criterion,
             split_name='Eval'):
    """
    在 data_loader（验证集或测试集）上跑一次，返回所有指标的字典，并打印：
      loss, accuracy, f1, precision, recall, auc
    split_name 用于打印时区分："Valid" 或 "Test"
    """
    model.eval()
    nb = len(data_loader)
    sum_full, sum_cls, sum_reg = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for bbox, bbox_motion, label, vel, traj in data_loader:
            label = label.reshape(-1,1).to(device).float()
            bbox = bbox.to(device)
            bbox_motion = bbox_motion.to(device)
            vel = vel.to(device)
            end_point = traj.to(device)[:, -1, :4]

            preds, points, s_cls, s_reg = model(bbox, bbox_motion, vel)

            cls_loss = class_criterion(preds, label).item()
            reg_loss = reg_criterion(points, end_point).item()
            full_loss = (
                cls_loss/(s_cls**2).item()
                + reg_loss/(s_reg**2).item()
                + torch.log(s_cls).item()
                + torch.log(s_reg).item()
            )

            sum_full += full_loss
            sum_cls  += cls_loss
            sum_reg  += reg_loss

            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

    # 拼接所有 batch
    preds_cat = torch.cat(all_preds).numpy().ravel()
    labs_cat  = torch.cat(all_labels).numpy().ravel().astype(int)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    metrics = {
        'full_loss': sum_full/nb,
        'cls_loss':  sum_cls/nb,
        'reg_loss':  sum_reg/nb,
        'accuracy':  accuracy_score(labs_cat, preds_cat>0.5),
        'f1':        f1_score(labs_cat, preds_cat>0.5),
        'precision': precision_score(labs_cat, preds_cat>0.5),
        'recall':    recall_score(labs_cat, preds_cat>0.5),
        'auc':       roc_auc_score(labs_cat, preds_cat),
    }

    # 把所有指标都打印出来
    print(
        f">>> {split_name}: "
        f"loss={metrics['full_loss']:.4f}  "
        f"acc={metrics['accuracy']:.4f}  "
        f"f1={metrics['f1']:.4f}  "
        f"prec={metrics['precision']:.4f}  "
        f"rec={metrics['recall']:.4f}  "
        f"auc={metrics['auc']:.4f}"
    )
    return metrics







def test(model, test_data):
    print('Tesing...')
    with torch.no_grad():
        model.eval()
        step = 0
        for bbox, bbox_motion, label, vel, traj in test_data:
            label = label.reshape(-1, 1).to(device).float()
            bbox = bbox.to(device)
            bbox_motion = bbox_motion.to(device)
            vel = vel.to(device)

            pred, _, _, _ = model(bbox, bbox_motion, vel)

            if step == 0:
                preds = pred
                labels = label
            else:
                preds = torch.cat((preds, pred), 0)
                labels = torch.cat((labels, label), 0)
            step += 1

    return preds, labels


def balance_dataset(dataset, flip=True):
    d = {'bbox': dataset['bbox'].copy(),
         'pid': dataset['pid'].copy(),
         'activities': dataset['activities'].copy(),
         'image': dataset['image'].copy(),
         'center': dataset['center'].copy(),
         'vehicle_act': dataset['vehicle_act'].copy(),
         'image_dimension': (1920, 1080)}
    gt_labels = [gt[0] for gt in d['activities']]
    num_pos_samples = np.count_nonzero(np.array(gt_labels))
    num_neg_samples = len(gt_labels) - num_pos_samples

    if num_neg_samples == num_pos_samples:
        print('Positive samples is equal to negative samples.')
    else:
        print('Unbalanced: \t Postive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
        if num_neg_samples > num_pos_samples:
            gt_augment = 1
        else:
            gt_augment = 0

        img_width = d['image_dimension'][0]
        num_samples = len(d['pid'])

        for i in range(num_samples):
            if d['activities'][i][0][0] == gt_augment:
                flipped = d['center'][i].copy()
                flipped = [[img_width - c[0], c[1]] for c in flipped]
                d['center'].append(flipped)

                flipped = d['bbox'][i].copy()
                flipped = [np.array([img_width - c[2], c[1], img_width - c[0], c[3]]) for c in flipped]
                d['bbox'].append(flipped)

                d['pid'].append(dataset['pid'][i].copy())

                d['activities'].append(d['activities'][i].copy())
                d['vehicle_act'].append(d['vehicle_act'][i].copy())

                flipped = d['image'][i].copy()
                flipped = [c.replace('.png', '_flip.png') for c in flipped]

                d['image'].append(flipped)

        gt_labels = [gt[0] for gt in d['activities']]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        if num_neg_samples > num_pos_samples:
            rm_index = np.where(np.array(gt_labels) == 0)[0]
        else:
            rm_index = np.where(np.array(gt_labels) == 1)[0]

        dif_samples = abs(num_neg_samples - num_pos_samples)

        np.random.seed(42)
        np.random.shuffle(rm_index)
        rm_index = rm_index[0:dif_samples]

        for k in d:
            seq_data_k = d[k]
            d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

        new_gt_labels = [gt[0] for gt in d['activities']]
        num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
        print('Balanced: Postive: %d \t Negative: %d \n' % (num_pos_samples, len(d['activities']) - num_pos_samples))
        print('Total Number of samples: %d\n' % (len(d['activities'])))

    return d


def tte_dataset(dataset, time_to_event, overlap, obs_length):
    d_obs = {'bbox': dataset['bbox'].copy(),
             'pid': dataset['pid'].copy(),
             'activities': dataset['activities'].copy(),
             'image': dataset['image'].copy(),
             'vehicle_act': dataset['vehicle_act'].copy(),
             'center': dataset['center'].copy()
             }

    d_tte = {'bbox': dataset['bbox'].copy(),
             'pid': dataset['pid'].copy(),
             'activities': dataset['activities'].copy(),
             'image': dataset['image'].copy(),
             'vehicle_act': dataset['vehicle_act'].copy(),
             'center': dataset['center'].copy()}

    if isinstance(time_to_event, int):
        for k in d_obs.keys():
            for i in range(len(d_obs[k])):
                d_obs[k][i] = d_obs[k][i][- obs_length - time_to_event: -time_to_event]
                d_tte[k][i] = d_tte[k][i][- time_to_event:]
        d_obs['tte'] = [[time_to_event]] * len(dataset['bbox'])
        d_tte['tte'] = [[time_to_event]] * len(dataset['bbox'])

    else:
        olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
        olap_res = 1 if olap_res < 1 else olap_res

        for k in d_obs.keys():
            seqs = []
            seqs_tte = []
            for seq in d_obs[k]:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                seqs.extend([seq[i:i + obs_length] for i in range(start_idx, end_idx, olap_res)])
                seqs_tte.extend([seq[i + obs_length:] for i in range(start_idx, end_idx, olap_res)])
                d_obs[k] = seqs
                d_tte[k] = seqs_tte
        tte_seq = []
        for seq in dataset['bbox']:
            start_idx = len(seq) - obs_length - time_to_event[1]
            end_idx = len(seq) - obs_length - time_to_event[0]
            tte_seq.extend([[len(seq) - (i + obs_length)] for i in range(start_idx, end_idx, olap_res)])
            d_obs['tte'] = tte_seq.copy()
            d_tte['tte'] = tte_seq.copy()

    remove_index = []
    try:
        time_to_event_0 = time_to_event[0]
    except:
        time_to_event_0 = time_to_event
    for seq_index, (seq_obs, seq_tte) in enumerate(zip(d_obs['bbox'], d_tte['bbox'])):
        if len(seq_obs) < 16 or len(seq_tte) < time_to_event_0:
            remove_index.append(seq_index)

    for k in d_obs.keys():
        for j in sorted(remove_index, reverse=True):
            del d_obs[k][j]
            del d_tte[k][j]

    return d_obs, d_tte


def normalize_bbox(dataset, width=1920, height=1080):
    normalized_set = []
    for sequence in dataset:
        if sequence == []:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0] / width
            np_bbox[2] = bbox[2] / width
            np_bbox[1] = bbox[1] / height
            np_bbox[3] = bbox[3] / height
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))

    return normalized_set


def normalize_traj(dataset, width=1920, height=1080):
    normalized_set = []
    for sequence in dataset:
        if sequence == []:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0]  # / width
            np_bbox[2] = bbox[2]  # / width
            np_bbox[1] = bbox[1]  # / height
            np_bbox[3] = bbox[3]  # / height
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))

    return normalized_set


def prepare_label(dataset):
    labels = np.zeros(len(dataset), dtype='int64')
    for step, action in enumerate(dataset):
        if action == []:
            continue
        labels[step] = action[0][0]

    return labels


def pad_sequence(inp_list, max_len):
    padded_sequence = []
    for source in inp_list:
        target = np.array([source[0]] * max_len)
        source = source
        target[-source.shape[0]:, :] = source

        padded_sequence.append(target)

    return padded_sequence




def make_motion_features(bbox_list):
    """
   bbox_list: List[np.ndarray] 或 List[list]，每个元素 shape=(T_i, 4)
   return:     List[np.ndarray]，每个元素 shape=(T_i, 4)
                每帧 Δ = bbox[t] - bbox[t-1] (t=0 时全 0)
   """

    out = []
    for seq in bbox_list:

        arr = np.asarray(seq, dtype=np.float32)  # (T,4)
        delta = np.zeros_like(arr)   # (T,4)
        if arr.shape[0] > 1:
            delta[1:] = arr[1:] - arr[:-1]
            out.append(delta)
    return out