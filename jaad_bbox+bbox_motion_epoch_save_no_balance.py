from utils.jaad_data import JAAD
from utils.jaad_preprocessing_bbox_bbox_motion import *

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.main_model_jaad_bbox_bbox_motion import Model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import argparse
import re
from datetime import datetime
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_pos_neg(name, tte_seq):
    gt = [a[0][0] for a in tte_seq['activities']]
    num_pos = sum(gt)
    num_neg = len(gt) - num_pos
    print(f"{name} 集样本数：Positive={num_pos}  Negative={num_neg}  Total={len(gt)}")


def train(model, train_loader, valid_loader, class_criterion, reg_criterion, optimizer, checkpoint_filepath, writer,
          args):
    best_valid_acc = 0.0
    improvement_ratio = 0.001
    best_valid_loss = np.inf
    num_steps_wo_improvement = 0
    save_times = 0
    epochs = args.epochs
    if args.learn:  # 调试模式： epoch = 5
        epochs = 5
    time_crop = args.time_crop
    for epoch in range(epochs):
        nb_batches_train = len(train_loader)
        train_acc = 0
        model.train()
        f_losses = 0.0
        cls_losses = 0.0
        reg_losses = 0.0

        print('Epoch: {} training...'.format(epoch + 1))
        for bbox, bbox_motion, label, vel, traj in train_loader:
            label = label.reshape(-1, 1).to(device).float()
            bbox = bbox.to(device)
            bbox_motion= bbox_motion.to(device)
            vel = vel.to(device)
            end_point = traj.to(device)[:, -1, :4]

            if np.random.randint(10) >= 5 and time_crop:
                crop_size = np.random.randint(args.sta_f, args.end_f)
                bbox = bbox[:, -crop_size:, :]
                vel = vel[:, -crop_size:, :]

            pred, point, s_cls, s_reg = model(bbox,bbox_motion, vel)

            cls_loss = class_criterion(pred, label)
            reg_loss = reg_criterion(point, end_point)
            f_loss = cls_loss / (s_cls * s_cls) + reg_loss / (s_reg * s_reg) + torch.log(s_cls) + torch.log(s_reg)

            model.zero_grad()  #
            f_loss.backward()

            f_losses += f_loss.item()
            cls_losses += cls_loss.item()
            reg_losses += reg_loss.item()

            optimizer.step()  #

            train_acc += binary_acc(label, torch.round(pred))

        writer.add_scalar('training full_loss',
                          f_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training cls_loss',
                          cls_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training reg_loss',
                          reg_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training Acc',
                          train_acc / nb_batches_train,
                          epoch + 1)

        print(
            f"Epoch {epoch + 1}: | Train_Loss {f_losses / nb_batches_train} | Train Cls_loss {cls_losses / nb_batches_train} | Train Reg_loss {reg_losses / nb_batches_train} | Train_Acc {train_acc / nb_batches_train} ")

        valid_f_loss, valid_cls_loss, valid_reg_loss, val_acc = evaluate(model, valid_loader, class_criterion,
                                                                         reg_criterion)

        writer.add_scalar('validation full_loss',
                          valid_f_loss,
                          epoch + 1)
        writer.add_scalar('validation cls_loss',
                          valid_cls_loss,
                          epoch + 1)
        writer.add_scalar('validation reg_loss',
                          valid_reg_loss,
                          epoch + 1)
        writer.add_scalar('validation Acc',
                          val_acc,
                          epoch + 1)

        if best_valid_loss > valid_cls_loss:
            best_valid_loss = valid_cls_loss
            num_steps_wo_improvement = 0
            save_times += 1
            print(str(save_times) + ' time(s) File saved.\n')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Accuracy': train_acc / nb_batches_train,
                'LOSS': f_losses / nb_batches_train,
            }, checkpoint_filepath)
            print('Update improvement.\n')
        else:
            num_steps_wo_improvement += 1
            print(str(num_steps_wo_improvement) + '/300 times Not update.\n')

        if num_steps_wo_improvement == 300:
            print("Early stopping on epoch:{}".format(str(epoch + 1)))
            break
    print('save file times: ' + str(save_times) + '.\n')


def train_with_epochly_excel(model,
                             train_loader,
                             valid_loader,
                             test_loader,
                             class_criterion,
                             reg_criterion,
                             optimizer,
                             checkpoint_filepath,
                             writer,
                             args):
    import os, torch, pandas as pd
    from datetime import datetime
    import glob

    # — A) 路径准备 —
    dataset_name = os.path.basename(os.path.normpath(args.set_path))
    subset = args.bh
    prefix = f"{dataset_name}_{subset}"  # e.g. "JAAD_all"
    ckpt_dir = os.path.dirname(checkpoint_filepath)

    # 1) latest 文件：首次不存在就用 prefix_model_latest.pt，否则 prefix_model_latest_01.pt,02...
    latest_glob = glob.glob(os.path.join(ckpt_dir, f"{prefix}_model_latest*.pt"))
    ver_latest = len(latest_glob)
    if ver_latest == 0:
        latest_name = f"{prefix}_model_latest.pt"
    else:
        latest_name = f"{prefix}_model_latest_{ver_latest:02d}.pt"
    latest_path = os.path.join(ckpt_dir, latest_name)

    # 2) best 文件：同理
    best_glob = glob.glob(os.path.join(ckpt_dir, f"{prefix}_model_best*.pt"))
    ver_best = len(best_glob)
    if ver_best == 0:
        best_name = f"{prefix}_model_best.pt"
    else:
        best_name = f"{prefix}_model_best_{ver_best:02d}.pt"
    best_path = os.path.join(ckpt_dir, best_name)

    print(f"[Init] latest→{latest_name}, best→{best_name}")

    # ckpt_dir    = os.path.dirname(checkpoint_filepath)
    # latest_path = os.path.join(ckpt_dir, "model_latest.pt")
    # best_path   = os.path.join(ckpt_dir, "model_best.pt")

    # # 最新模型总是覆盖
    # latest_name = f"{prefix}_model_latest.pt"
    # latest_path = os.path.join(ckpt_dir, latest_name)

    # Excel 记录
    excel_dir  = os.path.join(ckpt_dir, 'each_epoch_test')
    os.makedirs(excel_dir, exist_ok=True)
    excel_path = os.path.join(excel_dir, 'epoch_test_metrics.xlsx')
    if os.path.exists(excel_path):
        os.remove(excel_path)

    # Early stopping / 最佳更新追踪
    best_test_auc = float('-inf')
    no_improve    = 0
    max_no_improve= 300
    epochs = 5 if args.learn else args.epochs

    for epoch in range(1, epochs+1):
        # — B) 训练一个 epoch —
        model.train()
        sum_full, sum_cls, sum_reg, sum_acc = 0., 0., 0., 0.
        for bbox, bbox_motion, label, vel, traj in train_loader:
            label = label.reshape(-1,1).to(device).float()
            bbox  = bbox.to(device)
            bbox_motion = bbox_motion.to(device)
            vel   = vel.to(device)
            end_pt= traj.to(device)[:, -1, :4]

            preds, pts, s_cls, s_reg = model(bbox, bbox_motion, vel)
            cls_loss  = class_criterion(preds, label)
            reg_loss  = reg_criterion(pts, end_pt)
            full_loss = cls_loss/(s_cls**2) + reg_loss/(s_reg**2) + torch.log(s_cls) + torch.log(s_reg)

            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()

            sum_full += full_loss.item()
            sum_cls  += cls_loss.item()
            sum_reg  += reg_loss.item()
            sum_acc  += binary_acc(label, torch.round(preds))

        nb_train = len(train_loader)
        writer.add_scalar('train/full_loss', sum_full/nb_train, epoch)
        writer.add_scalar('train/cls_loss',  sum_cls/nb_train,  epoch)
        writer.add_scalar('train/reg_loss',  sum_reg/nb_train,  epoch)
        writer.add_scalar('train/accuracy',  sum_acc/nb_train,  epoch)

        # — C) 验证 & TensorBoard —
        val_full, val_cls, val_reg, val_acc, val_f1, val_prec, val_rec, val_auc = evaluate(model, valid_loader, class_criterion, reg_criterion)
        writer.add_scalar('valid/full_loss', val_full,  epoch)
        writer.add_scalar('valid/cls_loss',  val_cls,   epoch)
        writer.add_scalar('valid/reg_loss',  val_reg,   epoch)
        writer.add_scalar('valid/accuracy',  val_acc,   epoch)
        writer.add_scalar('valid/f1',        val_f1,    epoch)
        writer.add_scalar('valid/precision', val_prec,  epoch)
        writer.add_scalar('valid/recall',    val_rec,   epoch)
        writer.add_scalar('valid/auc',       val_auc,   epoch)

        # — D) 测试集评估 & TensorBoard —
        tst_full, tst_cls, tst_reg, tst_acc, tst_f1, tst_prec, tst_rec, tst_auc = evaluate(model, test_loader, class_criterion, reg_criterion)
        writer.add_scalar('test/full_loss',  tst_full,  epoch)
        writer.add_scalar('test/cls_loss',   tst_cls,   epoch)
        writer.add_scalar('test/reg_loss',   tst_reg,   epoch)
        writer.add_scalar('test/accuracy',   tst_acc,   epoch)
        writer.add_scalar('test/f1',         tst_f1,    epoch)
        writer.add_scalar('test/precision',  tst_prec,  epoch)
        writer.add_scalar('test/recall',     tst_rec,   epoch)
        writer.add_scalar('test/auc',        tst_auc,   epoch)

        # — E) 保存最新模型（始终覆盖） —
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_path)

        # — F) 如果 test_auc 更优，则更新最佳模型 —
        if tst_auc > best_test_auc:
            best_test_auc = tst_auc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_path)
            print(f"[Info] New best model @ epoch {epoch}, test_auc={tst_auc:.4f}")
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                print(f"[Info] Early stopping at epoch {epoch}")
                break

        # — G) 写入 Excel —
        row = {
            'epoch':      epoch,
            'test_loss':  tst_full,
            'test_acc':   tst_acc,
            'test_f1':    tst_f1,
            'test_prec':  tst_prec,
            'test_rec':   tst_rec,
            'test_auc':   tst_auc,
            'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        df = pd.read_excel(excel_path) if os.path.exists(excel_path) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_excel(excel_path, index=False)

        # —— H) 控制台打印 ——
        print(
            f"Epoch {epoch:3d}: "
            f"TrainLoss={sum_full/nb_train:.4f} TrainAcc={sum_acc/nb_train:.4f} | "
            f"ValLoss={val_full:.4f} ValAcc={val_acc:.4f} ValF1={val_f1:.4f} "
            f"ValPrec={val_prec:.4f} ValRec={val_rec:.4f} ValAUC={val_auc:.4f} | "
            f"TestLoss={tst_full:.4f} TestAcc={tst_acc:.4f} TestF1={tst_f1:.4f} "
            f"TestPrec={tst_prec:.4f} TestRec={tst_rec:.4f} TestAUC={tst_auc:.4f}"
        )


    print(f"\n✔ 训练完毕 → 最新模型：{latest_path} ，最佳模型：{best_path}")




from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def evaluate(model, loader, cls_crit, reg_crit):
    model.eval()
    full_loss = cls_loss = reg_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for bbox, bbox_motion, label, vel, traj in loader:
            label = label.reshape(-1,1).to(device).float()
            bbox  = bbox.to(device)
            bbox_motion = bbox_motion.to(device)
            vel   = vel.to(device)
            end_pt= traj.to(device)[:, -1, :4]

            pred, pts, s_cls, s_reg = model(bbox, bbox_motion, vel)
            l_cls = cls_crit(pred, label)
            l_reg = reg_crit(pts, end_pt)
            l_full = l_cls/(s_cls**2) + l_reg/(s_reg**2) + torch.log(s_cls) + torch.log(s_reg)

            full_loss += l_full.item()
            cls_loss  += l_cls.item()
            reg_loss  += l_reg.item()

            y_true.extend(label.cpu().numpy().ravel().tolist())
            y_pred.extend(pred.cpu().numpy().ravel().tolist())

    # 平均
    n = len(loader)
    full_loss /= n
    cls_loss  /= n
    reg_loss  /= n

    # 其它指标
    y_bin = [1 if p>0.5 else 0 for p in y_pred]
    acc   = sum([yt==yb for yt,yb in zip(y_true, y_bin)]) / len(y_true)
    f1    = f1_score(y_true, y_bin)
    prec  = precision_score(y_true, y_bin)
    rec   = recall_score(y_true, y_bin)
    auc   = roc_auc_score(y_true, y_bin)

    # print(f"Eval → loss={full_loss:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
    print(
        f"Eval → full_loss={full_loss:.4f} cls_loss={cls_loss:.4f} reg_loss={reg_loss:.4f} | "
        f"acc={acc:.4f} f1={f1:.4f} precision={prec:.4f} recall={rec:.4f} auc={auc:.4f}"
    )

    return full_loss, cls_loss, reg_loss, acc, f1, prec, rec, auc

def main(args):
    checkpoint_dir = '/media/robert/4TB-SSD/checkpoints'

    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.test_only:
        assert args.checkpoint, "--checkpoint 必须指定一个 .pt 文件"

        # 构造 test_loader (与训练时相同的数据预处理流程)
        data_opts = {
            'fstride': 1,
            'sample_type': 'beh',
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'random',
            'seq_type': 'crossing',
            'min_track_size': 76,
            'kfold_params': {'num_folds': 5, 'fold': 1},

            # 'random_params': {'ratios': [0.7, 0.15, 0.15], 'val_data': True, 'regen_data': False}
        }

        imdb = JAAD(data_path=args.set_path)
        seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

        tte = [30, 60]
        tte_seq_test, traj_seq_test = tte_dataset(
            seq_test, tte, 0.8, args.times_num
        )

        # tte_seq_test, traj_seq_test = tte_dataset(
        #     seq_test, tte, 0.8, args.times_num
        # )

        # 准备 test tensors
        bbox_test = tte_seq_test['bbox']
        norm_bbox_test = normalize_bbox(bbox_test)

        bbox_motion_test = make_motion_features(norm_bbox_test)
        X_motion_test = torch.Tensor(bbox_motion_test)

        X_test = torch.Tensor(norm_bbox_test)
        actions = tte_seq_test['activities']
        Y_test = torch.Tensor(prepare_label(actions))
        traj_bbox = traj_seq_test['bbox']
        # X_test_dec = torch.Tensor(pad_sequence(normalize_traj(traj_bbox), args.times_num))

        # 用 tte[1] (比如 60) 来填充 decode 轨迹
        obs_len, tte_max = args.times_num, tte[1]
        X_test_dec = torch.Tensor(pad_sequence(normalize_traj(traj_bbox), tte_max))

        # obd = torch.Tensor(tte_seq_test['obd_speed'])
        # gps = torch.Tensor(tte_seq_test['gps_speed'])
        # vel_test = torch.cat([obd, gps], dim=-1)
        vel_test = torch.Tensor(tte_seq_test['vehicle_act'])

        testset = TensorDataset(X_test, X_motion_test, Y_test, vel_test, X_test_dec)
        test_loader = DataLoader(testset, batch_size=1)

        # 加载模型
        model = Model(args).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        # 评估
        preds, labels = test(model, test_loader)
        probs = preds.detach().cpu().numpy()
        labs = labels.detach().cpu().numpy().astype(int)
        pred_bin = (probs > 0.5).astype(int)

        acc = accuracy_score(labs, pred_bin)
        f1 = f1_score(labs, pred_bin)
        pre = precision_score(labs, pred_bin)
        rec = recall_score(labs, pred_bin)
        auc = roc_auc_score(labs, pred_bin)
        tn, fp, fn, tp = confusion_matrix(labs, pred_bin).ravel()

        ######################################保持pt文件 用于复现模型

        # 只取路径的最后一级作为模型名
        model_folder_name = os.path.basename(os.path.normpath(args.set_path))
        # checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)  # 确保目录存在
        #
        # # 扫一遍 checkpoints/ 里所有以 "model_folder_name_" 开头、".pt" 结尾的
        # pattern = re.compile(rf'^{re.escape(model_folder_name)}_(\d+)\.pt$')
        # versions = []
        # for fn in os.listdir(checkpoint_dir):
        #     m = pattern.match(fn)
        #     if m:
        #         versions.append(int(m.group(1)))
        #
        # # 下一个版本号
        # next_ver = max(versions) + 1 if versions else 0
        # # 格式化成两位，比如 00, 01, 02…
        # ver_str = f"{next_ver:02d}"
        #
        # checkpoint_filename = f"{model_folder_name}_{ver_str}.pt"
        # checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)
        # writer = SummaryWriter('logs/{}'.format(model_folder_name))  # 生成tensorboard的路径

        ##############################################

        print("==== Test Only Mode ====")
        print("Accuracy :", accuracy_score(labs, pred_bin))
        print("F1       :", f1_score(labs, pred_bin))
        print("Precision:", precision_score(labs, pred_bin))
        print("Recall   :", recall_score(labs, pred_bin))
        print("AUC      :", roc_auc_score(labs, pred_bin))
        print("CM       :\n", confusion_matrix(labs, pred_bin))

        #######################################用于记录excel文件

        # Log to Excel
        excel_path = os.path.join(checkpoint_dir, 'experiments_location.xlsx')

        dataset_name = os.path.basename(os.path.normpath(args.set_path))
        subset = args.bh
        prefix = f"{dataset_name}_{subset}"  # e.g. "JAAD_beh" 或者 "JAAD_all"

        record = {
            'checkpoint': checkpoint_dir,
            'location':   f"{prefix}.pt",
            'mode': 'train_val_test',
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'times_num': args.times_num,
            'accuracy': acc,
            'f1': f1,
            'precision': pre,
            'recall': rec,
            'auc': auc,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'drop_out': args.time_transformer_dropout,
            'num_layers': args.num_layers,
            'num_bnks': args.num_bnks,
            'bnks_layers': args.bnks_layers,
            'weight_decay': args.weight_decay,
        }

        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            new_row = pd.DataFrame([record])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame([record])
        df.to_excel(excel_path, index=False)

        # if os.path.exists(excel_path):
        #     df = pd.read_excel(excel_path)
        #     df = df.append(record, ignore_index=True)
        # else:
        #     df = pd.DataFrame([record])
        # df.to_excel(excel_path, index=False)

        print(f"Experiment logged to {excel_path}")

        ########################################################3

        return


    if not args.learn:
        seed_all(args.seed)
        data_opts = {'fstride': 1,
                    'sample_type': args.bh,  # 'beh'
                    'subset': 'default',
                    'height_rng': [0, float('inf')],
                    'squarify_ratio': 0,
                    'data_split_type': 'default',  # kfold, random, default
                    'seq_type': 'crossing',
                    'min_track_size': 76,
                    'random_params': {'ratios': None,
                                        'val_data': True,
                                        'regen_data': False},
                    'kfold_params': {'num_folds': 5, 'fold': 1},
        }
        tte = [30, 60]
        imdb = JAAD(data_path=args.set_path)
        # seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        # balanced_seq_train = balance_dataset(seq_train)
        # tte_seq_train, traj_seq_train = tte_dataset(balanced_seq_train, tte, 0.8, args.times_num)

        seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)  # 生成训练集
        if args.balance:
            balanced_seq_train = balance_dataset(seq_train)
        else:
            balanced_seq_train = seq_train
        # balanced_seq_train = balance_dataset(seq_train)  # 平衡数据集
        tte_seq_train, traj_seq_train = tte_dataset(balanced_seq_train, tte, 0.8,args.times_num)  # 生成训练集的tte和轨迹
        print_pos_neg("Train", tte_seq_train)
        # print("Post-overlap train sample count (images):", len(tte_seq_train['image']))
        # 记录 train labels
        train_labels = [a[0][0] for a in tte_seq_train['activities']]



        # seq_valid = imdb.generate_data_trajectory_sequence('val', **data_opts)
        # balanced_seq_valid = balance_dataset(seq_valid)
        # tte_seq_valid, traj_seq_valid = tte_dataset(balanced_seq_valid, tte, 0.8, args.times_num)

        seq_valid = imdb.generate_data_trajectory_sequence('val', **data_opts)
        if args.balance:
            balanced_seq_valid = balance_dataset(seq_valid)
        else:
            balanced_seq_valid = seq_valid
        # balanced_seq_valid = balance_dataset(seq_valid)
        tte_seq_valid, traj_seq_valid = tte_dataset(balanced_seq_valid, tte, 0.8, args.times_num)
        print_pos_neg("Valid", tte_seq_valid)
        # print("Post-overlap valid sample count (images):", len(tte_seq_valid['image']))
        # 记录 valid labels
        valid_labels = [a[0][0] for a in tte_seq_valid['activities']]

        seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        tte_seq_test, traj_seq_test = tte_dataset(seq_test, tte, 0.8, args.times_num)
        print_pos_neg("Test", tte_seq_test)
        # print("Post-overlap test sample count (images):", len(tte_seq_test['image']))
        # 记录 test labels
        test_labels = [a[0][0] for a in tte_seq_test['activities']]


        # seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        # tte_seq_test, traj_seq_test = tte_dataset(seq_test, tte, 0.8, args.times_num)

        # —— 新增：合并三部分，输出总体正负样本数 ——
        all_labels = train_labels + valid_labels + test_labels
        total_pos = sum(all_labels)
        total_neg = len(all_labels) - total_pos
        print(f"Overall 正样本: {total_pos}，负样本: {total_neg}，总样本: {len(all_labels)}")

        bbox_train = tte_seq_train['bbox']
        bbox_valid = tte_seq_valid['bbox']
        bbox_test = tte_seq_test['bbox']

        bbox_dec_train = traj_seq_train['bbox']
        bbox_dec_valid = traj_seq_valid['bbox']
        bbox_dec_test  = traj_seq_test['bbox']

        vel_train = tte_seq_train['vehicle_act']
        vel_valid = tte_seq_valid['vehicle_act']
        vel_test = tte_seq_test['vehicle_act']

        action_train = tte_seq_train['activities']
        action_valid = tte_seq_valid['activities']
        action_test = tte_seq_test['activities']

        normalized_bbox_train = normalize_bbox(bbox_train)
        normalized_bbox_valid = normalize_bbox(bbox_valid)
        normalized_bbox_test = normalize_bbox(bbox_test)

        # 2. 运动差分，直接接受 list[np.ndarray]
        bbox_motion_train = make_motion_features(normalized_bbox_train)  # List[np.ndarray], 每个 (T_i,4)
        bbox_motion_valid = make_motion_features(normalized_bbox_valid)
        bbox_motion_test = make_motion_features(normalized_bbox_test)

        # normalized_bbox_train = torch.tensor(normalized_bbox_train, dtype=torch.float32).to(device)
        # normalized_bbox_valid = torch.tensor(normalized_bbox_valid, dtype=torch.float32).to(device)
        # normalized_bbox_test = torch.tensor(normalized_bbox_test, dtype=torch.float32).to(device)

        # bbox_motion_train = make_motion_features(normalized_bbox_train)
        # bbox_motion_valid = make_motion_features(normalized_bbox_valid)
        # bbox_motion_test = make_motion_features(normalized_bbox_test)

        # 扩展成 8 维
        # normalized_bbox_train = add_interp_features(normalized_bbox_train)
        # normalized_bbox_valid = add_interp_features(normalized_bbox_valid)
        # normalized_bbox_test = add_interp_features(normalized_bbox_test)

        # normalized_bbox_train = add_delta_features(normalized_bbox_train)
        # normalized_bbox_valid = add_delta_features(normalized_bbox_valid)
        # normalized_bbox_test = add_delta_features(normalized_bbox_test)

        normalized_bbox_dec_train = normalize_traj(bbox_dec_train)
        normalized_bbox_dec_valid = normalize_traj(bbox_dec_valid)
        normalized_bbox_dec_test  = normalize_traj(bbox_dec_test)

        ##################################### new
        # normalized_bbox_dec_train = add_delta_features(normalized_bbox_dec_train)
        # normalized_bbox_dec_valid = add_delta_features(normalized_bbox_dec_valid)
        # normalized_bbox_dec_test = add_delta_features(normalized_bbox_dec_test)

        label_action_train = prepare_label(action_train)
        label_action_valid = prepare_label(action_valid)
        label_action_test = prepare_label(action_test)

        X_train, X_valid = torch.Tensor(normalized_bbox_train), torch.Tensor(normalized_bbox_valid)
        Y_train, Y_valid = torch.Tensor(label_action_train), torch.Tensor(label_action_valid)
        X_test = torch.Tensor(normalized_bbox_test)
        Y_test = torch.Tensor(label_action_test)

        X_motion_train = torch.Tensor(bbox_motion_train)
        X_motion_valid = torch.Tensor(bbox_motion_valid)
        X_motion_test = torch.Tensor(bbox_motion_test)


        X_train_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_train, 60))
        X_valid_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_valid, 60))
        X_test_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_test, 60))

        vel_train = torch.Tensor(vel_train)
        vel_valid = torch.Tensor(vel_valid)
        vel_test = torch.Tensor(vel_test)

        trainset = TensorDataset(X_train,X_motion_train, Y_train, vel_train, X_train_dec)
        validset = TensorDataset(X_valid,X_motion_valid, Y_valid, vel_valid, X_valid_dec)
        testset = TensorDataset(X_test,X_motion_test, Y_test, vel_test, X_test_dec)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=1)
    else: # 生成随机数据
        train_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)),
                         torch.randn(size=(args.batch_size, 1)),
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
        valid_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)),
                         torch.randn(size=(args.batch_size, 1)),
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
        test_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)),
                        torch.randn(size=(args.batch_size, 1)),
                        torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),
                        torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
    print('Start Training Loop... \n')

    model = Model(args)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    cls_criterion = nn.BCELoss()
    reg_criterion = nn.MSELoss()

    ######################################保持pt文件 用于复现模型

    # 只取路径的最后一级作为模型名
    model_folder_name = os.path.basename(os.path.normpath(args.set_path))
    # checkpoint_dir = 'checkpoints'
    # os.makedirs(checkpoint_dir, exist_ok=True)  # 确保目录存在

    # 扫一遍 checkpoints/ 里所有以 "model_folder_name_" 开头、".pt" 结尾的
    pattern = re.compile(rf'^{re.escape(model_folder_name)}_(\d+)\.pt$')
    versions = []
    for fn in os.listdir(checkpoint_dir):
        m = pattern.match(fn)
        if m:
            versions.append(int(m.group(1)))

    # 下一个版本号
    next_ver = max(versions) + 1 if versions else 0
    # 格式化成两位，比如 00, 01, 02…
    ver_str = f"{next_ver:02d}"

    checkpoint_filename = f"{model_folder_name}_{ver_str}.pt"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)
    writer = SummaryWriter('logs/{}'.format(model_folder_name))  # 生成tensorboard的路径

    ##############################################

    # model_folder_name = args.set_path + '_' + args.bh
    # os.makedirs('checkpoints', exist_ok=True)
    # checkpoint_filepath = os.path.join('checkpoints', model_folder_name + '.pt')
    # # checkpoint_filepath = 'checkpoints/{}.pt'.format(model_folder_name)
    # writer = SummaryWriter('logs/{}'.format(model_folder_name))

    # train(model, train_loader, valid_loader, cls_criterion, reg_criterion, optimizer, checkpoint_filepath, writer, args=args)
    train_with_epochly_excel(model, train_loader, valid_loader, test_loader, cls_criterion, reg_criterion, optimizer, checkpoint_filepath, writer, args=args)



    #Test
    model = Model(args)
    model.to(device)

    # checkpoint = torch.load(checkpoint_filepath)
    ckpt_dir = os.path.dirname(checkpoint_filepath)
    best_path = os.path.join(ckpt_dir, "model_best.pt")
    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    preds, labels = test(model, test_loader)
    pred_cpu = torch.Tensor.cpu(preds)
    label_cpu = torch.Tensor.cpu(labels)

    acc = accuracy_score(label_cpu, np.round(pred_cpu))
    f1 = f1_score(label_cpu, np.round(pred_cpu))
    pre_s = precision_score(label_cpu, np.round(pred_cpu))
    recall_s = recall_score(label_cpu, np.round(pred_cpu))
    auc = roc_auc_score(label_cpu, np.round(pred_cpu))
    matrix = confusion_matrix(label_cpu, np.round(pred_cpu))
    tn, fp, fn, tp = confusion_matrix(label_cpu, np.round(pred_cpu)).ravel()

    print(f'Acc: {acc}\n f1: {f1}\n precision_score: {pre_s}\n recall_score: {recall_s}\n roc_auc_score: {auc}\n confusion_matrix: {matrix}')

    #################################### 自动保持excel文件
    # log to Excel
    excel_path = os.path.join(checkpoint_dir, 'experiments_location.xlsx')

    record = {
        'checkpoint': checkpoint_filename + 'BBox_motion',
        'location': model_folder_name + '_' + args.bh +  '.pt',
        'mode': 'train_val_test',
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'times_num': args.times_num,
        'accuracy': acc,
        'f1': f1,
        'precision': pre_s,
        'recall': recall_s,
        'auc': auc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'drop_out': args.time_transformer_dropout,
        'num_layers': args.num_layers,
        'num_bnks': args.num_bnks,
        'bnks_layers': args.bnks_layers,
        'weight_decay': args.weight_decay,
    }

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        new_row = pd.DataFrame([record])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_excel(excel_path, index=False)

    print(f"Experiment logged to {excel_path}")
    ####################################

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('Pedestrain Crossing Intention Prediction.')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
    parser.add_argument('--set_path', type=str, default='/home/robert/桌面/Pedestrian_Crossing_Intention_Prediction-main/JAAD')
    parser.add_argument('--bh', type=str, default='all', help='all or beh, in JAAD dataset.')
    parser.add_argument('--balance', type=bool, default=True, help='balance or not for test dataset.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=256, help='the dimension after embedding.')
    parser.add_argument('--dff', type=int, default=512, help='the number of the units.')
    parser.add_argument('--num_heads', type=int, default=8, help='number of the heads of the multi-head model.')
    parser.add_argument('--bbox_input', type=int, default=4, help='dimension of bbox.')
    parser.add_argument('--vel_input', type=int, default=1, help='dimension of velocity.')
    parser.add_argument('--time_crop', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch.')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train.')
    parser.add_argument('--num_layers', type=int, default=4, help='the number of layers.')
    parser.add_argument('--times_num', type=int, default=32, help='')
    parser.add_argument('--num_bnks', type=int, default=9, help='')
    parser.add_argument('--bnks_layers', type=int, default=9, help='')
    parser.add_argument('--sta_f', type=int, default=8)
    parser.add_argument('--end_f', type=int, default=12)
    parser.add_argument('--learn', action='store_true', help='If set, generate random data instead of real dataset')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay (L2 regularization) factor.')
    parser.add_argument('--time_transformer_num_heads', type=int, default=3, help='Number of heads for the TimeTransformer module.')
    parser.add_argument('--time_transformer_dropout', type=float, default=0.4, help='Dropout rate for the TimeTransformer module.')
    parser.add_argument('--test_only', action='store_true', help='只做测试，加载 --checkpoint 并 eval')
    parser.add_argument('--checkpoint', type=str, default='/media/robert/4TB-SSD/checkpoints/JAAD_bh_model_latest.pt', help='测试模式下指定 .pt 文件路径')
    parser.add_argument('--no-balance', dest='balance', action='store_false', default=True, help='禁用正负样本平衡，直接使用原始分布的训练/验证样本' )


    args = parser.parse_args()
    main(args)