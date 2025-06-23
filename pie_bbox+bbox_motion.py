from utils.pie_data_new import PIE
from utils.pie_preprocessing_bbox_bbox_motion import *

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.main_model_pie_bbox_bbox_motion import Model
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



def main(args):
    checkpoint_dir = '/media/robert/4TB-SSD/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.test_only:
        assert args.checkpoint, "--checkpoint 必须指定一个 .pt 文件"
        # 构造 test_loader (与训练时相同的数据预处理流程)
        data_opts = {
            'fstride': 1,
            'sample_type': 'all',
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'random',
            'seq_type': 'crossing',
            'min_track_size': 15,
            'kfold_params': {'num_folds': 1, 'fold': 1},
            'random_params': {'ratios': [0.7, 0.15, 0.15], 'val_data': True, 'regen_data': False},
            'tte': [30, 60],
            'batch_size': args.batch_size
        }
        imdb = PIE(data_path=args.set_path)
        seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        tte_seq_test, traj_seq_test = tte_dataset(
            seq_test, data_opts['tte'], 0.8, args.times_num
        )

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
        obs_len, tte_max = args.times_num, data_opts['tte'][1]
        X_test_dec = torch.Tensor(pad_sequence(normalize_traj(traj_bbox), tte_max))

        obd = torch.Tensor(tte_seq_test['obd_speed'])
        gps = torch.Tensor(tte_seq_test['gps_speed'])
        vel_test = torch.cat([obd, gps], dim=-1)

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
        # os.makedirs(checkpoint_dir, exist_ok=True)  # 确保目录存在
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

        record = {
            'checkpoint': os.path.basename(args.checkpoint),
            'location': model_folder_name + '.pt',
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
            'drop_out': args.time_transformer_dropout
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





    if not args.learn:  # 如果args.learn为False，则真实训练， 读取真实数据
        seed_all(args.seed)
        data_opts = {
            'fstride': 1,
            'sample_type': 'all',
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'default',  # kfold, random, default
            'seq_type': 'crossing',  # crossing , intention
            'min_track_size': 76,  # discard tracks that are shorter
            'kfold_params': {'num_folds': 5, 'fold': 1},
            'random_params': {'ratios': None,
                              'val_data': True,
                              'regen_data': True},
            'tte': [30, 60],
            'batch_size': 16
        }
        imdb = PIE(data_path=args.set_path)


        seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)  # 生成训练集
        if args.balance:
            balanced_seq_train = balance_dataset(seq_train)
        else: balanced_seq_train = seq_train
        # balanced_seq_train = balance_dataset(seq_train)  # 平衡数据集
        tte_seq_train, traj_seq_train = tte_dataset(balanced_seq_train, data_opts['tte'], 0.6, args.times_num)  # 生成训练集的tte和轨迹
        print_pos_neg("Train", tte_seq_train)
        # print("Post-overlap train sample count (images):", len(tte_seq_train['image']))
        # 记录 train labels
        train_labels = [a[0][0] for a in tte_seq_train['activities']]


        seq_valid = imdb.generate_data_trajectory_sequence('val', **data_opts)
        if args.balance:
            balanced_seq_valid = balance_dataset(seq_valid)
        else: balanced_seq_valid = seq_valid
        # balanced_seq_valid = balance_dataset(seq_valid)
        tte_seq_valid, traj_seq_valid = tte_dataset(balanced_seq_valid, data_opts['tte'], 0.6, args.times_num)
        print_pos_neg("Valid", tte_seq_valid)
        # print("Post-overlap valid sample count (images):", len(tte_seq_valid['image']))
        # 记录 valid labels
        valid_labels = [a[0][0] for a in tte_seq_valid['activities']]


        seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        tte_seq_test, traj_seq_test = tte_dataset(seq_test, data_opts['tte'], 0.6, args.times_num)
        print_pos_neg("Test", tte_seq_test)
        # print("Post-overlap test sample count (images):", len(tte_seq_test['image']))
        # 记录 test labels
        test_labels = [a[0][0] for a in tte_seq_test['activities']]

        # —— 新增：合并三部分，输出总体正负样本数 ——
        all_labels = train_labels + valid_labels + test_labels
        total_pos = sum(all_labels)
        total_neg = len(all_labels) - total_pos
        print(f"Overall 正样本: {total_pos}，负样本: {total_neg}，总样本: {len(all_labels)}")


        bbox_train = tte_seq_train['bbox']  # 训练集的bbox
        bbox_valid = tte_seq_valid['bbox']
        bbox_test = tte_seq_test['bbox']

        bbox_dec_train = traj_seq_train['bbox']  # 训练集的轨迹
        bbox_dec_valid = traj_seq_valid['bbox']
        bbox_dec_test = traj_seq_test['bbox']

        obd_train = tte_seq_train['obd_speed']  # 训练集的速度
        obd_valid = tte_seq_valid['obd_speed']
        obd_test = tte_seq_test['obd_speed']

        gps_train = tte_seq_train['gps_speed']  # 训练集的速度
        gps_valid = tte_seq_valid['gps_speed']
        gps_test = tte_seq_test['gps_speed']

        action_train = tte_seq_train['activities']  # 训练集的动作
        action_valid = tte_seq_valid['activities']
        action_test = tte_seq_test['activities']

        normalized_bbox_train = normalize_bbox(bbox_train)  # 归一化bbox
        normalized_bbox_valid = normalize_bbox(bbox_valid)
        normalized_bbox_test = normalize_bbox(bbox_test)

        ####新加的bbox_motion
        bbox_motion_train = make_motion_features(normalized_bbox_train)  # List[np.ndarray], 每个 (T_i,4)
        bbox_motion_valid = make_motion_features(normalized_bbox_valid)
        bbox_motion_test = make_motion_features(normalized_bbox_test)
        ###

        normalized_bbox_dec_train = normalize_traj(bbox_dec_train)  # 归一化轨迹
        normalized_bbox_dec_valid = normalize_traj(bbox_dec_valid)
        normalized_bbox_dec_test = normalize_traj(bbox_dec_test)

        label_action_train = prepare_label(action_train)  # 准备标签
        label_action_valid = prepare_label(action_valid)
        label_action_test = prepare_label(action_test)

        X_train, X_valid = torch.Tensor(normalized_bbox_train), torch.Tensor(normalized_bbox_valid)  # 转换为tensor
        Y_train, Y_valid = torch.Tensor(label_action_train), torch.Tensor(label_action_valid)
        X_test = torch.Tensor(normalized_bbox_test)
        Y_test = torch.Tensor(label_action_test)

        ###新加的bbox_motion
        X_motion_train = torch.Tensor(bbox_motion_train)
        X_motion_valid = torch.Tensor(bbox_motion_valid)
        X_motion_test = torch.Tensor(bbox_motion_test)
        ###

        temp = pad_sequence(normalized_bbox_dec_train, 60)
        X_train_dec = torch.Tensor(temp)
        X_valid_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_valid, 60))  # 转换为tensor
        X_test_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_test, 60))

        obd_train, gps_train = torch.Tensor(obd_train), torch.Tensor(gps_train)  # 转换为tensor
        obd_valid, gps_valid = torch.Tensor(obd_valid), torch.Tensor(gps_valid)
        obd_test, gps_test = torch.Tensor(obd_test), torch.Tensor(gps_test)

        vel_train = torch.cat([obd_train, gps_train], dim=-1)  # 拼接obd和gps
        vel_valid = torch.cat([obd_valid, gps_valid], dim=-1)
        vel_test = torch.cat([obd_test, gps_test], dim=-1)

        trainset = TensorDataset(X_train, X_motion_train, Y_train, vel_train, X_train_dec)  # 生成dataset
        validset = TensorDataset(X_valid, X_motion_valid,Y_valid, vel_valid, X_valid_dec)
        testset = TensorDataset(X_test, X_motion_test, Y_test, vel_test, X_test_dec)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)  # 生成dataloader
        valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=1)
    else:  # args.learn为True，不真实训练，生成随机数据。
        train_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)),  # bbox
                         torch.randn(size=(args.batch_size, 1)),  # label
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),  # velocity
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]  # trajectory
        valid_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)),
                         torch.randn(size=(args.batch_size, 1)),
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
        test_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)),
                        torch.randn(size=(args.batch_size, 1)),
                        torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),
                        torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
    print('Start Training Loop... \n')

    model = Model(args)  # 生成模型
    model.to(device)  # 放到gpu上

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)  # 生成优化器
    cls_criterion = nn.BCELoss()  # 生成损失函数 binary cross entropy
    reg_criterion = nn.MSELoss()  # 生成损失函数

    # model_folder_name = args.set_path
    # checkpoint_filepath = 'checkpoints/{}.pt'.format(model_folder_name)  # 生成checkpoint的路径

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

    # checkpoint_filepath = os.path.join(checkpoint_dir, f'{model_folder_name}.pt')
    # writer = SummaryWriter('logs/{}'.format(model_folder_name))  # 生成tensorboard的路径

    # Train
    train(model, train_loader, valid_loader, cls_criterion, reg_criterion, optimizer, checkpoint_filepath, writer,
          args=args)

    # #Test
    model = Model(args)
    model.to(device)

    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    preds, labels = test(model, test_loader)
    pred_cpu = torch.Tensor.cpu(preds)
    label_cpu = torch.Tensor.cpu(labels)

    acc = accuracy_score(label_cpu, np.round(pred_cpu))
    f1 = f1_score(label_cpu, np.round(pred_cpu))
    pre_s = precision_score(label_cpu, np.round(pred_cpu))
    recall_s = recall_score(label_cpu, np.round(pred_cpu))
    auc = roc_auc_score(label_cpu, np.round(pred_cpu))
    contrix = confusion_matrix(label_cpu, np.round(pred_cpu))
    tn, fp, fn, tp = confusion_matrix(label_cpu, np.round(pred_cpu)).ravel()



    print(
        f'Acc: {acc}\n f1: {f1}\n precision_score: {pre_s}\n recall_score: {recall_s}\n roc_auc_score: {auc}\n confusion_matrix: {contrix}')



    #################################### 自动保持excel文件
    # log to Excel
    excel_path = os.path.join(checkpoint_dir, 'experiments_location.xlsx')

    record = {
        'checkpoint': checkpoint_filename,
        'location': model_folder_name + '.pt',
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
    parser.add_argument('--set_path', type=str, default='/media/robert/4TB-SSD/PIE-master')
    parser.add_argument('--balance', type=bool, default=True, help='balance or not for test dataset.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=128, help='the dimension after embedding.')
    parser.add_argument('--dff', type=int, default=256, help='the number of the units.')
    parser.add_argument('--num_heads', type=int, default=8, help='number of the heads of the multi-head model.')
    parser.add_argument('--bbox_input', type=int, default=4, help='dimension of bbox.')
    parser.add_argument('--vel_input', type=int, default=2, help='dimension of velocity.')
    parser.add_argument('--time_crop', type=bool, default=False)  # 是否使用随机时间裁剪
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch.')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train.')
    parser.add_argument('--num_layers', type=int, default=4, help='the number of layers.')
    parser.add_argument('--times_num', type=int, default=16, help='')  # 数据的时间维度
    parser.add_argument('--num_bnks', type=int, default=3, help='')  # 瓶颈结构的单元数目
    parser.add_argument('--bnks_layers', type=int, default=7, help='')  # 瓶颈结构的层数
    parser.add_argument('--sta_f', type=int, default=8)  # 若采用随机时间裁剪，则从sta_f到end_f中随机选取一个时间点作为保留的时间段。
    parser.add_argument('--end_f', type=int, default=12)
    parser.add_argument('--learn', type=bool, default=False)  # 是否跳过真实数据读取，生成尺寸相同的随机数据。# 目的如果是为了了解项目的运行过程，则可以将learn设置为True，这样可以跳过真实数据读取，生成尺寸相同的随机数据。
    parser.add_argument('--test_only', action='store_true',help='只做测试，加载 --checkpoint 并 eval')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/PIE-master_07.pt',help='测试模式下指定 .pt 文件路径')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 概率')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay (L2 regularization) factor.')
    parser.add_argument('--time_transformer_num_heads', type=int, default=3, help='Number of heads for the TimeTransformer module.')
    parser.add_argument('--time_transformer_dropout', type=float, default=0.5, help='Dropout rate for the TimeTransformer module.')
    parser.add_argument('--no-balance', dest='balance', action='store_false', default=True, help='禁用正负样本平衡，直接使用原始分布的训练/验证样本' )


    args = parser.parse_args()
    main(args)
