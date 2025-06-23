from action_predict import action_prediction
from pie_data import PIE
from jaad_data import JAAD
import os
import sys
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )

def run_one_model(saved_files_path, model_file):
    with open(os.path.join(saved_files_path, 'configs.yaml'), 'r') as yamlfile:
        opts = yaml.safe_load(yamlfile)

    model_opts = opts['model_opts']
    data_opts = opts['data_opts']
    net_opts = opts['net_opts']

    tte = model_opts['time_to_event'] if isinstance(model_opts['time_to_event'], int) else model_opts['time_to_event'][1]
    data_opts['min_track_size'] = model_opts['obs_length'] + tte

    if model_opts['dataset'] == 'pie':
        imdb = PIE(data_path='/home/zzhonghang/Pedestrian_Crossing_Intention_Prediction/data/pie')
    elif model_opts['dataset'] == 'jaad':
        imdb = JAAD(data_path='/home/zzhonghang/Pedestrian_Crossing_Intention_Prediction/JAAD')
    else:
        raise ValueError(f"{model_opts['dataset']} dataset is incorrect")

    method_class = action_prediction(model_opts['model'])(**net_opts)
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

    print(f"\n🚀 正在测试模型: {model_file}")
    model_path = os.path.join(saved_files_path, model_file)

    acc, auc, f1, precision, recall = method_class.test(beh_seq_test, model_path=saved_files_path, model_file=model_file)

    print(f"✅ {model_file} → acc:{acc:.4f} auc:{auc:.4f} f1:{f1:.4f} precision:{precision:.4f} recall:{recall:.4f}")
    return model_file, acc

def test_model(saved_files_path=None):
    model_files = [f for f in os.listdir(saved_files_path)
                   if f.endswith('.h5') and f not in ['best_model.h5']]

    results = []

    if len(model_files) == 1 and model_files[0] == 'model.h5':
        results.append(run_one_model(saved_files_path, 'model.h5'))
    else:
        model_files = sorted(model_files)
        for mf in model_files:
            results.append(run_one_model(saved_files_path, mf))

    # ✅ 汇总结果
    print("\n📊 所有模型测试结果:")
    for name, acc in results:
        print(f"{name}: acc={acc:.4f}")

    best_model, best_acc = max(results, key=lambda x: x[1])
    print(f"\n🏆 最佳模型: {best_model} → acc={best_acc:.4f}")

if __name__ == '__main__':
    saved_files_path = sys.argv[1]
    test_model(saved_files_path)
