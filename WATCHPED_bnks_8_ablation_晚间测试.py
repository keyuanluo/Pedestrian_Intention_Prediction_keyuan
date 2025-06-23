# #!/usr/bin/env python3
# # run_all_ablation.py
# ###############################单独drop_out
# import subprocess
# import sys
# import os
#
# # 为每个脚本指定它要跑的dropout列表
# scripts_config = [
#     # JAAD BBOX Bbox_motion
#     ("/home/robert/桌面/PedCMT-main/watchped_ablation_different_layers.py", [1,2,3,4,5,6,7,8,9])
# ]
#
# def run_script(script_path, dropout):
#     if not os.path.isfile(script_path):
#         print(f"[ERROR] 找不到脚本：{script_path}")
#         sys.exit(1)
#
#     cmd = [
#         sys.executable,
#         script_path,
#         "--num_layers", str(dropout)
#     ]
#     print(f"\n>>> 正在运行：{' '.join(cmd)}")
#
#     proc = subprocess.run(cmd)
#     if proc.returncode != 0:
#         print(f"[ERROR] 脚本 {script_path} (dropout={dropout}) 以状态码 {proc.returncode} 退出，停止后续执行。")
#         sys.exit(proc.returncode)
#
#     print(f">>> 完成：{script_path} (dropout={dropout})")
#
# def main():
#     for script_path, dropout_list in scripts_config:
#         for d in dropout_list:
#             run_script(script_path, d)
#     print("\n所有脚本已执行完毕！")
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# run_all_ablation.py
#####################################6月2号晚间
# import subprocess
# import sys
# import os
#
# # 要运行的脚本路径（假设三种消融都调用同一个脚本）
# SCRIPT_PATH = "/home/robert/桌面/PedCMT-main/watchped_ablation_different_layers.py"
#
# # 定义三个消融实验，每个实验对应一个命令行参数名，以及它要遍历的取值范围（1 到 9）
# ablations = [
#     # ("--num_layers",   list(range(1, 10))),  # 从 1 到 9
#     ("--num_bnks",     [3, 6, 9,12,15,18,21,24,27]),  # 从 1 到 9
#     ("--bnks_layers",  list(range(1, 10))),  # 从 1 到 9
# ]
#
#
# def run_script(script_path, flag_name, flag_value):
#     if not os.path.isfile(script_path):
#         print(f"[ERROR] 找不到脚本：{script_path}")
#         sys.exit(1)
#
#     cmd = [
#         sys.executable,
#         script_path,
#         flag_name, str(flag_value)
#     ]
#     print(f"\n>>> 正在运行：{' '.join(cmd)}")
#
#     proc = subprocess.run(cmd)
#     if proc.returncode != 0:
#         print(f"[ERROR] 脚本 {script_path} ({flag_name}={flag_value}) 以状态码 {proc.returncode} 退出，停止后续执行。")
#         sys.exit(proc.returncode)
#
#     print(f">>> 完成：{script_path} ({flag_name}={flag_value})")
#
#
# def main():
#     for flag_name, values in ablations:
#         print(f"\n===== 开始消融实验：只改变 {flag_name} 从 1 到 9 =====")
#         for v in values:
#             run_script(SCRIPT_PATH, flag_name, v)
#
#     print("\n所有消融实验已执行完毕！")
#
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# run_all_ablation.py
################6月3号晚间
import subprocess
import sys
import os

# 脚本及对应要遍历的 --bnks_layers 范围
scripts_config = [
    # 第一个脚本，只跑 --bnks_layers 从 6 到 9
    (
        "/home/robert/桌面/PedCMT-main/watchped_ablation_different_layers.py",
        "--bnks_layers",
        list(range(10, 16))  # [6,7,8,9]
    ),
#     # # 第二个脚本，跑 --bnks_layers 从 1 到 9
#     # (
#     #     "/home/robert/桌面/PedCMT-main/watchped_ablation_different_layers_12_num_bnks.py",
#     #     "--bnks_layers",
#     #     list(range(36,37 ))  # [1,2,3,4,5,6,7,8,9]
#     ),
]

def run_script(script_path, flag_name, flag_value):
    if not os.path.isfile(script_path):
        print(f"[ERROR] 找不到脚本：{script_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        script_path,
        flag_name,
        str(flag_value)
    ]
    print(f"\n>>> 正在运行：{' '.join(cmd)}")

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[ERROR] 脚本 {script_path} ({flag_name}={flag_value}) 以状态码 {proc.returncode} 退出，停止后续执行。")
        sys.exit(proc.returncode)

    print(f">>> 完成：{script_path} ({flag_name}={flag_value})")

def main():
    for script_path, flag_name, values in scripts_config:
        print(f"\n===== 开始对脚本 {os.path.basename(script_path)} 进行消融：{flag_name} → {values} =====")
        for v in values:
            run_script(script_path, flag_name, v)

    print("\n所有消融实验已执行完毕！")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# # run_all_ablation.py
# ##############################6月5号晚间测试
# import subprocess
# import sys
# import os
#
# # 配置需要批量跑的实验
# # 1. watchped_ablation_different_weight_bnks_8.py 用不同的 weight_decay 运行
# # 2. watchped_ablation_different_weight_bnks_8.py 用不同的 time_transformer_dropout 运行
# # 3. 最后分别运行下面 6 个脚本（只跑一次，使用默认参数）
# # scripts_config = [
# #     # 1) 同一脚本下，不同 weight_decay
# #     (
# #         "/home/robert/桌面/PedCMT-main/watchped_ablation_different_weight_bnks_8.py",
# #         "--weight_decay",
# #         [1e-4, 1e-5, 1e-6]
# #     ),
# #     # 2) 同一脚本下，不同 time_transformer_dropout
# #     (
# #         "/home/robert/桌面/PedCMT-main/watchped_ablation_different_weight_bnks_8.py",
# #         "--time_transformer_dropout",
# #         [0.1, 0.2, 0.3, 0.4]
# #     ),
# # ]
#
# # 最后需要单独跑一次的脚本列表（不带额外可变参数）
# single_run_scripts = [
#     "/home/robert/桌面/PedCMT-main/watchped_input_ablation_acc+velocity.py",
#     "/home/robert/桌面/PedCMT-main/watchped_input_ablation_acc.py",
#     "/home/robert/桌面/PedCMT-main/watchped_input_ablation_bbox+acc.py",
#     "/home/robert/桌面/PedCMT-main/watchped_input_ablation_bbox+velocity.py",
#     "/home/robert/桌面/PedCMT-main/watchped_input_ablation_bbox.py",
#     "/home/robert/桌面/PedCMT-main/watchped_input_ablation_velocity.py",
# ]
#
# def run_script_with_flag(script_path, flag_name, flag_value):
#     """
#     以带单个可变参数的方式运行脚本：
#       python3 script_path flag_name flag_value
#     """
#     if not os.path.isfile(script_path):
#         print(f"[ERROR] 找不到脚本：{script_path}")
#         sys.exit(1)
#
#     cmd = [sys.executable, script_path, flag_name, str(flag_value)]
#     print(f"\n>>> 正在运行：{' '.join(cmd)}")
#     proc = subprocess.run(cmd)
#     if proc.returncode != 0:
#         print(f"[ERROR] 脚本 {script_path} ({flag_name}={flag_value}) 以状态码 {proc.returncode} 退出，停止后续执行。")
#         sys.exit(proc.returncode)
#     print(f">>> 完成：{script_path} ({flag_name}={flag_value})")
#
# def run_script_once(script_path):
#     """
#     以不带额外可变参数的方式运行脚本（只调用一次，使用脚本内部默认参数）。
#     """
#     if not os.path.isfile(script_path):
#         print(f"[ERROR] 找不到脚本：{script_path}")
#         sys.exit(1)
#
#     cmd = [sys.executable, script_path]
#     print(f"\n>>> 正在运行（单次执行）：{' '.join(cmd)}")
#     proc = subprocess.run(cmd)
#     if proc.returncode != 0:
#         print(f"[ERROR] 脚本 {script_path} 单次执行以状态码 {proc.returncode} 退出，停止后续执行。")
#         sys.exit(proc.returncode)
#     print(f">>> 完成（单次执行）：{script_path}")
#
# def main():
#     # # 先跑带可变参数的实验
#     # for script_path, flag_name, values in scripts_config:
#     #     print(f"\n===== 开始对脚本 {os.path.basename(script_path)} 进行批量实验：{flag_name} → {values} =====")
#     #     for v in values:
#     #         run_script_with_flag(script_path, flag_name, v)
#
#     # 然后逐一跑 6 个脚本（只跑一次，使用默认参数）
#     print("\n===== 开始对下面脚本进行单次执行（使用默认参数） =====")
#     for script_path in single_run_scripts:
#         run_script_once(script_path)
#
#     print("\n所有实验已执行完毕！")
#
# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # run_all_ablation.py
#
# import subprocess
# import sys
# import os
#
# # 为 /home/robert/桌面/PedCMT-main/jaad_bbox+bbox_motion.py 指定要跑的 batch_size 列表
# scripts_config = [
#     ("/home/robert/桌面/PedCMT-main/jaad_bbox+bbox_motion.py", [2, 4])
# ]
#
# def run_script(script_path, batch_size):
#     if not os.path.isfile(script_path):
#         print(f"[ERROR] 找不到脚本：{script_path}")
#         sys.exit(1)
#
#     cmd = [
#         sys.executable,
#         script_path,
#         "--batch_size", str(batch_size)
#     ]
#     print(f"\n>>> 正在运行：{' '.join(cmd)}")
#
#     proc = subprocess.run(cmd)
#     if proc.returncode != 0:
#         print(f"[ERROR] 脚本 {script_path} (batch_size={batch_size}) 以状态码 {proc.returncode} 退出，停止后续执行。")
#         sys.exit(proc.returncode)
#
#     print(f">>> 完成：{script_path} (batch_size={batch_size})")
#
# def main():
#     for script_path, batch_list in scripts_config:
#         for b in batch_list:
#             run_script(script_path, b)
#     print("\n所有脚本已执行完毕！")
#
# if __name__ == "__main__":
#     main()
