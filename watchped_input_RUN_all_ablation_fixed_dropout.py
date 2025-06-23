#!/usr/bin/env python3
# run_all_ablation.py

import subprocess
import sys
import os

# 为每个脚本指定它要跑的dropout列表
scripts_config = [
    # 只跑一次 0.1
    ("/home/robert/桌面/PedCMT-main/watchped_input_ablation_acc.py",       [0.1]),
    # 只跑一次 0.3
    ("/home/robert/桌面/PedCMT-main/watchped_input_ablation_velocity.py",[0.3]),
    # 先跑 0.1 再跑 0.5
    ("/home/robert/桌面/PedCMT-main/watchped_input_ablation_bbox+acc.py",[0.1, 0.5]),
    # 你可以继续为第四个脚本指定它的dropout列表
    ("/home/robert/桌面/PedCMT-main/watchped_input_ablation_acc+velocity.py",[0.3]),
]

def run_script(script_path, dropout):
    if not os.path.isfile(script_path):
        print(f"[ERROR] 找不到脚本：{script_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        script_path,
        "--time_transformer_dropout", str(dropout)
    ]
    print(f"\n>>> 正在运行：{' '.join(cmd)}")

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[ERROR] 脚本 {script_path} (dropout={dropout}) 以状态码 {proc.returncode} 退出，停止后续执行。")
        sys.exit(proc.returncode)

    print(f">>> 完成：{script_path} (dropout={dropout})")

def main():
    for script_path, dropout_list in scripts_config:
        for d in dropout_list:
            run_script(script_path, d)
    print("\n所有脚本已执行完毕！")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# run_all_ablation.py

# import subprocess
# import sys
# import os
#
# # 为每个脚本指定它要跑的dropout列表
# scripts_config = [
#     # 只跑一次 0.1
#     ("/home/robert/桌面/PedCMT-main/pie_bbox+bbox_motion.py",[0.1])
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
#         "--time_transformer_dropout", str(dropout)
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
