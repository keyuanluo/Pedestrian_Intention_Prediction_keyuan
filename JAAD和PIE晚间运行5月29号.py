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
#     ("/home/robert/桌面/PedCMT-main/jaad_bbox+bbox_motion.py", [0.1,0.2,0.3,0.4,0.5]),
#     # 只跑一次 0.3
#     ("/home/robert/桌面/PedCMT-main/pie_bbox+bbox_motion.py",  [0.1,0.2,0.3,0.4,0.5]),
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

#!/usr/bin/env python3
# run_all_ablation.py
##################################dropout和lr排列组合运行
import subprocess
import sys
import os

# 为每个脚本指定它要跑的dropout列表和 lr 列表
# 格式： (脚本路径, [dropout1,dropout2,…], [lr1,lr2,…])
scripts_config = [
    # JAAD BBOX + Bbox_motion
    # ("/home/robert/桌面/PedCMT-main/jaad_bbox+bbox_motion.py",
    #  [0.1, 0.2, 0.3, 0.4, 0.5],
    #  [0.001, 0.0005,0.0001]),
    # PIE BBOX + Bbox_motion
    ("/home/robert/桌面/PedCMT-main/pie_bbox+bbox_motion.py",
     [0.1, 0.2, 0.3, 0.4, 0.5],
     [0.001, 0.0005,0.0001]),
]

def run_script(script_path, dropout, lr):
    if not os.path.isfile(script_path):
        print(f"[ERROR] 找不到脚本：{script_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        script_path,
        "--time_transformer_dropout", str(dropout),
        "--lr", str(lr)
    ]
    print(f"\n>>> 正在运行：{' '.join(cmd)}")

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[ERROR] 脚本 {script_path} (dropout={dropout}, lr={lr}) 以状态码 {proc.returncode} 退出，停止后续执行。")
        sys.exit(proc.returncode)

    print(f">>> 完成：{script_path} (dropout={dropout}, lr={lr})")

def main():
    for script_path, dropout_list, lr_list in scripts_config:
        for d in dropout_list:
            for lr in lr_list:
                run_script(script_path, d, lr)
    print("\n所有脚本已执行完毕！")

if __name__ == "__main__":
    main()
