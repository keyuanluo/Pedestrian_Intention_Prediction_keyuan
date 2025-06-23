#!/usr/bin/env python3
# run_all_ablation.py

import subprocess
import sys
import os

# 要依次执行的脚本列表
scripts = [
    "/home/robert/桌面/PedCMT-main/watchped_input_ablation_acc.py",
    "/home/robert/桌面/PedCMT-main/watchped_input_ablation_velocity.py",
    "/home/robert/桌面/PedCMT-main/watchped_input_ablation_bbox+acc.py",
    "/home/robert/桌面/PedCMT-main/watchped_input_ablation_acc+velocity.py",
]

# 要额外传入的全局参数
extra_args = [
    "--time_transformer_dropout", "0.3"
]

def run_script(script_path):
    if not os.path.isfile(script_path):
        print(f"[ERROR] 找不到脚本：{script_path}")
        sys.exit(1)

    cmd = [sys.executable, script_path] + extra_args
    print(f"\n>>> 正在运行：{' '.join(cmd)}")

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[ERROR] 脚本 {script_path} 以状态码 {proc.returncode} 退出，停止后续执行。")
        sys.exit(proc.returncode)

    print(f">>> 完成：{script_path}")

def main():
    for script in scripts:
        run_script(script)
    print("\n所有脚本已执行完毕！")

if __name__ == "__main__":
    main()
