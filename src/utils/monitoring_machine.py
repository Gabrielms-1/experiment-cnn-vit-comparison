import time
import subprocess
import psutil
from datetime import datetime

INTERVAL = 15  # 15 seconds

with open("data/metrics_log.csv", "w") as f:
    f.write("Timestamp,GPU_Util(%),GPU_Mem_Util(%),GPU_Mem_Used(MB),GPU_Mem_Total(MB),CPU_Usage(%),Mem_Usage(%),Mem_Used(MB),Mem_Total(MB)\n")

while True:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cmd = [
        "nvidia-smi", 
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]
    result = subprocess.check_output(cmd).decode("utf-8").strip()
    gpu_stats = result.split(", ")  
    
    gpu_util = float(gpu_stats[0])
    gpu_mem_util = float(gpu_stats[1])
    gpu_mem_used = float(gpu_stats[2])
    gpu_mem_total = float(gpu_stats[3])

    cpu_usage = psutil.cpu_percent(interval=None)

    mem = psutil.virtual_memory()
    mem_usage_percent = mem.percent
    mem_used = mem.used / (1024*1024)
    mem_total = mem.total / (1024*1024)

    metrics = (
        f"Timestamp: {now}, \n"
        f"GPU Util: {gpu_util}, \n"
        f"GPU Mem Util: {gpu_mem_util}, \n"
        f"GPU Mem Used: {gpu_mem_used}, \n"
        f"GPU Mem Total: {gpu_mem_total}, \n"
        f"CPU Usage: {cpu_usage}, \n"
        f"Mem Usage: {mem_usage_percent}, \n"
        f"Mem Used: {mem_used:.2f}, \n"
        f"Mem Total: {mem_total:.2f} \n"
    )

    with open("data/metrics_log.csv", "a") as f:
        f.write(metrics + "\n")

    print(metrics)

    time.sleep(INTERVAL)
