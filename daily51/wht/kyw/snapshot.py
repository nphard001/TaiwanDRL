import os
import psutil


def get_file_snapshot(walk_start = "C:\\"):
    data = []
    for root, dirs, files in os.walk(walk_start):
        data.append([root, len(dirs), len(files)])
    return data

# source: https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/356516/
PROC_TAGS = [
    "name", # p.name()  #程序名
    "exe", # p.exe()  #程序的bin路徑
#     "cwd", # p.cwd()  #程序的工作目錄絕對路徑
    "status", # p.status()  #程序狀態
    "create_time", # p.create_time() #程序建立時間
#     "uids", # p.uids()  #程序uid資訊
#     "gids", # p.gids()  #程序的gid資訊
    "cpu_times", # p.cpu_times()  #程序的cpu時間資訊,包括user,system兩個cpu資訊
    "cpu_affinity", # p.cpu_affinity() #get程序cpu親和度,如果要設定cpu親和度,將cpu號作為參考就好
    "cpu_percent", # 
    "memory_percent", # p.memory_percent() #程序記憶體利用率
    "memory_info", # p.memory_info()  #程序記憶體rss,vms資訊
    "io_counters", # p.io_counters()  #程序的IO資訊,包括讀寫IO數字及引數
    "num_threads", # p.num_threads() #程序開啟的執行緒數
]


def get_proc_snapshot():
    data = []
    for proc in psutil.process_iter():
        try:
            pid = proc.pid
            row = {"pid": pid}
            for tag in PROC_TAGS:
                row[tag] = str(eval(f"proc.{tag}()"))
            data.append(row)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print("!!! EXCEPTION !!!")
            print(e)
    return data
