import threading
import time
import queue
import multiprocessing

# 全局队列
g_queue = queue.Queue()

# 初始化队列
def init_queue():
    while not g_queue.empty():
        g_queue.get()
    for i in range(50):
        g_queue.put(i)

g_search_list = list(range(10000))

# IO密集型任务
def task_io(task_id):
    print("IOTask[%s] start" % task_id)
    while not g_queue.empty():
        time.sleep(1)
        try:
            data = g_queue.get(timeout=1)
            print("IOTask[%s] got: %s" % (task_id, data))
        except Exception as ex:
            print("IOTask[%s] error: %s" % (task_id, str(ex)))
    print("IOTask[%s] end" % task_id)

# CPU密集型任务
def task_cpu(task_id):
    print("CPUTask[%s] start" % task_id)
    while not g_queue.empty():
        count = 0
        for i in range(50000):
            count += pow(3 * 2, 3 * 2) if i in g_search_list else 0
        try:
            data = g_queue.get(timeout=1)
            print("CPUTask[%s] got: %s" % (task_id, data))
        except Exception as ex:
            print("CPUTask[%s] error: %s" % (task_id, str(ex)))
    print("CPUTask[%s] end" % task_id)


if __name__ == '__main__':
    print("CPU 核心数:", multiprocessing.cpu_count(), "\n")

    print("========== 直接执行 CPU 密集型任务 ==========")
    init_queue()
    t0 = time.time()
    task_cpu(0)
    print("结束：", time.time() - t0, "\n")

    print("========== 多线程执行 CPU 密集型任务 ==========")
    init_queue()
    t0 = time.time()
    thread_list = [
        threading.Thread(target=task_cpu, args=(i,))
        for i in range(10)
    ]
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()
    print("结束：", time.time() - t0, "\n")