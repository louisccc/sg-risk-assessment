import numpy as np
import os

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > gpu_stats')
    memory_available = [int(x.split()[2]) for x in open('gpu_stats', 'r').readlines()]
    i = 0
    while max(memory_available) < 2000:
        if i % 1000 == 0:
            print("All GPUs have less than 2GB RAM")
        i += 1
        #raise Exception("All GPUs have less than 2GB RAM")
    print('GPU memory in MB: {}'.format(memory_available))
    return str(np.argmax(memory_available))


if __name__ == "__main__":
    stats = get_free_gpu()
    print(stats)
    print(type(stats))
