import cv2
import numpy as np
import os

def search_files(path):
    archives = []
    count = 0
    for files in os.listdir(path):
        if os.path.isfile(os.path.join(path, files)):
            if "." not in files:
                count += 1
                archives.append(files)
    return archives, count

if __name__ == "__main__":
    
    pass