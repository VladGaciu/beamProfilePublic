import os, sys
import random
import shutil

path = "/A12/"
dirList = os.listdir(path)

#count = 0
for i in dirList:
    x = random.random()
    
    if x <= 0.25:
        print("Less than 0.25: %s", i)
        shutil.move("/A12/" + str(i), "/1/")
    if x > 0.25 and x <= 0.5:
        print("Greater than 0.25 and less than 0.5: %s", i)
        shutil.move("/A12/" + str(i), "/2/")
    if x > 0.5 and x <= 0.75:
        print("Greater than 0.5 and less than 0.75: %s", i)
        shutil.move("/A12/" + str(i), "/3/")
    if x > 0.75:
        print("Greater than 0.75: %s", i)
        shutil.move("/A12/" + str(i), "/4/")
    #count +=1
#print(count)