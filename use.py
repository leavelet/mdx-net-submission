from genericpath import exists
import sys
import subprocess
import os
import shutil

# for file in os.listdir("./data/toconv"):
#     if(file[0] == '.' or file.endswith("lrc")):
#         continue
#     if file.endswith(".flac"):
#         cmd = "cd /root/autodl-tmp/mns/data/toconv && ffmpeg -i " + "\"" + file + "\"" + " \"" + file.split(".")[-2] + ".wav" + "\" "
#         print(cmd)
#         subprocess.call(cmd, shell=True)
#     if not exists("./data/test/" + file.split(".")[-2]):
#         os.mkdir("./data/test/" + file.split(".")[-2])  
#     shutil.copy("./data/toconv/" + file.split(".")[-2] + ".wav", "./data/test/" + file.split(".")[-2] + "/mixture.wav")

subprocess.call("cd /root/autodl-tmp/mns && python3 predict_blend.py", shell=True)

dicToput = "./data/final/"
if not exists(dicToput):
    os.mkdir(dicToput)

dicOrigin = "./data/results/baseline/"
for dic in os.listdir(dicOrigin):
    if os.path.isdir(dicOrigin + dic):
        os.rename(dicOrigin + dic + "/" + "vocals.wav", dicToput + dic + ".wav")
