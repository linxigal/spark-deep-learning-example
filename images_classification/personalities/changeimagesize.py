import os
import time
from PIL import Image

def alter(path,object):
    s = os.listdir(path)
    count = 1
    for i in s:
        document = os.path.join(path,i)
        img = Image.open(document)
        out = img.resize((200,200))
        listStr = [str(int(time.time())), str(count)]
        fileName = ''.join(listStr)
        out.save(object+os.sep+'yiyi%s.jpg' % str(count))
        count = count + 1

alter('./yiyi','./yiyi_new')
