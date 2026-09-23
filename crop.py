import sys
from PIL import Image
src,out=sys.argv[1],sys.argv[2]
x0,y0,x1,y1=map(int,sys.argv[3:7])
im=Image.open(src)
im.crop((x0,y0,x1,y1)).save(out)
