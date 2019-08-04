from cv2 import cv2
import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFilter
def angle():
    """
    This method can support defining different perspective angle
    """



def get_perspective_offset(M,w,h):
    """
        We must consider the position offset after perspectived to get correct img size;
        This lambda method is bulit by perspective formula;
        如果原四边形和目标四边形角度差异过大，原四边形内的某些像素会落到无限远的地方
    """
    
    width = 0
    height = 0


    A = np.zeros((3,2*(w+h)))
    i = 0 
    for x in range(h):
        A[:,i] = [x,0,1]
        i += 1
    for x in range(h):
        A[:,i] = [x,w,1]
        i += 1
    for y in range(w):
        A[:,i] = [0,y,1]
        i += 1
    for y in range(w):
        A[:,i] = [h,y,1]
        i += 1

                    
        
    AM = np.dot(M,A)
    AM1 = np.zeros(AM.shape)
    AM1[:2,:] = AM[:2,:] / AM[2,:]
    width = np.max(AM1[0,:])    
    height = np.max(AM1[1,:])

    
    width = int(round(width+0.5))
    height = int(round(height+0.5))
    return width,height


def _apply_func_perspective(image):
    """
        Apply a perspective to an image
    """
    rgb_image = image.convert('RGBA')
    
    img_arr = np.array(rgb_image)

    a=img_arr
    w,h = a.shape[0],a.shape[1]
    if h//w >3:
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGBA2BGRA)
        img = cv2.copyMakeBorder(img,20,20,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
        #img = cv2.imread(img)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGRA2RGBA))
        img = img.resize((48,48),Image.ANTIALIAS)
        return img
    '''
        Set random vertex to target quadrilateral
    '''
    random_flag = random.uniform(0,2)
    if random_flag>1:
        vertex1 = [0,0]
        vertex4 = [random.uniform(1.0000,1.1618)*(w-1),0]
        lens = vertex4[0] - vertex1[0]
        vertex2 = [random.uniform(0.1,0.1618)*(w-1),h-1]
        vertex3 = [vertex2[0]+lens*random.uniform(0.932,1),h-1]
    else:
        vertex4 = [(w-1)*random.uniform(1.0000,1.1618),0]
        vertex1 = [random.uniform(0.1000,0.2618)*(w-1),0]
        lens = vertex4[0] - vertex1[0]
        vertex2 = [random.uniform(0.0000,0.0618)*(w-1),h-1]
        vertex3 = [vertex2[0]+lens*random.uniform(0.932,1),h-1]


    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ])
    pts1 = np.float32([ vertex1,vertex2,vertex3,vertex4 ])
    '''
        get 3*3 transform martix M
    '''
    M = cv2.getPerspectiveTransform(pts,pts1)

    dsize = get_perspective_offset(M,w,h)

    dst = cv2.warpPerspective(a,M,dsize)
    img_arr = np.array(dst)
    img = Image.fromarray(np.uint8(img_arr))

    img = img.resize((48,48),Image.ANTIALIAS)
    return img