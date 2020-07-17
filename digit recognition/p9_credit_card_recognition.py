# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:10:49 2020

@author: Administrator
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import myutils


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
FIRST_NUMBER = {
    '3':'American Express',
    '4':'Visa',
    '5':'MasterCard',
    '6':'Discover Card'}

img = cv2.imread('ocr_a_reference.png') ## read the template
cv_show('img',img)


## make it gray
ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)


## make it binary
ref = cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)

## compute contour
refCnts, hierarchy =cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,refCnts,-1,(0,0,255),3)
cv_show('img',img)
print(np.array(refCnts).shape)  # contour count
refCnts =  myutils.sort_contours(refCnts,method='left-to-right')[0]

digits={}

for (i,c) in enumerate(refCnts):
    ### compute the bouding rectangle and resize into appropriate size
    (x,y,w,h) = cv2.boundingRect(c)
    roi = ref[y:y+h, x:x+w]
    roi = cv2.resize(roi,(57,88))

    digits[i] = roi
   
    
### initilize convolution kernel size
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))


#image = cv2.imread('test_pics/test_54.png')
image = cv2.imread('test_pics/credit_card_03.png')
cv_show('image',image)
image = myutils.resize(image,width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

###  tophat = original - open operation
## to enhance the bright area
tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
cv_show('tophat',tophat)


gradX = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)  ## ksize=-1 means 3*3
gradX = np.absolute(gradX)
(minVal,maxVal) = (np.min(gradX),np.max(gradX))
gradX = (255 * ((gradX- minVal) / (maxVal- minVal)))
gradX = gradX.astype('uint8')

print(np.array(gradX).shape)
cv_show('gradX',gradX)




####  to make chunks  -> 'xxxx xxxx xxxx xxxx'  4 chunks
gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel) 
cv_show('chunks',gradX)

thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] ## OTSU detect threshold automatically
cv_show('thresh',thresh)


#### do the close operation one more time to fill the gap between numbers
thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
cv_show('thresh',thresh)

threshCnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3) ### -1 means all contours
cv_show('img',cur_img)


locs = []
#### to filter the contours and find the 4 chunks
for (i,c) in enumerate (cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    ar =  w / float(h)

    if ar >2.5 and ar < 4.0:
        if( w > 40 and w <55) and (h > 10 and  h < 20):
            locs.append((x,y,w,h))
            
locs = sorted(locs,key=lambda x:x[0])
output = []


### detect each number in each chunk
for (i, (gX,gY,gW,gH)) in enumerate(locs):
    groupOutput = []
    
    group = gray[gY-5:gY+gH+5,gX-5:gX+gW+5]
    cv_show('group',group)
    
    ### in order to detect numbers, we need to do the pre-processing
    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group',group)
    
    digitCnts,hierarchy = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = myutils.sort_contours(digitCnts,method='left-to-right')[0]
    
    
    for c in digitCnts:
        (x,y,w,h) =cv2.boundingRect(c)
        roi = group[y:y+h,x:x+w]
        roi = cv2.resize(roi,(57,88))
        cv_show('roi',roi)
        
        scores = []
        
        for ( digit,digitROI) in digits.items():
            ## match each number with template
            result = cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
            (_,score,_,_) = cv2.minMaxLoc(result)
            scores.append(score)
         
        groupOutput.append(str(np.argmax(scores)))
    
    ### draw the number
    cv2.rectangle(image,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gX,gY-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
    
    output.extend(groupOutput)
    
#print('card type:{}'.format(FIRST_NUMBER[output[0]]))
print('card number: {}'.format(''.join(output)))
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()    
    
















