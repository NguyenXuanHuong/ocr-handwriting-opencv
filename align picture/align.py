import numpy as np
import imutils
import cv2
def align_images(image,template,maxFeatures=500,keepPercent=0.2,debug=True):
    '''maxFeature: số lượng điểm lớn nhất dùng để làm điểm keyPoints'''
    imageGray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    templateGray=cv2.cvtColor(template,cv2.COLOR_RGB2GRAY)
    orb=cv2.ORB_create(maxFeatures)
    (kpsA,descsA)=orb.detectAndCompute(imageGray,None)
    (kpsB,descsB)=orb.detectAndCompute(templateGray,None)
    method=cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher=cv2.DescriptorMatcher_create(method)
    matches=matcher.match(descsA,descsB,None)
    matches=sorted(matches,key=lambda x:x.distance)
    keep=int(len(matches)*keepPercent)
    matches=matches[:keep]
    if debug:
        matchedVis=cv2.drawMatches(image,kpsA,template,kpsB,matches,None)
        matchedVis=imutils.resize(matchedVis,width=1000)
        cv2.imshow('match',matchedVis)
        cv2.waitKey(27)
    ptsA=np.zeros((len(matches),2),dtype='float')
    ptsB=np.zeros((len(matches),2),dtype='float')
    for(i,m) in enumerate(matches):
        ptsA[i]=kpsA[m.queryIdx].pt
        ptsB[i]=kpsB[m.trainIdx].pt
    (H,mask)=cv2.findHomography(ptsA,ptsB,method=cv2.RANSAC)

    (h,w)=template.shape[:2]
    aligned=cv2.warpPerspective(image,H,(w,h))
    return aligned
iamge=cv2.imread('image.jpg')
template=cv2.imread('template.jpg')
aligned=align_images(iamge,template,debug=False)
aligned=imutils.resize(aligned,width=700)
template=imutils.resize(template,width=700)
stacked=np.hstack([aligned,template])
overlay=template.copy()
output=aligned.copy()
cv2.addWeighted(overlay,0.5,output,0.5,0,output)
cv2.imshow('stacked',stacked)
cv2.imshow('overlay',output)
cv2.waitKey(0)



