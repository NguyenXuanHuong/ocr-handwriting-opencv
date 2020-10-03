import cv2
import imutils
import matplotlib.pyplot as plt

image=cv2.imread('image.jpg')
image=imutils.resize(image,width=500)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(15,15),0)
'''hạn chế bớt các cạnh'''
edge=cv2.Canny(gray,30,225)
# plt.style.use('grayscale')
# plt.subplot(211)
# plt.imshow(thresh)
# plt.subplot(212)
# plt.imshow(erode)
# plt.show()
cnts=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
# cnts=sorted(cnts,key=cv2.contourArea,reverse=False)
for c in cnts:
    (x,y,w,h)=cv2.boundingRect(c)

    rectRatio=w/float(h)
    if (rectRatio> 1 and rectRatio<1.8):
        contourArea=cv2.contourArea(c)
        '''nhìn thấy ảnh to thì dùng area không dùng paramiter'''
        if contourArea>13000 and contourArea<15000:
            roi=image[y:y+h,x:x+w]
            rect = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


