from imutils import paths
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
rectkernel=cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
sqkernel=cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
image=cv2.imread('image.jpg')
image=imutils.resize(image,height=600)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(3,3),0)
blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,rectkernel)
'''kernel sẽ  quyết định kết quả, nếu kết quả không như ý thì thay đổi kích thước của kernek'''
gradx=cv2.Sobel(blackhat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
'''nếu chuyển từ đen sang trắng thì gradx có giá trị âm, nếu để cv2.u8int thì số âm sẽ =0 nên mất cạnh
do đó phải daungf cv2.32F, sau đó dùng absolute để chuyển số âm thành số dương, sau đó có thể dùng lại u8int'''
gradx=np.absolute(gradx)
'''gradx không nằm trong khoảng 0 đến 255, max= 1600, không dùng được max (), phải dùng np.max()'''
(minval,maxval)=(np.min(gradx),np.max(gradx))
gradx=(255*((gradx-minval)/(maxval-minval))).astype('uint8')
# plt.style.use('grayscale')
# # gradx=cv2.cvtColor(gradx,cv2.COLOR_BGR2RGB) # cách 2
# '''trong matplotlib hình ảnh ở dạng bgr cần chuyển thành dạng grb'''
# plt.subplot(211,xticks=[],yticks=[])
# '''x ticks: bỏ đi các tọa độ bên côt x và cột y'''
# plt.imshow(blackhat)
# plt.subplot(212,xticks=[],yticks=[])
# plt.imshow(gradx)
# plt.show()
close=cv2.morphologyEx(gradx,cv2.MORPH_CLOSE,rectkernel)
'''MORPH_CLOSE dùng để mở rộng vùng sáng, thu được khung của mrx
gradx lúc này là ảnh xám không phải ảnh đen trắng nên phải thresh'''
thresh=cv2.threshold(close,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
'''hàm thresh trả về 2 giá trị ret và thresh nên chỉ lấy cái thứ 2'''
erodekernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
erode=cv2.erode(thresh,erodekernel,iterations=1)


cnts=cv2.findContours(erode.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
'''RETREIVE_EXTERNAL: chỉ lấy contours ngoài cùng, xác định viền ngoài cùng
CHAIN_APPROX_SIMPLE: ước lượng 1 hcn thành 4 đỉnh'''
cnts=imutils.grab_contours(cnts)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)
'''cv2.contourArea: chọn ra contour lớn nhất
reverse: bắt đầu đánh số ngược từ dưới lên, bounding box cần tìm sẽ có chỉ số là 0'''
for c in cnts:
    (x,y,w,h)=cv2.boundingRect(c)
    rectRatio=w/float(h)
    rectPaperRatio=w/float(gray.shape[1])
    '''tỉ lệ chiều rộng và chiều dài của hcn lớn hơn 5
    và tỉ lệ cạnh dài của hcn với tờ giấy lớp hơn 0.75'''
    if (rectRatio>3 and rectPaperRatio>0.5):
    # if (rectRatio > 0.5):
        '''pad lại phần bị erode trước đó'''
        px=int((x+w)*0.06)
        py=int((y+h)*0.06)
        (x,y)=(x-px,y-py)
        (w,h)=(w+(px*2),h+(py*2))
        roi=image[y:y+h,x:x+w].copy()
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        print('out')
        break
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.subplot(211)
plt.imshow(image)
plt.subplot(212)
plt.imshow(roi)
plt.show()