from easyocr import Reader
import cv2
def cleanup_text(text):
    '''nếu kí tự đưa vào không nằm trong bảng ACII thì không put text được '''
    return "".join([c if ord(c)<128 else "" for c in text]).strip()
'''nếu không đưa tham số nào vào strip thì mặc định là dấu cách,
nếu text đưa vào là 1 câu thì sẽ tách thành các chữ, return ra 1 chuỗi gồm các chữ'''
lang=['en']
image=cv2.imread('image.jpg')
reader=Reader(lang,gpu=True)
results=reader.readtext(image)
for (bbox,text,prob) in results:
    print(f'{prob}:{text}')
    (tl,tr,br,bl)=bbox
    tl=(int(tl[0]),int(tl[1]))
    tr=(int(tr[0]),int(tr[1]))
    br=(int(br[0]),int(br[1]))
    bl=(int(bl[0]),int(bl[1]))
    text=cleanup_text(text)
    with open('result.txt','a') as f:
        f.write(text)
        f.close()
cv2.imshow('image',image)
cv2.waitKey(0)
