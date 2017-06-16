import cv2

def resizeImage(img, sizestep=16):
    sy = img.shape[0] 
    sy = sy - sy % sizestep
    sx = img.shape[1]
    sx = sx - sx % sizestep
    img = cv2.resize(img, (sx, sy), interpolation=cv2.INTER_AREA)
    return img