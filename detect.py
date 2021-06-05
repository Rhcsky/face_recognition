import cv2
import face_recognition


def crop_face(img, location, adj):
    top, right, bottom, left = location
    return img[top:bottom, left:right]


adjust = 100

img = cv2.imread('facebank/images/seungmin/KakaoTalk_20210531_104010527.jpg')
location = face_recognition.face_locations(img, model='hog')  # top, right, bottom, left
location = location[0]
cv2.rectangle(img, (location[3], location[0]), (location[1], location[2]), (0, 0, 255), 1)
face_region = crop_face(img, location, adjust)
cv2.imwrite("rectangle.jpg", img)
cv2.imwrite("face.jpg", face_region)
