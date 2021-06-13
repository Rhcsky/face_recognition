import cv2
import face_recognition

from infer import get_model, inference

inference_args = get_model('face_vector')
new_inference_args = get_model()


def crop_face(img, location):
    top, right, bottom, left = location
    img = img[top:bottom, left:right]

    cv2.imwrite(f'capture/detect_face.jpg', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


frame_rate = 30
flip = 0
width = 1080
height = 1080
adjust = 100

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out_video = cv2.VideoWriter('capture/out.avi', fourcc, frame_rate, (width, height))

video_file = 'videos/sanhak.mp4'

cap = cv2.VideoCapture(video_file, cv2.IMREAD_GRAYSCALE)
i = 0
if cap.isOpened():
    while True:
        ret, img = cap.read()
        print('.',end='')
        if i > 240:
            inference_args = new_inference_args
        if ret:
            location = face_recognition.face_locations(img, model='hog')  # top, right, bottom, left
            if len(location) != 0:
                location = location[0]

                face_image = crop_face(img, location)
                best_prob, idx, ans = inference(face_image, *inference_args)

                cv2.rectangle(img, (location[3], location[0]), (location[1], location[2]), (0, 0, 255), 2)
                if ans != "":
                    cv2.putText(img, f'{ans}', (location[3], location[2] + 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    cv2.imwrite(f'capture/{i}.jpg', img)

            out_video.write(img)
            i += 1
        else:
            break


else:
    print("Can't open file")

out_video.release()
cap.release()
