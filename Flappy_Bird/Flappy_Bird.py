import cv2
import os
import copy
import math
import random
import time

import mediapipe as mp
import numpy as np


def check_dir():
    '''Checks if the Captured Video directory exists'''

    path = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Captured Video\\'
    if not os.path.exists(path):
        os.makedirs(path)


def get_num() -> int:
    '''Gets the no. of the last recorded file'''

    num = -1

    for file in os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Captured Video\\'):
        if os.path.splitext(file)[-1] == '.mp4':
            if 'record_' in file:
                file = file.replace('.mp4', '')
                t_num = int(file.split('_')[1])
                if t_num > num:
                    num = t_num
    
    return num


def mask(img):
    '''Masks the image'''

    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

    return mask


def rotate_image(image, angle):
    '''Rotates the image by a certain angle'''

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def calculate_angle(a, b, c) -> int:
    '''Calculates the angle'''

    angle = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    
    return round(angle)


if __name__ == '__main__':
    # Defaults
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    _, frame = vid.read()
    h, w, _ = frame.shape

    vid_cod = cv2.VideoWriter_fourcc(*'mp4v') # .mp4

    check_dir()
    i = get_num()
    fps = 20
    save_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Captured Video\\', f'record_{i+1}.mp4')
    output = cv2.VideoWriter(save_path, vid_cod, fps, (640,480))

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp.solutions.face_detection.FaceDetection()

    nose_tip_x, nose_tip_y = None, None
    prev_nose_tip_x, prev_nose_tip_y = None, None
    prev_flappy = None

    fl = True

    pillar_val = 1
    pillar_num = 3

    pillar_list = [None]*pillar_num
    mask_pillar_list = [None]*pillar_num
    pillar_point_list = [False]*pillar_num

    pv_list = [525]*pillar_num
    pillar_resize = [0]*pillar_num

    flappy_coords = []

    flappy_size = 100
    angle = 0
    dest_angle = 0

    score = 0
    
    start = False
    finish = False

    buffer = time.time()

    # File path
    flappy_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Data\\', 'Flappy Bird.png')
    pillar_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Data\\', 'Pillar.png')

    # Image
    flappy = cv2.imread(flappy_path)
    flappy = cv2.resize(flappy, (flappy_size, flappy_size))
    mask_flappy = mask(flappy)

    originial_flappy = copy.deepcopy(flappy)

    original_pillar = cv2.imread(pillar_path)
    p_h, p_w, _ = original_pillar.shape

    while True:
        if (time.time() - buffer) > 3: start = True

        _, frame = vid.read() # Gets the video from the camera
        frame = cv2.flip(frame, 1) # Flips the camera

        face_frame = copy.deepcopy(frame) # Face detection frame

        # Face Detection
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face = face_detection.process(face_frame)

        if pv_list[pillar_val-1] <= 350 and fl:
            pillar_val += 1
        if pillar_val > len(pillar_list): 
            pillar_val = len(pillar_list)
            fl = False

        for i, pillar in enumerate(pillar_list[:pillar_val]):
            if pillar is None:
                random_resize = random.randint(50, 260)
                pillar_resize[i] = random_resize
                pillar = copy.deepcopy(original_pillar)

                pillar = pillar[random_resize: random_resize+480, 0: p_w]
                pillar = cv2.resize(pillar, (p_w, h))
                mask_pillar = mask(pillar)
            
            pv_val = pv_list[i]

            # Death
            pillar_exc_y = set(range((280-pillar_resize[i]), (490-pillar_resize[i])+1))
            pillar_all_y = set(range(0, h+1))

            pillar_x = list(range(pv_val+10, pv_val + 100))
            pillar_y = list(pillar_all_y - pillar_exc_y)

            if flappy_coords:
                begin, end = flappy_coords
                begin_x, begin_y = begin
                end_x, end_y = end

                flappy_x = list(range(begin_x+15, end_x-15))
                flappy_y = list(range(begin_y+20, end_y-15))

                if any(val in pillar_x for val in flappy_x) and any(val in pillar_y for val in flappy_y):
                    finish = True

            # Score
            if pillar_point_list[i] is False and finish is False and start:
                if flappy_coords:
                    if (flappy_coords[1][0] - 80) > pv_val:
                        pillar_point_list[i] = True
                        score += 1
            
            if mask_pillar_list[i] is not None:
                mask_pillar = mask_pillar_list[i]

            rois = frame[0:h, pv_val: p_w+pv_val]
            rois[np.where(mask_pillar)] = 0
            rois += pillar

            pillar_list[i] = pillar
            mask_pillar_list[i] = mask_pillar

            if finish is False and start:
                pv_list[i] -= 10

            if pv_list[i] <= 0: 
                pv_list[i] = 525
                pillar_resize[i] = 0

                mask_pillar_list[i] = None
                pillar_list[i] = None

                pillar_point_list[i] = False

        if face.detections:
            for detection in face.detections:
                face_data = detection.location_data
                nose_tip = face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(2).value]

                if nose_tip and finish is False: # Updates only if it detects nose
                    nose_tip_x = round(nose_tip.x * w)
                    nose_tip_y = round(nose_tip.y * h)

                    if not w - flappy_size + 200 > nose_tip_x > 200: nose_tip_x = prev_nose_tip_x
                    if not h - flappy_size + 55 > nose_tip_y > 55: nose_tip_y = prev_nose_tip_y

                # Rotates the flappy bird
                if prev_flappy is not None:
                    flappy = originial_flappy

                    if angle > dest_angle: angle -= 5
                    elif angle < dest_angle: angle += 5

                    if abs(prev_nose_tip_x - nose_tip_x) > 3 or abs(prev_nose_tip_y - nose_tip_y) > 3:
                        dest_angle = calculate_angle([nose_tip_x, nose_tip_y], prev_flappy, [prev_nose_tip_x, prev_nose_tip_y]) * 10
                        if dest_angle > 45: dest_angle = 45
                        elif dest_angle < -45 : dest_angle = -45
                    else:
                        dest_angle = 0

                    flappy = rotate_image(flappy, angle)
                    mask_flappy = mask(flappy)
        
        # Displays score
        txt = 'SCORE: '
        cv2.putText(frame, txt, (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, False)
        cv2.putText(frame, str(score), (380, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, False)

        if nose_tip_x is not None and nose_tip_y is not None:
            prev_nose_tip_x = nose_tip_x
            prev_nose_tip_y = nose_tip_y
            prev_flappy = [prev_nose_tip_x-100, prev_nose_tip_y-25]

            flappy_coords = [[nose_tip_x-200, nose_tip_y-55], [nose_tip_x-200+flappy_size, nose_tip_y-55+flappy_size]]

            # Masks the flappy bird image
            roi = frame[nose_tip_y-55: nose_tip_y-55+flappy_size, nose_tip_x-200: nose_tip_x-200+flappy_size]
  
            # Set an index of where the mask is
            roi[np.where(mask_flappy)] = 0
            roi += flappy

        # Shows and records the video
        cv2.imshow('Cam', frame)
        output.write(frame)

        # Closes the window
        # Q button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Esc button
        if cv2.waitKey(1) == 27:
            break

        # X button on the top of the window
        if cv2.getWindowProperty('Cam', cv2.WND_PROP_VISIBLE) < 1:
            break


    vid.release()
    output.release()
    cv2.destroyAllWindows()
