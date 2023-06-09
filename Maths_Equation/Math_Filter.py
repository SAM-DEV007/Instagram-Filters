import random
import cv2
import os
import time


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


def generate() -> dict:
    '''Generates dictionary containing the maths equation with answer'''

    # Signs and Placeholders
    sign = ['+', '-', '*', '/']
    placeholder = ['+', '-', 'x', '/']

    #placeholder = ['+', '-', '×', '÷']
    # cv2.putText() does not support non-ascii characters yet (atleast in this version)

    # Equations Dictionary
    dict_eq = dict()

    for _ in range(5):
        eq = ''
        random_step = random.randint(1, 2)
        n = 30

        # Creates the Equation
        for j in range(random_step+1):
            num = random.randint(1, n)
            if eq:
                r_sign = random.sample(sign, k=1)[0]
                r_num = random.randint(1, n - (j*5))

                eq += f' {str(r_sign)}'
                eq += f' {str(r_num)}'
                continue
            eq += str(num)
        
        # Separates the equation and the answer
        pl = ''
        for t in eq:
            if t in sign:
                pl += placeholder[sign.index(t)]
                continue
            pl += t
        
        dict_eq[pl] = round(eval(eq), 2)
    
    return dict_eq


def get_direction(prev_left: int, prev_right: int, x: int, w: int) -> str:
    '''Returns the direction of the face'''

    if 0 not in (prev_left, prev_right):
        if prev_right < (x + w):
            if (x + w) - prev_right > 6:
                return 1
        elif prev_left > x:
            if prev_left - x > 6:
                return -1
    return 0


def generate_options(curr_ques: str, curr_ans: int):
    '''Generates the correct and incorrect option'''

    correct = curr_ans
    if '-' in curr_ques: incorrect = -curr_ans
    else:
        random_no = random.randint(1, 5)
        incorrect = random.sample([correct + random_no, correct - random_no], k=1)[0]
    
    return str(correct), str(incorrect)


def put_text_options(frame, x: int, y: int, w: int, direction: int, left_op: int, right_op: int):
    '''Adds options in the Quiz Area'''

    cv2.putText(frame, left_op, (x - w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA) # Option 1
    cv2.putText(frame, right_op, (x + (2 * w) - (18 * len(right_op)), y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA) # Option 2


def put_text(frame, txt: str, x: int, y: int, w: int, h: int, color: tuple):
    '''Add the text in the Quiz Area'''

    t_x = (((x - w) + (x + (2 * w))) // 2)
    t_y = (((y + 20) + (y - h + 20)) // 2)
    cv2.putText(frame, txt, (t_x - (len(txt) * 9), t_y), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2, cv2.LINE_AA) # Puts question text


def add_quiz_area(frame, x: int, y: int, w: int, h: int, color_val: list, first_launch: bool, first_launch_time, d_ques: list, d_ans: list, ans: int, ans_corr: int, direction: int, question_asked: bool, left_op: str, right_op: str, correct_op_direction: int, broadcast_corr: bool, broadcast_incorr: bool, last: bool, end: bool):
    '''Add quiz box near the forehead'''

    get_dir = False

    if not broadcast_corr and not broadcast_incorr: cv2.rectangle(frame, (x - w, y + 20), (x + (2 * w), y - h + 20), color_val[0], -1) # Question
    elif broadcast_corr:
        cv2.rectangle(frame, (x - w, y + 20), (x + (2 * w), y - h + 20), color_val[2], -1)
        broadcast_corr = False
    elif broadcast_incorr:
        cv2.rectangle(frame, (x - w, y + 20), (x + (2 * w), y - h + 20), color_val[3], -1)
        broadcast_incorr = False

    cv2.rectangle(frame, (x - w, y + 20), (x + (2 * w), y - h + 20), (255, 255, 255)) # Border

    if (time.time() - first_launch_time) > 5:
        first_launch = False

    if first_launch:
        put_text(frame, 'Welcome to Maths Quiz!', x, y, w, h, color_val[1])
    elif end:
        put_text(frame, f'Correct answers: {ans_corr}', x, y, w, h, color_val[1])
    else:
        if last:
            curr_ques = d_ques[ans-1]
            curr_ans = d_ans[ans-1]
        else:
            curr_ques = d_ques[ans]
            curr_ans = d_ans[ans]

        get_dir = True

        if not question_asked and not last:
            question_asked = True

            # -1  -> Left
            #  1  -> Right
            correct_op_direction = random.sample([-1, 1], k=1)[0]

            if correct_op_direction == -1: left_op, right_op = generate_options(curr_ques, curr_ans)
            else: right_op, left_op = generate_options(curr_ques, curr_ans)

        put_text(frame, curr_ques, x, y, w, h, color_val[1]) # Adds question
        put_text_options(frame, x, y, w, direction, left_op, right_op) # Adds options

    return first_launch, ans, ans_corr, get_dir, left_op, right_op, question_asked, correct_op_direction, broadcast_corr, broadcast_incorr


def main():
    '''The main function for starting the program'''

    # Default Variables
    vid = cv2.VideoCapture(0) # Uses webcam
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    vid_cod = cv2.VideoWriter_fourcc(*'mp4v') # .mp4

    hide = False
    get_dir = False
    question_asked = False
    broadcast_corr = False
    broadcast_incorr = False
    last = False
    end = False

    first_launch = True

    first_launch_time = time.time()
    curr = time.time()

    check_dir() # Checks for the directory
    i = get_num()
    fps = 20
    save_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Captured Video\\', f'record_{i+1}.mp4')
    output = cv2.VideoWriter(save_path, vid_cod, fps, (640,480))

    face_cascade_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Data\\', 'frontface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    dict_eq = generate() # Generates the equation dictionary
    d_ques = list(dict_eq.keys())
    d_ans = list(dict_eq.values())

    left_op, right_op = '', ''

    x, y, w, h = 0, 0, 0, 0
    prev_left, prev_right = 0, 0
    ans, ans_corr = 0, 0
    direction = 0
    debounce = 0
    end_time = 0
    correct_op_direction = 0

    while True:
        # Checks if the questions are over
        if ans > (len(dict_eq) - 1):
            if not last: last = True
            else:
                end = True
                #_, _, _, _, _, _, _, _, _, _ = add_quiz_area(frame, x, y, w, h, color_val, first_launch, first_launch_time, d_ques, d_ans, ans, ans_corr, direction, question_asked, left_op, right_op, correct_op_direction, broadcast_corr, broadcast_incorr, last, end)
                if not end_time:
                    end_time = time.time()
                if time.time() - end_time > 5:
                    break

        # Opens the webcam
        _, frame = vid.read()
        frame = cv2.flip(frame, 1) # Flips the camera

        # Face Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Draws rectangle
        prev_left, prev_right = x, x + w

        color = (0, 0, 0) # Background color
        text_color = (255, 255, 255) # Text color
        correct_color = (0, 255, 0) # Correct background color
        incorrect_color = (0, 0, 255) # Incorrect background color

        color_val = [color, text_color, correct_color, incorrect_color]

        if type(face) == tuple:
            if not hide:
                first_launch, ans, ans_corr, get_dir, left_op, right_op, question_asked, correct_op_direction, broadcast_corr, broadcast_incorr = add_quiz_area(frame, x, y, w, h, color_val, first_launch, first_launch_time, d_ques, d_ans, ans, ans_corr, direction, question_asked, left_op, right_op, correct_op_direction, broadcast_corr, broadcast_incorr, last, end)

                prev = time.time()
                if (prev - curr) > 1:
                    hide = True # Hides the quiz area
        else:
            if hide: hide = False
            else:
                (x, y, w, h) = face[0]
                first_launch, ans, ans_corr, get_dir, left_op, right_op, question_asked, correct_op_direction, broadcast_corr, broadcast_incorr = add_quiz_area(frame, x, y, w, h, color_val, first_launch, first_launch_time, d_ques, d_ans, ans, ans_corr, direction, question_asked, left_op, right_op, correct_op_direction, broadcast_corr, broadcast_incorr, last, end)

                if get_dir:
                    #  0  -> Straight
                    #  1  -> Right
                    # -1  -> Left
                    direction = get_direction(prev_left, prev_right, x, w) # Gets alignment
                    get_dir = False
                
                if direction and question_asked:
                    if not debounce: 
                        debounce = time.time()
                    if time.time() - debounce > 1:
                        debounce = 0
                        
                        if direction == correct_op_direction:
                            ans_corr += 1
                            broadcast_corr = True
                        else: broadcast_incorr = True

                        direction = 0
                        ans += 1
                        question_asked = False

                curr = time.time() # Gets and updates the time

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


if __name__ == '__main__':
    main()