from flask import Blueprint, render_template, request, flash, redirect, url_for
from random import randint
from flask import Flask, render_template, Response
import requests
import os
import tensorflow as tf
import cv2
import time
import argparse
import posenet


home = Blueprint('home', __name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = './../static/'

region = "westus" #For example, "westus"
api_key = "6c8a89a3dbc142e592ae18b97061ae27"



@home.route('/home')
@home.route('/', methods=['GET'])
def homepage():
    return render_template('home.html', title='Home')


@home.route('/about')
def about():
    return render_template('about.html', title='About')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
# @app.route('/')
# def upload_form():
#     return render_template('upload.html')


@home.route('/', methods=['POST'])
def upload_image():
    # if 'file' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)
    # file = request.files['file']
    # print(file.filename)
    # if file.filename == '':
    #     print(file.filename)
    #     flash('No image selected for uploading')
    #     return redirect(request.url)
    # if file and allowed_file(file.filename):
    #     filename = file.filename
    #     dirname = os.path.dirname(__file__)
    #     filename1 = os.path.join(dirname, '../static/')
    #     file.save(os.path.join(filename1, filename))
    #     print(os.path.join(filename1, filename))
    #     #print('upload_image filename: ' + filename)
    #     flash('Image successfully uploaded and displayed below')
    #     with open(os.path.join(filename1, filename), 'rb') as f:
    #         data = f.read()
        
    #     prediction,confidence = getTBPrediction(data)
    return render_template('video.html', flag=1)
    # else:
    #     flash('Allowed image types are -> png, jpg, jpeg, gif')
    #     return redirect(request.url)        

# parser = argparse.ArgumentParser()
# parser.add_argument('--tolerance', type=int, default=30, help='The tolerance for the model in integers')
# parser.add_argument('--model', type=int, default=101, help='The model to use, available versions are 101 (def.), 102, 103 etc')
# parser.add_argument('--cam_id', type=int, default=0, help='The respective cam id to use (default 0)')
# parser.add_argument('--cam_width', type=int, default=1280, help='The width of the webcam in pixels (def. 1280)')
# parser.add_argument('--cam_height', type=int, default=720, help='The height of the webcam in pixels (def. 780)')
# parser.add_argument('--scale_factor', type=float, default=0.4, help='The scale factor to use (default: .7125)')
# parser.add_argument('--file', type=str, default=None, help="Use the video file at specified path instead of live cam")
# args = parser.parse_args()

keyValues = ['Nose', 'Left eye', 'Right eye', 'Left ear', 'Right ear', 'Left shoulder',
             'Right shoulder', 'Left elbow', 'Right elbow', 'Left wrist', 'Right wrist',
             'Left hip', 'Right hip', 'Left knee', 'Right knee', 'Left ankle', 'Right ankle']
tolerance = 30
tf.compat.v1.disable_v2_behavior()

def countRepetition(previous_pose, current_pose, previous_state, flag):
    if current_pose[0][10][0] == 0 and current_pose[0][10][1] == 0:
        return 'Cannot detect any joint in the frame', previous_pose, previous_state, flag
    else:
        string, current_state = '', previous_state.copy()
        sdy,sdx = 0,0
        # Discard first 5 (0-4 indices) values, we don't need the value of nose, eye etc
        for i in range(5, 17):
            # The fancy text overlay
            string = string + keyValues[i] + ': '
            string = string + str('%.2f' % (current_pose[0][i][0])) + ' ' + str('%.2f' % current_pose[0][i][1]) + '\n'
            # If the difference is greater or lesser than tolerance or -tolerance sum it up or add 0 to sdx and sdy
            dx = (current_pose[0][i][0] - previous_pose[0][i][0])
            dy = (current_pose[0][i][1] - previous_pose[0][i][1])
            if(dx < tolerance and dx > (-1 * tolerance)):
                dx = 0
            if(dy < tolerance and dy > (-1 * tolerance)):
                dy = 0
            sdx += dx
            sdy += dy
        # if an overall average increase in value is detected set the current_state's bit to 1, if it decrease set it to 0
        # if it is between tolerance*3 and -tolerance*3, do nothing (then current_state will contain same value as previous)
        if(sdx > (tolerance*3)):
            current_state[0] = 1
        elif(sdx < (tolerance*-3)):
            current_state[0] = 0
        if(sdy > (tolerance*3)):
            current_state[1] = 1
        elif(sdy < (tolerance*-3)):
            current_state[1] = 0
        if(current_state != previous_state):
            flag = (flag + 1)%2
        return string, current_pose, current_state.copy(), flag


# if args.file is not None: # Frame source, speicifed file or the specified(or default) live cam
#     cap = cv2.VideoCapture(args.file)
# else:
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)


def getFrame():
    with tf.compat.v1.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride']
        previous_pose = '' # '' denotes it is empty, really fast checking!
        count = 0 # Stores the count of repetitions
        # A flag denoting change in state. 0 -> previous state is continuing, 1 -> state has changed
        flag = -1
        # Novel string stores a pair of bits for each of the 12 joints denoting whether the joint is moving up or down
        # when plotted in a graph against time, 1 denotes upward and 0 denotes downward curving of the graph. It is initialised
        # as '22' so that current_state wont ever be equal to the string we generate unless there is no movement out of tolerance
        current_state = [2,2]
        while True:
            # Get a frame, and get the model's prediction
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=0.4, output_stride=output_stride)
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.4)
            # print(pose_scores, keypoint_scores, keypoint_coords)
            keypoint_coords *= output_scale # Normalising the output against the resolution

            if(isinstance(previous_pose, str)): # if previous_pose was not inialised, assign the current keypoints to it
                previous_pose = keypoint_coords
            
            text, previous_pose, current_state, flag = countRepetition(previous_pose, keypoint_coords, current_state, flag)

            if(flag == 1):
                count += 1
                flag = -1

            image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.4, min_part_score=0.1)

            # OpenCV does not recognise the use of \n delimeter
            y0, dy = 20, 20
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                image = cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255),1)

            image = cv2.putText(image, 'Count: ' + str(count), (10, y+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0),2)
            # cv2.imshow('RepCounter', image)

            ch = cv2.waitKey(1)
            if(ch == ord('q') or ch == ord('Q')):
                break # Exit the loop on press of q or Q
            elif(ch == ord('r') or ch == ord('R')):
                count = 0

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
        # cap.release()
        # cv2.destroyAllWindows()

def gen_frame1(cap1):  # generate frame by frame from camera
    success,image = cap1.read()
    count = 0
    success = True
    while (success):
        # Capture frame-by-frame
        success, frame = cap1.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@home.route('/video_feed')
def video_feed():
    return Response(getFrame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@home.route('/video_feed1/')
def video_feed1():
    dirname = os.path.dirname(__file__)
    filename1 = os.path.join(dirname, '../static/abc.mp4')
    cap1 = cv2.VideoCapture(filename1)
    return Response(gen_frame1(cap1), mimetype='multipart/x-mixed-replace; boundary=frame')
