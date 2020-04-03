import cv2
from PIL import Image
import numpy as np

from subprocess import Popen, PIPE
from enum import IntEnum, auto
import sys, math, os, time, argparse
import threading
import queue

from keras.models import load_model
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), 'UGATIT'))
from UGATIT import UGATIT


'''
depress warning
'''
#import logging, warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=Warning)
#tf.get_logger().setLevel('INFO')
#tf.autograph.set_verbosity(0)
#tf.get_logger().setLevel(logging.ERROR)

'''
Command line arguments
'''
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='specify input and output device')
parser.add_argument('--input_video_num', type=int, required=True,
                    help='input video device number. ex) if input is /dev/video0 then the value is 0')
parser.add_argument('--output_video_dev', type=str, required=True,
                    help='input video device. ex) /dev/video2')
parser.add_argument('--emotion_mode', type=str2bool, required=False, default=False,
                    help='enable emotion mode')
parser.add_argument('--anime_mode', type=str2bool, required=False, default=False,
                    help='enable anime mode')

'''
args for anime mode 
'''
parser.add_argument('--phase', type=str, default='test', help='[train / test]')
parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset_name')

parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')

parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
parser.add_argument('--GP_ld', type=int, default=10, help='The gradient penalty lambda')
parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
parser.add_argument('--cycle_weight', type=int, default=10, help='Weight about Cycle')
parser.add_argument('--identity_weight', type=int, default=10, help='Weight about Identity')
parser.add_argument('--cam_weight', type=int, default=1000, help='Weight about CAM')
parser.add_argument('--gan_type', type=str, default='lsgan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]')

parser.add_argument('--smoothing', type=str2bool, default=True, help='AdaLIN smoothing effect')

parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

parser.add_argument('--img_size', type=int, default=256, help='The size of image')
parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                help='Directory name to save the checkpoints')
parser.add_argument('--result_dir', type=str, default='results',
                    help='Directory name to save the generated images')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='Directory name to save training logs')
parser.add_argument('--sample_dir', type=str, default='samples',
                    help='Directory name to save the samples on training')


args       = parser.parse_args()
BATCH_SIZE = args.batch_size

'''
Queue for anime mode
'''
anime_mode_input_queue  = queue.Queue()
anime_mode_output_queue = queue.Queue()
anime_buffer_image      = None
anime_frame_num         = 0
anime_fps_start         = time.time()
anime_fps               = 0
'''
Mode definition
'''
class modes(IntEnum):
    SIMPLE_SMILE_MODE = auto()
    EMOTION_MODE      = auto()
    ANIME_MODE        = auto()
'''
Classifiers
'''
face_classifier_classifier = None
anime_session              = None
anime_model                = None

'''
Path for resources
'''
face_cascade_path  = './models/haarcascade_frontalface_default.xml'



def anime_mode_worker():
    frames = []                

    while True:
        item_num = anime_mode_input_queue.qsize()
        #print(item_num)
        for i in range(item_num):
            frame = anime_mode_input_queue.get()
            frame = cv2.resize(frame, dsize=(256, 256))
            frames.append(frame)
            #print(f'{i}/{item_num}')
        if len(frames) < BATCH_SIZE:
            if item_num == 0:
                pass
                #time.sleep(1)
                
            continue
        frames = np.array(frames)
        #print(sys.stderr, frames.shape)
        
        new_frames = anime_model.predict(frames[-1 * BATCH_SIZE:])

        for i, (old_frame, new_frame) in enumerate(zip(frames[-1 * BATCH_SIZE:], new_frames)):
            anime_mode_output_queue.put( (old_frame, new_frame))
        frames = []


def load_resources(mode):
    global face_classifier_classifier
    face_classifier_classifier = cv2.CascadeClassifier(face_cascade_path)

    if mode == modes.ANIME_MODE:
        global anime_session, anime_model
        
        anime_session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        anime_model   = UGATIT(anime_session, args)
        anime_model.build_model()
        anime_model.load_model(anime_session)
        
        
        

def paste(img, imgback, x, y, angle, scale):
    if img.shape [0] > imgback.shape[0] or img.shape[1] > imgback.shape[1]:
        h_ratio = imgback.shape[0] / img.shape[0]
        w_ratio = imgback.shape[1] / img.shape[1]
        if h_ratio < w_ratio:
            new_h = int(img.shape[0] * h_ratio)
            new_w = int(img.shape[1] * h_ratio)
        else:
            new_h = int(img.shape[0] * w_ratio)
            new_w = int(img.shape[1] * w_ratio)
        if new_h % 2 != 0:
            new_h += 1
        if new_w % 2 != 0:
            new_w += 1
            
        img = cv2.resize(img, (new_w, new_h))
        #print(sys.stderr, f'pate resize img : {new_h}, {new_w}')    
    r   = img.shape[0]
    c   = img.shape[1]
    rb  = imgback.shape[0]
    cb  = imgback.shape[1]    
    hrb = round(rb/2)
    hcb = round(cb/2)
    hr  = round(r/2)
    hc  = round(c/2)

    #print(sys.stderr, f'(2) -> {r}, {c},    {rb},{cb}')    

    
    # Copy the forward image and move to the center of the background image
    imgrot = np.zeros((rb,cb,3),np.uint8)
    imgrot[hrb-hr:hrb+hr,hcb-hc:hcb+hc,:] = img[:hr*2,:hc*2,:]

    # Rotation and scaling
    M = cv2.getRotationMatrix2D((hcb,hrb),angle,scale)
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))
    # Translation
    M = np.float32([[1,0,x],[0,1,y]])
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))

    # Makeing mask
    imggray = cv2.cvtColor(imgrot,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imggray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of the forward image in the background image
    img1_bg = cv2.bitwise_and(imgback,imgback,mask = mask_inv)

    # Take only region of the forward image.
    img2_fg = cv2.bitwise_and(imgrot,imgrot,mask = mask)

    # Paste the forward image on the background image
    imgpaste = cv2.add(img1_bg,img2_fg)

    return imgpaste

def apply_offsets_for_anime_mode(face_location, offsets):
    x, y, width, height = face_location
    x_off, y_off = offsets # x_off is ignored here.

    ### At first Top and Bottom are determined.
    top    = y - y_off
    bottom = y + height + y_off
    if top < 0:
        top = 0

    ### determin x_off so as to make square.
    new_height = bottom - top
    x_off = int((new_height - width ) / 2)

    ### Then Left and Right are determined.
    left  = x - x_off
    right = x + width + x_off 
    if left < 0 :
        left = 0
        
    ### return
    return (x - x_off, x + width + x_off, top, bottom)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def edit_frame(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier_classifier.detectMultiScale(gray, 1.1, 5)
    
    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0),2)
        if mode == modes.ANIME_MODE:
            global anime_buffer_image, anime_frame_num, anime_fps_start, anime_fps

            ### new frame entry to process (raw frame)
            anime_offsets    = (60, 60)
            x1, x2, y1, y2    = apply_offsets_for_anime_mode((x,y,w,h), anime_offsets)
            anime_rgb        = frame[y1:y2, x1:x2]
            
                
            try:
                cv2.imwrite('tmp.png',anime_rgb)
                img = cv2.imread('tmp.png', flags=cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                
                anime_rgb = img
                anime_mode_input_queue.put(anime_rgb)
            except Exception as e:
                ### if exception occur put original frame
                anime_mode_input_queue.put(frame)


            ### show edited frame

            
            try:
                new_frame = anime_mode_output_queue.get(block=False)
                # to be shown frame(animated frame)
                (old_frame, new_frame) = new_frame
                old_frame = cv2.resize(old_frame, (50, 50))
                new_frame = paste(old_frame, new_frame, +80, -80, 0, 1.0)

                anime_frame_num += 1
                anime_fps_now    = time.time()
                if anime_fps_now - anime_fps_start > 5:
                    spend_time       = anime_fps_now - anime_fps_start
                    anime_fps        = round((anime_frame_num / spend_time),2)
                    anime_fps_start  = anime_fps_now
                    anime_frame_num  = 0
                    
                # for fps
                font_scale=0.5
                color = (200,200,200)
                thickness=1
                cv2.putText(new_frame, f'fps:{anime_fps}',
                            (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, thickness, cv2.LINE_AA
                )
                
                
                anime_buffer_image = new_frame

            except queue.Empty as e:
                if anime_buffer_image is None:
                    anime_buffer_image = np.zeros((256, 256, 3), np.uint8)
                pass

    ### If face is not detected, show previous frame or blank frame
    if mode == modes.ANIME_MODE:
        if anime_buffer_image is not None:
            frame = anime_buffer_image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = np.zeros((256, 256, 3), np.uint8)
    return frame


if __name__=="__main__":
    input  = args.input_video_num
    output = args.output_video_dev
    cap    = cv2.VideoCapture(input)
    if args.anime_mode == True:
        mode = modes.ANIME_MODE
    else:
        mode = modes.SIMPLE_SMILE_MODE

    print(f'start with mode: {mode}')
    load_resources(mode)

    print('web camera hook start!')
    p = Popen(['ffmpeg', '-y', '-i', '-', '-pix_fmt', 'yuyv422', '-f', 'v4l2', output], stdin=PIPE)
    
    try:
        if mode == modes.ANIME_MODE:
            t = threading.Thread(target=anime_mode_worker)
            t.start()
        while True:
            ret,im = cap.read()
            im     = edit_frame(im)
            im     = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im     = Image.fromarray(np.uint8(im))
            im.save(p.stdin, 'JPEG')
            
    except KeyboardInterrupt:
        pass


    anime_session.close()
    p.stdin.close()
    p.wait()

    print('web camera hook fin!')
    
