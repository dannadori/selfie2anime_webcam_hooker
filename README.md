# selfie2anime_webhooker
- old version: https://github.com/dannadori/WebCamHooker

<p align="center">
<img src="./doc/frame_test_gpu_2080ti.gif" width="800" />
</p>

## 1. selfie2anime_webhooker
selfie2anime_webhooker takes the input image from a physical webcam and modifies it then outputs it to a virtual webcam.
This vritual cam can be used for video chats such as Teams and Zoom.

This image is with CPU on teams.
<p align="center">
<img src="./doc/demo_teams_cpu.gif" width="800" />
</p>


You can use gpu, if you want to more fps.
This is comparisontable.
- top-left      : CPU Intel 4770
- top-right     : CPU Intel 9900KF
- bottom-left   : CPU Intel 9900KF with GPU 2080ti
- bottom-right  : CPU Intel 9900KF with GPU 2080ti throttling by skip_frame=3
<p align="center">
<img src="./doc/frame_test_4screen.gif" width="800" />
</p>



## 2. Prerequisite
It should work fine on most Linux systems, but the environment I worked in is a Debian Buster.
```
$ cat /etc/debian_version
10.3
```

Also, if you don't seem to have python3 on board, please introduce it.
```
$ python3 --version
Python 3.7.3
```

#### Install related software.
##### Virtual Webcam Device
This time, we will use a virtual webcam device called v4l2loopback.
https://github.com/umlaeute/v4l2loopback


We need to identify the virtual webcam device and the actual webcam, so we first check the device file of the actual webcam.
In the example below, it looks like video0 and video1 are assigned to the actual webcam.
```
$ ls /dev/video*.
/dev/video0 /dev/video1
```

So, let's introduce v4l2loopback.
First of all, please git clone, make and install.
```
$ git clone https://github.com/umlaeute/v4l2loopback.git
$ cd v4l2loopback
$ make
$ sudo make install
```
Next, load the module. In this case, it is necessary to add exclusive_caps=1 to make it recognized by chrome. [https://github.com/umlaeute/v4l2loopback/issues/78]
```
sudo modprobe v4l2loopback exclusive_caps=1
```
Now that the module is loaded, let's check the device file. In the example below, video2 has been added.
```
$ ls /dev/video*.
/dev/video0 /dev/video1 /dev/video2
```

##### ffmpeg
The easiest way to send data to a virtual webcam device is to use ffmpeg.
You can use apt-get and so on to introduce it quickly.




### setup
##### git clone
First, clone the following repository files to install the module.
```
$ git clone https://github.com/dannadori/selfie2anime_webhooker.git
$ cd selfie2anime_webhooker.git
$ pip3 install -r requirements.txt
```

##### get face detection cascade
Here you will get the cascade file for face detection, you can find out more about cascade file in opencv official.
https://github.com/opencv/opencv/tree/master/data/haarcascades
```
$ wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -P models
```

##### get pre-trained selfie2anime model
get a pre-trained model from [UGATIT](https://github.com/taki0112/UGATIT)

URL is [here](https://drive.google.com/file/d/19xQK2onIy-3S5W5K-XIh85pAg_RNvBVf/view?usp=sharing)
This file is very big zip. And we cannot unzip this zip by using standard unzip. please use 7zip on Windows or Mac.(more detail, see issues of UGATIT)

Here is the hash value (md5sum) of the model that runs normally. (Probably because this is the biggest stumbling block.)
```
$ find . -type f |xargs -I{} md5sum {}
43a47eb34ad056427457b1f8452e3f79 . /UGATIT.model-1000000.data-00000-of-00001
388e18fe2d6cedab8b1dbaefdddab4da . /UGATIT.model-1000000.meta
a0835353525ecf78c4b6b33b0b2ab2b75c . /UGATIT.model-1000000.index
f8c38782b22e3c4c61d4937316cd3493 . /checkpoint
```

### Let' start video chat
```
$ python3 webcamhooker.py --input_video_num 0 --output_video_dev /dev/video2 --anime_mode True
```
When you want to have a video chat, you should see something called "dummy~" in the list of video devices, so select it.
