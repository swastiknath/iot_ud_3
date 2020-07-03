# Cursor Pointer Controller
### Swastik Nath.
Made With Intel(R) Distribution of OpenVINO Toolkit 2020.R3 LTS.

![](https://github.com/swastiknath/iot_ud_3/raw/master/docs/output.gif)

The Cursor Pointer Controller leverages multiple Computer Vision Models and delivers the estimation of eye-gaze vectors from head angles and eye images from the detected faces in provided frames of video. In this project, we have used the Intel (R) OpenVINO Toolkit 2020.R3 to perform edge inference over pre-recorded video files or video streams over WebCam. Several Intel (R) OpenVINO Pretrained Models connected to form a cascade of a detection-estimation workflow.  

### Words about the Models in Work:

 - #### Face Detection Model (face-detection-adas-binary-0001) [PreTrained Model URL](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
 This model uses a deep neural network with MobileNet architecture implemented with **depth-wise** convolution layers to reduce the amount of computation for the 3x3 convolution block. There are also some 1x1 Convolution layers which are binary which can be implemented using effective binary XNOR + POPCOUNT approach. The model takes image as input with name `input` and size of 1x3x384x672 with expected color order as BGR and outputs the confidence, and the cordinates of the detected faces.  

At the onset, a face detection model object is fed the original video frames to infer the coordinates of the detected face above a define confidence threshold. 

 - #### Head Pose Estimation Model (head-pose-estimation-adas-0001) [PreTrained Model URL](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
 Head Pose estimation model is based on Convolution Neural Networks and the angular parameters are estimated using Regression approach using the following layout:
   `CNN -> RELU -> Batch Normalization -> Fully Connected Layers`
 This model takes image as input with name `data` with shape of 1x3x60x60 in BGR color channel. It outputs three vectors with size [1, 1] for Angular Yaw, Angular Pitch and Angular Roll in degrees. 
 
In this project we feed this model the cropped face from the detected faces after resizing it. 

 - #### Facial Landmarks Detection Model (facial-landmarks-35-adas-0002) [PreTrained Model URL](https://docs.openvinotoolkit.org/latest/_models_intel_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html)
  Facial Landmarks Detection Model uses a custom-architecture neural network which approximates a total of 35 key landmarks in a given detected face.
  The `p1` and `p2` gives us the centre point of the left and right eye respectively and we use the co-ordinates of these two landmarks to draw out the left and right eye image from the frame. We also use the landmark `p4` which represents the nose-tip to draw the head pose angles. 
  We feed in the image of the detected face with name `data` with shape of 1x3x60x60 and which in turn results in 35 pair of x, y normed co-ordinates for the facial landmarks. 

 - #### Gaze Estimation Model (gaze-estimation-adas-0002) [PreTrained Model URL](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

 The Model uses a custom VGG-like Convolutional Network in order to estimate the gaze direction estimation. The network takes in three inputs, a square crop of the left eye image, square crop image of the right eye_image and a vector of 3 head pose angles. The network outputs a Three dimensional vector in the Cartesian Coordinate system.   


## Project Set Up and Installation

<hr></hr>

We need to make sure that everything is correctly initialized before starting up the program. 
#### Directory Strucure: 
 - bin : In this directory we save in the pre-recorded video like for this project the provided video. We also save in the file with .avi format if the format of the video is not compatible.
 - charts : We save in the Output Videos in this directory. 
 - docs : Documentation about how the things in this project work.
 - models :  We save in the pre-trained IR model in this directory after downloading them with `model_downloader.py`.
 - src :  Source Python Files for several model objects and the inference code.
 #### Initializing the Environment[The following are for Windows only]
  - Extract the .zip package of this package and Open the Terminal in the extracted folder. 
  - Create a Virtual Environment(We assume to create iot_ud_3 here):
  ```python
  pip install virtualenv
  virtualenv iot_ud_3
  iot_ud_3/Source/activate.bat
  ```
  - Installing Project Dependencies:
  Make sure you are in the project directory and have successfully initiated your virtual environment and make sure you have administrator privillages.  
  ```
  pip install -r requirements.txt
  ```
  - Initialize the OpenVino Environment :
  ```python
  <OPENVINO_INSTALL_DIR>/bin/setupvars.sh
  python src/env_test.py
  ```
  If everything is working properly we should see a success message. 

 #### Downloading and Saving the models: 
 We need to issue the following commands to download and save the PreTrained Models in the IR format in the `models` folder. 
 ```
 mkdir models 

 python <OPENVINO_INSTALL_DIR>/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-001 --output_dir models/

 python <OPENVINO_INSTALL_DIR>/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --output_dir models/

 python <OPENVINO_INSTALL_DIR>/deployment_tools/tools/model_downloader/downloader.py --name facial-landmarks-35-adas-0002 --output_dir models/

 python <OPENVINO_INSTALL_DIR>/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --output_dir models/

 ```


## Demonstration:

<hr></hr>

In order to get the program up and running, you can follow the following examples to get started. Make sure you have completed all the steps in the `Project Setup and Installation` before attempting the following command.

- ### Running Inference on Pre-recorded Videos with Intermediate Visuals ON:  

```
python src\cursorcontroller.py -f "models\intel\face-detection-retail-0005\FP32\face-detection-retail-0005.xml" -p "models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml" -l "models\intel\facial-landmarks-35-adas-0002\FP32\facial-landmarks-35-adas-0002.xml" -g "models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml" -i "bin\demo.mp4" -v True -o "charts\"
```  

- ### Running Inference on WebCam Streaming with Intermediate Visuals ON:

```
python src\cursorcontroller.py -f "models\intel\face-detection-retail-0005\FP32\face-detection-retail-0005.xml" -p "models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml" -l "models\intel\facial-landmarks-35-adas-0002\FP32\facial-landmarks-35-adas-0002.xml" -g "models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml" -i "CAM" -v True

```

## Documentation



 The following is the list of command-line arguments which can be configured upto your choice. 
 
 
| Short Flag | Long Flag | What it does | Default Value |
|------------|-----------|--------------|---------------|
| -h   | --help   |  Show help message and exit|    - |
| -f  | --face_detection_model | PATH TO .XML FILE OF FACE DETECTION MODEL| -|
|-p | --head_pose_model  |PATH TO .XML FILE OF HEAD POSE DETECTION MODEL | -|
|-l | --facial_landmark_model|PATH TO .XML FILE OF FACIAL DETECTION MODEL| -|
|-g | --gaze_estimator_model|PATH TO .XML FILE OF GAZE ESTIMATOR MODEL|- |
|-i | --input |PATH TO THE VIDEO/IMAGE FILE, USE ['CAM'] TO USE YOUR WEBCAM AS STREAM INPUT.| - |
|-e| --cpu_extension |PATH TO MATH KERNEL LIBRARY FILE FOR CUSTOM LAYER IMPLEMENTATION| None|
|-d | --device |INFERENCE TARGET, SELECT FROM [CPU, GPU, MYRIAD, FPGA]| CPU |
|-o |--output_dir|PATH TO THE OUTPUT DIRECTORY FOR INFERENCE STATS.| False |
|-v | --intermediate_visuals |FLAG TO VISUALISE THE INTERMEDIATE RESULTS, should be one of True / False| False |
|-m | --print_layer_metrics |FLAG TO PRINT LAYER WISE METRICS| False |
|-t | --threshold |CONFIDENCE THRESHOLD FOR INFERENCE.| 0.5 |
|-q | --mirror_mode |Flag to Mitigate the Mirror Mode to Normal Mode, Horizontally Flips the Image| True |
|-x | --mouse_precision|Mouse Pointer Movement Precision, must be one of [high, medium, or low]| Medium |
|-s | --speed |Speed of the Mouse Pointer Movement, must be one of[fast, medium, slow]|Medium |


## Benchmarks

In order to benchmark the application across different hardwares we use the *Intel(R) DevCloud* and multiple precisions of the pre-trained model IR files. We take into account only two device scenario in this case, one is CPU and another one is IGPU becasue the application will mostly run of these types of hardware if deployed. 

 - ### Model 1 : Face Detection 
    - FP16/INT8 precisions are not available for `face-detection-adas-binary-0001` pre-trained model. 

    - GFlops: 0.611
    
    
|Benchmark | Device | FP32 (seconds) | FP16 (seconds)| INT8 (seconds) |
 |-|--------|-----|------|------|
 |Inference Time: | CPU |   0.0241  |  N/A  | N/A  |
 |Loading Time:   | CPU | 0.4748   | N/A  | N/A  |
 |Pre-Processing Time | CPU   | 0.0035   | N/A |N/A |
 |Post-Processing Time | CPU| 0.0010 |N/A | N/A |
 |Inference Time: | IGPU |   0.00143  |  N/A  | N/A  |
 |Loading Time:   | IGPU | 0.7023   | N/A  | N/A  |
 |Pre-Processing Time | IGPU   | 0.0032   | N/A |N/A |
 |Post-Processing Time | IGPU| 0.0043 |N/A | N/A |  
  

- ### Model 2 : Head Pose Estimation :
     - GFlops: 0.105


|Benchmark | Device | FP32 (seconds) | FP16 (seconds)| INT8 (seconds) |
 |-|--------|-----|------|------|
 |Inference Time: | CPU |   0.0056  |  0.0025  | 0.0025  |
 |Loading Time:   | CPU | 0.1933   | 0.2257  | 0.2882  |
 |Pre-Processing Time | CPU   | 0   | 0 |0 |
 |Post-Processing Time | CPU| 0 |0 | 0 |
 |Inference Time: | IGPU |   0.0043  |  0.0018  | 0.0013  |
 |Loading Time:   | IGPU | 0.2429   | 0.2927  | 0.3287  |
 |Pre-Processing Time | IGPU   | 0   | 0 |0 |
 |Post-Processing Time | IGPU| 0 |0 | 0 |  
  

- ### Model 3 : Facial Landmark :
     - GFlops : 0.042


|Benchmark | Device | FP32 (seconds) | FP16 (seconds)| INT8 (seconds) |
 |-|--------|-----|------|------|
 |Inference Time: | CPU |   0.0067  |  0.0075  | 0.0070  |
 |Loading Time:   | CPU | 0.7737    | 0.9425  | 2.8837  |
 |Pre-Processing Time | CPU   | 0   | 0 |0 |
 |Post-Processing Time | CPU| 0 |0.010 | 0.0010 |
 |Inference Time: | IGPU |   0.0043  |  0.0059  | 0.0047  |
 |Loading Time:   | IGPU | 1.2312   | 1.5412  | 3.1223  |
 |Pre-Processing Time | IGPU   | 0   | 0 |0 |
 |Post-Processing Time | IGPU| 0.0010 |0 | 0 | 


- ### Model 4: Gaze Estimation :
    - GFlops: 0.139
    
|Benchmark | Device | FP32 (seconds) | FP16 (seconds)| INT8 (seconds) |
 |-|--------|-----|------|------|
 |Inference Time: | CPU |   0.0057  |  0.0050  | 0.0054  |
 |Loading Time:   | CPU | 0.2597  | 0.3556  | 0.2512 |
 |Pre-Processing Time | CPU   | 0   | 0 |0 |
 |Post-Processing Time | CPU| 0 |0 | 0 |
 |Inference Time: | IGPU |   0.0043  |  0.0038  | 0.0041  |
 |Loading Time:   | IGPU | 0 .2798  | 0.3876  | 0.3004  |
 |Pre-Processing Time | IGPU   | 0   | 0 |0 |
 |Post-Processing Time | IGPU| 0.0010 |0 | 0 |  
  

## Results

From the observations above we get to see that the model inference time is slightly lower in case of IGPU, but the model loading time is slightly larger than that of CPU in case of IGPU. 

In case of FP32, FP16 and INT8 models both in case of CPU and IGPU the models load slightly faster from FP32 to INT8, and the models perfroms inference slightly faster from FP32 to INT8, as because with less precision there left less complexities to be taken care of. That's why a model with INT8 precision perfroms faster than that of the FP32 and that is why there is less data that needs to be accessed in case of 8bit Integet than the 32bit Floating points. 

## Stand Out Suggestions
Keeping in mind the stand out suggestions, I have implemented the following points laid out. 
 - Benchmarking the Running Times of Different Layers: I have used the `get_perf_counts()` API to print out the layerwise execution time which the user can enable through the CLI using `-m` or `--print_layer_metrics` argument set to `True`. 
 
 - Toggle to Enable/Disable the Intermediate Visualisations: I have implemented a CLI argument `-v` or `--intermediate_visuals` set to `True` to enable or disable the video feed. If the said argument is true it will only then actually go ahead and calculate those visualisations to display in order to increase speed and performance. As in some of the machines, `pyautogui` package really slows down the application. Closing in the Video Frame if not required will significantly decrease the complexities needed to calculate gaze vector and camera matrix calculations and eventually will result in less wastage of computational power. 

 - Inference Pipeline for both Video File as Webcam feed as Input: I have implemented a solution to aceept both Video Files and Webcam Feed as input. In case of Webcame we just need to use the `--input "CAM"` CLI argument and for video files we can use `--input <FILENAME>`. 

 - Mitigating Horizontal Flip on Some Camera Modules: Some WebCam Modules horizontally flip the camera feed, but horizontally flipping the input image will result in inaccurate estimations by the models. So we can use `-q` or `--mirror_mode` set to `True` to strainght up the flipped image. 

### Async Inference
I have implemented only Asynchronous Inference in each of the model, the Asynchronous inference actually comes into help if the thread is multi-threaded, when instead of blocking all the threads it can perform asynchronously. That is why power-usage and efficiency will be enhanced for sure. 

### Edge Cases

 -  In case of multiple people in the frame the gaze vector will not be correctly sensed as because the two people might be staring at opposite directions of each other, thereby leading to a great confusion. So, I have decided to pause the inference and issue a Warning to the User to avoid inappropriate gaze prediction and confusion for the model whenever there is more than 1 person is present in the frame. 


