# UIGA-FER
The test code for the paper 《Unpaired Images based Generator Architecture for Facial Expression Recognition》

# #Pre-requisites
 (1) Python 3.6.7.
 
 (2) Scipy.
 
 (3) PyTorch (r1.0.1) .
 

 ##Datasets
 (1) You may use any dataset with labels of the expression. In our experiments, we use Multi-PIE (http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html) and RAF-DB (http://www.whdeng.cn/RAF/model1.html). 
 
 (2) It is better to detect the face before you train the model. In this paper, we use a lib face detection algorithm (https://github.com/ShiqiYu/libfacedetection)

Besides, please ensure that you have the following directory tree structure in your repo.
├── datasets
│   └── raf
│   ├────data
│       ├────images_test.list
│   └── multiple
│       ├──── data
│       ├──── images_test.list


##Pre-trained models

Please put all the pre-trained models into the folder "models".

####1.Facial expression synthesis
The pre-trained GAN model for facial expression synthesis on RAF-DB can be download from the following links:
Encoder model(https://pan.baidu.com/s/1iRTsx1_QgTju0JVxQoTx5w)
Decoder model(https://pan.baidu.com/s/1eN2fngmaMUP-2N6ts-_jBw)

####2.Facial expression recognition
The pre-trained VGG model for facial expression rocogition can be download from the following links:
#####On the Multi-PIE dataset:
FER(https://pan.baidu.com/s/1ZD_pZX8C4l_bGReflt-p7Q)
FER_enc(https://pan.baidu.com/s/1A1lU5Vafxml7cqgcAiWy7g)
#####On the RAF-DB:
FER(https://pan.baidu.com/s/1eiHCK8Bgik9B4pGNSJekYw)
FER_enc(https://pan.baidu.com/s/1fGWiriXTbM5wov5d2z_r7A)

##Testing
####1.Facial expression synthesis

To swap the expressions between two unpaired images, you can run the following code. Here `-a surprised fearful disgusted happy sad angry neutral` indicates the expression names. And `--swap_list 3 6` means the expression id of the input image and target image, respectively. The generated image is saved as `result.jpg`, which includes the original images and generated images with exchanged expressions.

Notes. Please do not change the order of expressions in `-a surprised fearful disgusted happy sad angry neutral`
```
$ python exp_synthesis.py -a surprised fearful disgusted happy sad angry neutral --swap_list 3 6 --input ./images/happy.jpg --target ./images/neutral.jpg --gpu 0
```

####2.Facial Expression Recognition

To evaluate the facial expression recognition model, you can run the following  code.
```
$ python FER.py --multipie --gpu 0 # run on the Multi-PIE dataset
$ python FER.py --raf --gpu 0  # run on the RAF-DB
```
Then, it is supposed to print out the following message.
****
Namespace(gpu='0', multiple=False, raf=True)
Load pre-trained model successfully!
CM end!
['disgust', 'scream', 'smile', 'squint', 'surprise', 'neutral']
The accuracy for each expression is :
......
The average accuracy is : ...
***

## Files
(1) 'exp_synthesis.py' consists of the class which builds and loads the facial expression synthesis model, and several functions that implements the process of generating facial expressions with unpaired images.

(2) 'FER.py' includes the class builds and loads the facial expression recognition model, and several functions to do the test.

(3) 'nets.py' defines the network structures of the GAN and Classifier in our model

(4) 'config_dataset.py' includes the class that loads data and hyper-parameters for the model.
