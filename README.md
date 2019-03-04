# Single-Shot Refinement Neural Network for Object Detection

## Abstract
This is a MxNet_gluon re-implementation of [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/pdf/1711.06897.pdf).     

This project is based on MxNet_GluonCV, and completed by [YangJirui](https://github.com/yangJirui).     

## Train on VOC 2007+2012 trainval and test on VOC 2007 test.   

## Comparison

### use voc2007_metric
**SSD 300(vgg-16): 77.2 mAP \
Paper's refineDet 320(vgg-16): 80.0 mAP \
Our refineDet 320(vgg-16) : 78.9 mAP**


**Our re-implementation of RefineDet performs about 1mAP lower than Paper's. \
We will try our best to fix it in the future.**    
 
## My Development Environment
1、python2.7 (anaconda recommend)             
2、cuda9.0 (cuda 8.0 may cause Nan when training)                    
3、[opencv(cv2)](https://pypi.org/project/opencv-python/)    
   
## Train
1、train
```  
cd $PATH_ROOT/scripts/detection/refineDet
./train_scipt.sh
```



       
