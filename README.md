# hand_posecontrol
通过手势与动作识别实现cyberdog的控制

## 依赖下载
首先需要下载虚拟平台，链接如下：https://github.com/MiRoboticsLab/cyberdog_sim  
此外，需要额外下载mediapipe：（此处为清华镜像源）  
`pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple`  

## 文件夹位置
建议将该文件夹与cyberdog_sim放在同一文件夹下  

## 运行方式
1.打开虚拟机摄像头    
2.在cyberdog_sim下运行虚拟平台：  
`python3 src/cyberdog_simulator/cyberdog_gazebo/script/launchsim.py`  
3.在hand_posecontrol中运行posecontrol_test.py：  
`python3 posecontrol_test.py`
