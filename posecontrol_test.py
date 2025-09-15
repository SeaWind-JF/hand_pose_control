import lcm
import sys
import time
import threading
from threading import Thread, Lock
from robot_control_cmd_lcmt import robot_control_cmd_lcmt
from robot_control_response_lcmt import robot_control_response_lcmt
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
import cv2
import mediapipe as mp
import math

# 初始化 MediaPipe Hands 模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
                       
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

def find_camera_topic():
    """自动检测可用的相机话题"""
    try:
        # 创建临时节点来获取话题信息
        node = rclpy.create_node('temp_topic_detector')
        topics = node.get_topic_names_and_types()
        
        # 筛选图像话题
        image_topics = []
        for topic_name, topic_types in topics:
            if 'sensor_msgs/msg/Image' in topic_types:
                image_topics.append(topic_name)
        
        # 如果没有找到任何图像话题
        if not image_topics:
            print("未找到任何图像话题！")
            return None
        
        # 打印所有找到的图像话题
        print("找到的图像话题:")
        for topic in image_topics:
            print(f" - {topic}")
        
        # 优先选择包含"rgb"或"color"的话题
        for topic in image_topics:
            if 'rgb' in topic or 'color' in topic:
                print(f"选择话题: {topic}")
                return topic
        
        # 返回第一个图像话题（如果没有匹配的）
        print(f"没有找到包含'rgb'或'color'的图像话题，使用第一个图像话题: {image_topics[0]}")
        return image_topics[0]
    
    except Exception as e:
        print(f"查找相机话题时出错: {str(e)}")
        return None
    finally:
        # 确保销毁临时节点
        if 'node' in locals():
            node.destroy_node()

class Robot_Ctrl(object):
    def __init__(self):
        self.rec_thread = Thread(target=self.rec_responce)
        self.send_thread = Thread(target=self.send_publish)
        self.lc_r = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")
        self.lc_s = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd_msg = robot_control_cmd_lcmt()
        self.rec_msg = robot_control_response_lcmt()
        self.send_lock = Lock()
        self.delay_cnt = 0
        self.mode_ok = 0
        self.gait_ok = 0
        self.runing = 1

    def run(self):
        self.lc_r.subscribe("robot_control_response", self.msg_handler)
        self.send_thread.start()
        self.rec_thread.start()

    def msg_handler(self, channel, data):
        self.rec_msg = robot_control_response_lcmt().decode(data)
        if (self.rec_msg.order_process_bar >= 95):
            self.mode_ok = self.rec_msg.mode
        else:
            self.mode_ok = 0

    def rec_responce(self):
        while self.runing:
            self.lc_r.handle()
            time.sleep(0.002)

    def Wait_finish(self, mode, gait_id, timeout=5.0):
        count = 0
        max_count = int(timeout / 0.005)
        while self.runing and count < max_count:
            if self.mode_ok == mode and self.gait_ok == gait_id:
                return True
            else:
                time.sleep(0.005)
                count += 1
        return False

    def send_publish(self):
        while self.runing:
            self.send_lock.acquire()
            if self.delay_cnt > 20:  # Heartbeat signal 10HZ
                self.lc_s.publish("robot_control_cmd", self.cmd_msg.encode())
                self.delay_cnt = 0
            self.delay_cnt += 1
            self.send_lock.release()
            time.sleep(0.005)

    def Send_cmd(self, msg):
        self.send_lock.acquire()
        self.delay_cnt = 50
        self.cmd_msg = msg
        self.send_lock.release()

    def quit(self):
        self.runing = 0
        self.rec_thread.join()
        self.send_thread.join()

class GestureControl:
    """手势识别控制类"""
    def __init__(self):
        self.current_gesture = "Unknown"
        self.gesture_lock = Lock()
        self.gesture_start_time = {}
        self.min_hold_time = 2.0
        
    def calculate_distance(self, point1, point2):
        """计算两个点之间的欧氏距离"""
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
    
    def count_fingers_and_get_status(self, hand_landmarks, hand_label):
        """计算手指伸直状态"""
        count = 0
        status = {}
        landmarks = hand_landmarks.landmark

        # 手指判断
        fingers = {
            "index": (8, 6),
            "middle": (12, 10),
            "ring": (16, 14),
            "pinky": (20, 18)
        }
        for finger, (tip_id, pip_id) in fingers.items():
            if landmarks[tip_id].y < landmarks[pip_id].y:
                status[finger] = True
                count += 1
            else:
                status[finger] = False

        # 拇指判断
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        if hand_label == "Right":
            status["thumb"] = thumb_tip.x > thumb_ip.x
        else:
            status["thumb"] = thumb_tip.x < thumb_ip.x
        if status["thumb"]:
            count += 1

        return count, status
    
    def detect_gesture(self, hand_landmarks, hand_label, status):
        """手势检测"""
        landmarks = hand_landmarks.landmark
        gesture = "Unknown"

        thumb = status["thumb"]
        index = status["index"]
        middle = status["middle"]
        ring = status["ring"]
        pinky = status["pinky"]

        # 基础手势检测
        if thumb and not any([index, middle, ring, pinky]):
            gesture = "Jump"  
        elif all([thumb, index, middle, ring, pinky]):
            gesture = "Stand"   
        elif not any([thumb, index, middle, ring, pinky]):
            gesture = "Slow_walk"   
        elif index and not any([thumb, middle, ring, pinky]):
            gesture = "Left"    
        elif pinky and not any([thumb, index, middle, ring]):
            gesture = "Right"  
        elif index and middle and not any([thumb, ring, pinky]):
            gesture = "Hold_left_hand"
        elif ring and pinky and not any([thumb, index, middle]):
            gesture = "Hold_right_hand"
        elif all([index, middle, ring, pinky]) and not thumb:
            gesture = "Dance"
        elif all([index, middle, ring]) and not any([thumb, pinky]):
            gesture = "Sit"
        ''' else:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            dist = self.calculate_distance(thumb_tip, index_tip)
            cross_threshold = 0.06
    
            if hand_label == "Right":
                cross_cond = thumb_tip.x > index_tip.x
            else:
                cross_cond = thumb_tip.x < index_tip.x
    
            if (dist < cross_threshold and cross_cond
                    and not middle and not ring and not pinky):
                gesture = "Heart" '''
        return gesture

    def check_gesture_duration(self, current_gesture):
        """检查手势持续时间是否达到要求"""
        current_time = time.time()
        
        # 如果手势发生变化，重置计时器
        if current_gesture not in self.gesture_start_time:
            # 新手势，开始计时
            self.gesture_start_time = {current_gesture: current_time}
            return False
        
        # 检查当前手势是否已经持续了足够时间
        if current_gesture in self.gesture_start_time:
            duration = current_time - self.gesture_start_time[current_gesture]
            if duration >= self.min_hold_time:
                # 达到要求时间，重置计时器并返回True
                self.gesture_start_time.pop(current_gesture, None)
                return True
        
        return False
    
    def process_frame(self, frame):
        """处理帧并检测手势"""
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        gesture = "Unknown"
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                # 手指状态和手势检测
                count, finger_status = self.count_fingers_and_get_status(hand_landmarks, hand_label)
                gesture = self.detect_gesture(hand_landmarks, hand_label, finger_status)
                
                # 绘制手部关键点和连接线
                self.draw_landmarks(frame, hand_landmarks, finger_status)
                
                # 只处理一只手
                break
        
        with self.gesture_lock:
            self.current_gesture = gesture
            
        return frame, gesture
    
    def draw_landmarks(self, frame, hand_landmarks, finger_status):
        """绘制手部关键点和连接线"""
        h, w = frame.shape[:2]
        lm_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
        
        # 绘制骨骼连接线
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),          # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),     # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),   # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)    # 小指
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            cv2.line(frame, lm_coords[start_idx], lm_coords[end_idx], (0, 255, 0), 2)
        
        # 绘制关节点
        for coord in lm_coords:
            cv2.circle(frame, coord, 5, (255, 0, 0), -1)
    
    def get_current_gesture(self):
        """获取当前手势"""
        with self.gesture_lock:
            return self.current_gesture

class BodyPoseControl:
    """上半身姿态识别控制类"""
    def __init__(self):
        self.current_pose = "Unknown"
        self.pose_lock = Lock()
        self.pose_start_time = {}
        self.min_hold_time = 2.0
        
    def calculate_distance(self, point1, point2):
        """计算两个点之间的欧氏距离"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def calculate_angle(self, a, b, c):
        """计算三个点之间的角度"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    def detect_upper_body_pose(self, landmarks):
        """检测上半身姿态"""
        pose = "Unknown"
        
        # 获取上半身关键点坐标
        h, w = 480, 640  # 假设图像尺寸
        keypoints = {}
        
        # 上半身关键点
        upper_body_points = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        
        for point in upper_body_points:
            lm = landmarks.landmark[point.value]
            keypoints[point.name] = (int(lm.x * w), int(lm.y * h))
        
        try:
            # 获取关键点
            left_shoulder = keypoints['LEFT_SHOULDER']
            right_shoulder = keypoints['RIGHT_SHOULDER']
            left_elbow = keypoints['LEFT_ELBOW']
            right_elbow = keypoints['RIGHT_ELBOW']
            left_wrist = keypoints['LEFT_WRIST']
            right_wrist = keypoints['RIGHT_WRIST']
            
            # 计算肩膀中心点
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2, 
                              (left_shoulder[1] + right_shoulder[1]) // 2)
            
            # 1. 左手放在右侧肩膀
            if (self.calculate_distance(left_wrist, right_shoulder) < 40):  
                pose = "Fast_walk"
            
            # 2. 双手交叉检测
            elif (abs(left_wrist[0] - right_wrist[0]) < 50 and  # 双手腕部水平位置相近
                  abs(left_wrist[1] - right_wrist[1]) < 50):    # 双手腕部垂直位置相近
                pose = "Heart"
            
            # 3. 手前伸检测
            elif ((left_wrist[0] > left_elbow[0] + 100 and  # 左手腕在左手肘前方
                   left_elbow[0] > left_shoulder[0] + 80) or  # 左手肘在左肩膀前方
                  (right_wrist[0] < right_elbow[0] - 100 and  # 右手腕在右手肘前方
                   right_elbow[0] < right_shoulder[0] - 80)):  # 右手肘在右肩膀前方
                pose = "Goback"
                
        except Exception as e:
            print(f"姿态检测错误: {e}")
        
        return pose
    
    def check_pose_duration(self, current_pose):
        """检查姿态持续时间是否达到要求"""
        current_time = time.time()
        
        if current_pose not in self.pose_start_time:
            self.pose_start_time = {current_pose: current_time}
            return False
        
        if current_pose in self.pose_start_time:
            duration = current_time - self.pose_start_time[current_pose]
            if duration >= self.min_hold_time:
                self.pose_start_time.pop(current_pose, None)
                return True
        
        return False
    
    def process_frame(self, frame):
        """处理帧并检测上半身姿态"""
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        pose_name = "Unknown"
        
        if results.pose_landmarks:
            pose_name = self.detect_upper_body_pose(results.pose_landmarks)
            
            # 绘制上半身关键点
            self.draw_upper_body_landmarks(frame, results.pose_landmarks)
        
        with self.pose_lock:
            self.current_pose = pose_name
            
        return frame, pose_name
    
    def draw_upper_body_landmarks(self, frame, landmarks):
        """绘制上半身关键点和连接线"""
        h, w = frame.shape[:2]
        
        # 上半身连接线
        upper_body_connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
            (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value)
        ]
        
        # 绘制连接线
        for connection in upper_body_connections:
            start_idx, end_idx = connection
            start_point = (int(landmarks.landmark[start_idx].x * w), 
                          int(landmarks.landmark[start_idx].y * h))
            end_point = (int(landmarks.landmark[end_idx].x * w), 
                        int(landmarks.landmark[end_idx].y * h))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # 绘制关键点
        upper_body_points = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        
        for point in upper_body_points:
            lm = landmarks.landmark[point.value]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (255, 0, 0), -1)
    
    def get_current_pose(self):
        """获取当前姿态"""
        with self.pose_lock:
            return self.current_pose

def standup(ctrl, msg):
    """站立"""
    msg.mode = 12  # Recovery stand
    msg.gait_id = 0
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(12, 0, 10.0)
    print("站立完成")

def temp_stand(ctrl, msg):
    """用于跳跃等行为后的临时插入"""
    msg.mode = 12  
    msg.gait_id = 0
    msg.duration = 500
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(7, 0, 3.0)

def slow_walk(ctrl, msg):
    """慢走"""
    print("执行慢走动作")
    msg.mode = 11  
    msg.gait_id = 27  
    msg.vel_des = [0.2, 0, 0]  
    msg.duration = 2000  
    msg.step_height = [0.05, 0.05]
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(11, 27, 5.0)

def jump(ctrl, msg):
    """跳跃"""
    print("执行跳跃动作")
    msg.mode = 16
    msg.gait_id = 6
    msg.duration = 800
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(11, 27, 5.0)

def turn_left(ctrl, msg):
    """左转"""
    print("执行左转动作")
    msg.mode = 16
    msg.gait_id = 0
    msg.duration = 800
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(11, 27, 5.0)

def turn_right(ctrl, msg):
    """右转"""
    print("执行右转动作")
    msg.mode = 16
    msg.gait_id = 3
    msg.duration = 800
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(11, 27, 5.0)

def stand(ctrl, msg):
    """站定"""
    print("执行站定动作")
    msg.mode = 12  
    msg.gait_id = 0
    msg.duration = 7000
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(7, 0, 3.0)

def hold_left_hand(ctrl, msg):
    """握左手"""
    print("执行握左手动作")
    msg.mode = 62  
    msg.gait_id = 1
    msg.duration = 9000
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(7, 0, 3.0)

def hold_right_hand(ctrl, msg):
    """握右手"""
    print("执行握右手动作")
    msg.mode = 62  
    msg.gait_id = 2
    msg.duration = 9000
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(7, 0, 3.0)

def dance(ctrl, msg):
    """跳芭蕾"""
    print("执行跳芭蕾动作")
    msg.mode = 62  
    msg.gait_id = 11
    msg.duration = 28000
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(7, 0, 3.0)

def sit(ctrl, msg):
    """坐下"""
    print("执行坐下动作")
    msg.mode = 62  
    msg.gait_id = 3
    msg.duration = 1400
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(7, 0, 3.0)

def heart(ctrl, msg):
    """作揖"""
    print("执行作揖动作")
    msg.mode = 64  
    msg.gait_id = 0
    msg.duration = 9000
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(7, 0, 3.0)

def fast_walk(ctrl, msg):
    """小跑"""
    print("执行小跑动作")
    msg.mode = 11 
    msg.gait_id = 10
    msg.vel_des = [1.0, 0, 0]  
    msg.duration = 2000  
    msg.step_height = [0.05, 0.05]
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(11, 27, 5.0)

def goback(ctrl, msg):
    """后退"""
    print("执行后退动作")
    msg.mode = 11  
    msg.gait_id = 27  
    msg.vel_des = [-0.2, 0, 0]  
    msg.duration = 2000  
    msg.step_height = [0.05, 0.05]
    msg.life_count += 1
    ctrl.Send_cmd(msg)
    ctrl.Wait_finish(11, 27, 5.0)



def main():
    # 初始化手势识别
    gesture_control = GestureControl()
    pose_control = BodyPoseControl()
    
    # 初始化机器人控制
    ctrl = Robot_Ctrl()
    ctrl.run()
    msg = robot_control_cmd_lcmt()
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 站立
    standup(ctrl, msg)
    
    print("手势/动作控制说明:")
    print("握拳: 慢走")
    print("拇指伸开: 跳跃")
    print("五指伸开: 站立")
    print("二拇指和中指伸开: 左转")
    print("小拇指伸开: 右转")
    print("二拇指和中指伸开：握左手")
    print("无名指和小指伸开：握右手")
    print("除大拇指外四指伸开：跳芭蕾")
    print("伸开中间三根手指：坐下")
    print("左手放在右侧肩膀（手腕放到肩膀处）：小跑")
    print("伸手：后退")
    print("双手交叉：作揖")
    #print("比心：作揖")
    print("按ESC退出")
    
    last_executed_gesture = "Unknown"
    last_executed_pose = "Unknown"
    last_display_gesture = "Unknown"
    last_display_pose = "Unknown"
    last_action_time = time.time()
    action_cooldown = 2.0  # 动作冷却时间(秒)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                break
            
            # 处理帧并检测手势和上半身姿态
            gesture_frame, current_gesture = gesture_control.process_frame(frame.copy())
            pose_frame, current_pose = pose_control.process_frame(frame.copy())

            # 合并两个处理结果
            combined_frame = cv2.hconcat([gesture_frame, pose_frame])

            if current_gesture != last_display_gesture:
                last_display_gesture = current_gesture
                print(f"检测到手势: {current_gesture}")

            if current_pose != last_display_pose:
                last_display_pose = current_pose
                print(f"检测到姿态: {current_pose}")
            
            # 显示当前手势和姿态
            cv2.putText(combined_frame, f"Gesture: {current_gesture}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_frame, f"Pose: {current_pose}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            '''# 检查手势持续时间
            if gesture_control.check_gesture_duration(current_gesture):
                if current_gesture != last_executed_gesture:
                    last_executed_gesture = current_gesture
                    print(f"手势 {current_gesture} 已保持2秒，执行相应动作...")
                
                    if current_gesture == "Jump":
                        jump(ctrl, msg)
                        temp_stand(ctrl, msg)
                    elif current_gesture == "Stand":
                        stand(ctrl, msg)
                    elif current_gesture == "Slow_walk":
                        slow_walk(ctrl, msg)
                        temp_stand(ctrl, msg)
                    elif current_gesture == "Left":
                        turn_left(ctrl, msg)
                        temp_stand(ctrl, msg)
                    elif current_gesture == "Right":
                        turn_right(ctrl, msg)
                        temp_stand(ctrl, msg)
                    elif current_gesture == "Hold_left_hand":
                        hold_left_hand(ctrl, msg)
                        temp_stand(ctrl, msg)
                    elif current_gesture == "Hold_right_hand":
                        hold_right_hand(ctrl, msg)
                    elif current_gesture == "Dance":
                        dance(ctrl, msg)
                        temp_stand(ctrl, msg)
                    elif current_gesture == "Sit":
                        sit(ctrl, msg)
                    #elif current_gesture == "Heart":
                    #    heart(ctrl, msg) '''
            
            pose_check = pose_control.check_pose_duration(current_pose)
            gesture_check = gesture_control.check_gesture_duration(current_gesture)

            # 检查姿态持续时间
            if pose_check or gesture_check:
                if pose_check:
                    if current_pose != last_executed_pose:
                        last_executed_pose = current_pose
                        print(f"姿态 {current_pose} 已保持2秒，执行相应动作...")
                    
                        # 上半身姿态处理逻辑
                        if current_pose == "Fast_walk":
                            fast_walk(ctrl, msg)
                            temp_stand(ctrl, msg)
                        elif current_pose == "Heart":
                            heart(ctrl, msg)
                            temp_stand(ctrl, msg)
                        elif current_pose == "Goback":
                            goback(ctrl, msg)
                            temp_stand(ctrl, msg)
                else:
                    if current_gesture != last_executed_gesture:
                        last_executed_gesture = current_gesture
                        print(f"手势 {current_gesture} 已保持2秒，执行相应动作...")
                    
                        if current_gesture == "Jump":
                            jump(ctrl, msg)
                            temp_stand(ctrl, msg)
                        elif current_gesture == "Stand":
                            stand(ctrl, msg)
                        elif current_gesture == "Slow_walk":
                            slow_walk(ctrl, msg)
                            temp_stand(ctrl, msg)
                        elif current_gesture == "Left":
                            turn_left(ctrl, msg)
                            temp_stand(ctrl, msg)
                        elif current_gesture == "Right":
                            turn_right(ctrl, msg)
                            temp_stand(ctrl, msg)
                        elif current_gesture == "Hold_left_hand":
                            hold_left_hand(ctrl, msg)
                            temp_stand(ctrl, msg)
                        elif current_gesture == "Hold_right_hand":
                            hold_right_hand(ctrl, msg)
                        elif current_gesture == "Dance":
                            dance(ctrl, msg)
                            temp_stand(ctrl, msg)
                        elif current_gesture == "Sit":
                            sit(ctrl, msg)
                


            # 显示图像
            cv2.imshow("Gesture and Upper Body Pose Control", combined_frame)
            
            # 检查退出键
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        ctrl.quit()
        print("程序退出")

if __name__ == '__main__':
    main()
