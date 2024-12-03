import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('/root/Jujube-YOLO/ultralytics-main/ultralytics/cfg/models/11/Jujube-YOLO.yaml')
    model.train(data='/root/ultralytics-main/yumi.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                single_cls=False,  # 是否是单类别检测
                batch=12,
                workers=12,
                device='0',
                optimizer='SGD',
                amp=True,
                project='runs',
                name='',
                )
  