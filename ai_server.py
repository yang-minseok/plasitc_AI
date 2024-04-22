import torch
import os

# 모델 파일 경로

# 이미지 파일 경로
img_path = 'image'
pt_file_path = 'AI_model/best.pt'
# model load
def model_load(pt_path):
    # CUDA 사용 가능시 cuda 사용, 없으면 cpu 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 모델 로드
    model = torch.hub.load('ultralytics/yolov5','custom',pt_path,force_reload=True)
    model = model.to(device)
    return model

model = model_load(pt_file_path)

# 각 이미지마다 결과 표출
image_list = os.listdir(img_path)
for image in image_list:
    result = model(img_path + '/' + image)
    result.show()

