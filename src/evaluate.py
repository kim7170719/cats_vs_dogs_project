import torch
from torchvision import transforms, models
from PIL import Image
import os

# 資料轉換
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加載模型
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('models/cats_vs_dogs_resnet18.pth'))
model.eval()

def predict_image(image_path):
    img = Image.open(image_path)
    img = data_transform(img)
    img = img.unsqueeze(0)  # 增加 batch 維度

    outputs = model(img)
    _, preds = torch.max(outputs, 1)

    return "Dog" if preds.item() == 1 else "Cat"

if __name__ == "__main__":
    test_dir = 'data/test'
    test_images = os.listdir(test_dir)

    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        prediction = predict_image(img_path)
        print(f'{img_name}: {prediction}')
