import os
import shutil
import random

def split_data(base_dir, train_dir, val_dir, split_ratio=0.8):
    # 檢查資料夾是否存在
    cat_dir = os.path.join(base_dir, 'Cat')
    dog_dir = os.path.join(base_dir, 'Dog')

    if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
        print(f"錯誤: 找不到資料夾 {cat_dir} 或 {dog_dir}")
        return

    # 創建訓練與驗證資料夾
    os.makedirs(os.path.join(train_dir, 'Cat'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'Dog'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'Cat'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'Dog'), exist_ok=True)

    # 遍歷並隨機分配圖片
    for category in ['Cat', 'Dog']:
        src_dir = os.path.join(base_dir, category)
        all_images = os.listdir(src_dir)
        random.shuffle(all_images)
        split_idx = int(len(all_images) * split_ratio)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        for img in train_images:
            shutil.copy(os.path.join(src_dir, img), os.path.join(train_dir, category, img))

        for img in val_images:
            shutil.copy(os.path.join(src_dir, img), os.path.join(val_dir, category, img))

    print('資料分割完成！')

if __name__ == "__main__":
    base_dir = 'data/raw'  # 根據實際情況更新路徑
    train_dir = 'data/processed/train'
    val_dir = 'data/processed/val'
    split_data(base_dir, train_dir, val_dir)
