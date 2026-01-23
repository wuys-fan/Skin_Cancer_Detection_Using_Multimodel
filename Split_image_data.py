import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === Cấu hình đường dẫn ===
image_dir = None # Thư mục chứa ảnh gốc
metadata_path = None  # File ground truth
output_dir = None # Nơi lưu dữ liệu sau khi chia

# === Đọc file ground truth ===
metadata = pd.read_csv(metadata_path)

# Xác định cột class
class_cols = [col for col in metadata.columns if col != 'image']

# Thêm cột nhãn (label) tương ứng với cột có giá trị 1
metadata['label'] = metadata[class_cols].idxmax(axis=1)

# === Tạo thư mục đầu ra ===
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# === Danh sách lưu thông tin chia ===
train_meta, val_meta, test_meta = [], [], []

# === Chia dữ liệu theo từng lớp ===
for label in class_cols:
    label_df = metadata[metadata[label] == 1]

    # Nếu không có ảnh nào thuộc lớp này thì bỏ qua
    if len(label_df) == 0:
        print(f"⚠️ Bỏ qua lớp '{label}' vì không có ảnh nào.")
        continue

    # Nếu chỉ có 1 hoặc 2 ảnh → không chia nhỏ nữa
    if len(label_df) < 3:
        train_df = label_df
        val_df = pd.DataFrame(columns=metadata.columns)
        test_df = pd.DataFrame(columns=metadata.columns)
    else:
        # 70% train, 30% còn lại chia tiếp 20% val, 10% test
        train_df, temp_df = train_test_split(label_df, test_size=0.3, random_state=42, shuffle=True)
        val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42, shuffle=True)

    # Lưu lại cho thống kê
    train_meta.append(train_df)
    val_meta.append(val_df)
    test_meta.append(test_df)

    # === Tạo thư mục con cho từng class ===
    for split, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        split_class_dir = os.path.join(output_dir, split, label)
        os.makedirs(split_class_dir, exist_ok=True)

        count = 0
        for img_name in df['image']:
            # Tự động nhận diện đuôi ảnh
            found = False
            for ext in [".jpg", ".jpeg", ".png"]:
                src = os.path.join(image_dir, img_name + ext)
                if os.path.exists(src):
                    dst = os.path.join(split_class_dir, img_name + ext)
                    shutil.copy2(src, dst)
                    found = True
                    count += 1
                    break
            if not found:
                print(f"⚠️ Không tìm thấy ảnh cho {img_name}")
        
        print(f"📂 {split.upper()} - {label}: {count} ảnh")

# === Gộp tất cả lại để thống kê tổng ===
train_meta = pd.concat(train_meta, ignore_index=True)
val_meta = pd.concat(val_meta, ignore_index=True)
test_meta = pd.concat(test_meta, ignore_index=True)

# === In thống kê tổng ===
print("\n✅ Hoàn tất chia dataset ISIC 2019 (70% train, 20% val, 10% test)")
print(f"Tổng ảnh - Train: {len(train_meta)}, Val: {len(val_meta)}, Test: {len(test_meta)}")

# === In thống kê số lượng ảnh từng lớp ===
print("\n📊 Số lượng ảnh theo từng lớp:")
for label in class_cols:
    n_train = len(train_meta[train_meta[label] == 1]) if label in train_meta.columns else 0
    n_val = len(val_meta[val_meta[label] == 1]) if label in val_meta.columns else 0
    n_test = len(test_meta[test_meta[label] == 1]) if label in test_meta.columns else 0
    print(f"{label}: Train={n_train}, Val={n_val}, Test={n_test}")
