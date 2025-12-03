import numpy as np
import os

# 데이터 경로
data_dir = "Graph-WaveNet/data"
files = ["train.npz", "val.npz", "test.npz"]

print("🔧 데이터 차원 교정 (Swap 1 <-> 3) 시작...")

for f in files:
    file_path = os.path.join(data_dir, f)
    
    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        continue

    with np.load(file_path) as data:
        x = data['x']
        y = data['y']
        print(f"\n📄 [{f}] 현재 모양: {x.shape}")

        # 목표: (Samples, 12, 706, 1) 
        # 현재: (Samples, 1, 706, 12)
        
        # 1번째(Index 1)와 3번째(Index 3) 차원을 맞바꿉니다.
        # Transpose 순서: (0, 3, 2, 1) -> (Sample, Time, Node, Feature)
        
        if x.shape[1] == 1 and x.shape[3] == 12:
            x_new = x.transpose(0, 3, 2, 1)
            y_new = y.transpose(0, 3, 2, 1)
            
            np.savez(file_path, x=x_new, y=y_new)
            print(f"  ✅ 교정 완료! 바뀐 모양: {x_new.shape}")
            print(f"     (Sample={x_new.shape[0]}, Time={x_new.shape[1]}, Node={x_new.shape[2]}, Feat={x_new.shape[3]})")
        else:
            print("  ⚠️ 이미 바뀌어 있거나 다른 모양입니다. 건너뜁니다.")

print("-" * 30)
print("📢 이제 Graph-WaveNet 폴더로 데이터를 옮기고 학습하세요!")