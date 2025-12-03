import numpy as np
import os

# 확인하고 싶은 데이터가 있는 폴더
# (혹시 Graph-WaveNet/data/MY_BUS 라면 경로를 수정하세요)
target_dir = "Graph-WaveNet/data" 

print(f"📂 '{target_dir}' 폴더의 데이터 모양을 확인합니다...\n")

files = ["train.npz", "val.npz", "test.npz"]

for f in files:
    file_path = os.path.join(target_dir, f)
    
    if not os.path.exists(file_path):
        print(f"❌ {f}: 파일이 없습니다. 경로를 확인해주세요.")
        continue
    
    try:
        data = np.load(file_path)
        x = data['x']
        y = data['y']
        
        print(f"📄 파일명: {f}")
        print(f"   👉 입력(X) 모양: {x.shape}")
        print(f"   👉 정답(Y) 모양: {y.shape}")
        
        # 차원 해석 (Make Speed Dataset 기준)
        # 보통 (Samples, Time, Nodes, Features) 순서로 생성됨
        dims = x.shape
        print(f"   🔍 해석:")
        print(f"      - 데이터 개수 (Samples): {dims[0]}개")
        print(f"      - 첫번째 차원: {dims[1]} (아마도 Time/Seq_Len)")
        print(f"      - 두번째 차원: {dims[2]} (아마도 Nodes/정류장수)")
        print(f"      - 세번째 차원: {dims[3]} (아마도 Features/속도)")
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ {f} 읽기 실패: {e}")

print("\n✅ 확인 완료!")