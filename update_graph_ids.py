import numpy as np
import pandas as pd
import pickle
import os

# ==========================================
# 🔧 설정
# ==========================================
# 우리가 만든 데이터 파일 (여기서 진짜 ID를 뽑아옵니다)
input_file = "Graph-WaveNet/MY_BUS/train.npz" 
# 저장할 지도 파일 위치
graph_file = "Graph-WaveNet/data/sensor_graph/adj_mx.pkl"

# ==========================================
# 🕵️‍♂️ 진짜 ID 추출하기
# ==========================================
# 1. 원본 데이터 로드 (순서를 알기 위해)
# 아까 make_speed_dataset.py에서 저장할 때 
# columns(노드 순서) 정보를 따로 저장 안 했으므로,
# 원본 CSV에서 다시 순서를 알아내야 합니다.

csv_file = "data/14X번_버스_모음_데이터.csv"
print(f"📂 원본 CSV({csv_file})에서 노드 ID를 추출합니다...")

try:
    df = pd.read_csv(csv_file, encoding='cp949')
except:
    df = pd.read_csv(csv_file, encoding='utf-8')

# 아까 코드와 똑같은 로직으로 정렬해서 순서를 맞춥니다.
node_col = '출발_정류장_ID' # 또는 '구간ID'
df.columns = df.columns.str.strip()

# 유니크한 노드 ID를 뽑고 정렬 (make_speed_dataset.py의 pivot_table은 자동 정렬됨)
real_node_ids = sorted(df[node_col].unique().astype(str))

print(f"✅ 추출된 노드 개수: {len(real_node_ids)}개")
print(f"   (예시: {real_node_ids[:5]} ...)")

# ==========================================
# 🗺️ 지도 파일 업데이트
# ==========================================
# 1. 기존 빈 지도 로드 (없으면 새로 생성)
num_nodes = len(real_node_ids)
adj_mx = np.eye(num_nodes) # 단위 행렬 (연결 관계는 AI가 학습)

# 2. ID 매핑 딕셔너리 생성
# {'100100124': 0, '100100125': 1 ...}
sensor_id_to_ind = {k: v for v, k in enumerate(real_node_ids)}

# 3. 덮어쓰기
os.makedirs(os.path.dirname(graph_file), exist_ok=True)
with open(graph_file, 'wb') as f:
    pickle.dump([real_node_ids, sensor_id_to_ind, adj_mx], f)

print(f"🎉 지도 파일 업데이트 완료: {graph_file}")
print("이제 '0번 노드'가 아니라 '100100124번 노드'라고 부를 수 있습니다!")