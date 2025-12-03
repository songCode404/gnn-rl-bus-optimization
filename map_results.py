import pandas as pd
import pickle
import os

# ==========================================
# 🔧 설정 (파일 경로를 정확히 입력하세요!)
# ==========================================
# 1. 학생분이 가지고 있는 "예측 결과 CSV 파일" 경로
target_csv_file = "Graph-WaveNet/all_predictions.csv" 

# 2. 진짜 ID 정보가 들어있는 "지도 파일" 경로
graph_file = "Graph-WaveNet/data/sensor_graph/adj_mx.pkl"

# 3. 결과를 저장할 새로운 파일 이름
output_csv = "final_result_mapped.csv"

# ==========================================
# 🚀 매핑 작업 시작
# ==========================================
print(f"📂 CSV 파일({target_csv_file})을 불러옵니다...")

# 1. CSV 파일 읽기 (여기가 중요! np.load 대신 read_csv 사용)
try:
    df = pd.read_csv(target_csv_file, encoding='cp949')
except:
    df = pd.read_csv(target_csv_file, encoding='utf-8')

print(f"✅ CSV 로드 완료! 크기: {df.shape}")
print(f"   컬럼 목록: {df.columns.tolist()}")

# 2. 진짜 ID 리스트 불러오기 (지도 파일에서)
if not os.path.exists(graph_file):
    print(f"❌ 지도 파일({graph_file})이 없습니다! update_graph_ids.py를 먼저 실행하세요.")
    exit()

with open(graph_file, 'rb') as f:
    sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f)

print(f"✅ 진짜 ID 목록 로드 완료! (총 {len(sensor_ids)}개)")

# ==========================================
# 🔗 ID 매핑하기
# ==========================================
# 가정: CSV 파일에 'node' 또는 'index'라는 컬럼이 있거나, 
# 혹은 행 순서(Index) 자체가 0번~705번 노드를 의미한다고 가정합니다.

# 진짜 ID를 담을 리스트 만들기
real_ids = []

# 케이스 1: CSV 안에 'node'라는 컬럼이 있어서 거기에 0, 1, 2... 가 적혀있는 경우
if 'node' in df.columns:
    print("👉 'node' 컬럼을 기준으로 매핑합니다.")
    for idx in df['node']:
        real_ids.append(sensor_ids[int(idx)])
    df['Real_Node_ID'] = real_ids

# 케이스 2: 컬럼은 없고 그냥 행 순서대로 0번 노드, 1번 노드인 경우 (보통 이렇습니다)
elif len(df) == len(sensor_ids): # 행 개수가 딱 706개라면
    print("👉 행 순서(Index)를 기준으로 매핑합니다.")
    df['Real_Node_ID'] = sensor_ids

# 케이스 3: 데이터가 너무 많아서 (시간 x 노드)인 경우 -> 이건 좀 복잡해서 패스
else:
    print("⚠️ 경고: CSV 파일 구조를 정확히 몰라서 매핑 방식을 추측합니다.")
    # 혹시 'index' 컬럼이 있는지 확인
    if 'index' in df.columns:
         df['Real_Node_ID'] = df['index'].apply(lambda x: sensor_ids[int(x)] if x < len(sensor_ids) else 'Unknown')
    else:
        print("❌ 매핑할 기준 컬럼(node, index 등)을 못 찾았습니다.")
        print("   -> sensor_ids 리스트만 따로 저장해 드릴 테니 엑셀에서 붙여넣으세요.")
        pd.DataFrame({'Real_Node_ID': sensor_ids}).to_csv("real_ids_list.csv", index=False)
        print("   💾 'real_ids_list.csv' 저장됨. 이걸 복사해서 쓰세요!")
        exit()

# ==========================================
# 💾 저장
# ==========================================
# 보기 좋게 ID 컬럼을 맨 앞으로 보내기
cols = ['Real_Node_ID'] + [c for c in df.columns if c != 'Real_Node_ID']
df = df[cols]

df.to_csv(output_csv, index=False, encoding='cp949')
print(f"\n🎉 변환 성공! '{output_csv}' 파일을 열어보세요.")
print("이제 0번 대신 진짜 정류장 ID가 보일 겁니다!")