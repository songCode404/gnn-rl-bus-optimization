import pandas as pd
import numpy as np
import os

# ==========================================
# 🔧 1. 설정
# ==========================================
input_file = "data/14X번_버스_모음_데이터.csv"  # 경로 확인 필수!
output_dir = "data/MY_BUS"             
seq_len = 12                           
horizon = 12                           

# ==========================================
# 📂 2. 데이터 불러오기
# ==========================================
print(f"📂 데이터({input_file})를 불러옵니다...")
try:
    df = pd.read_csv(input_file, encoding='cp949')
except:
    df = pd.read_csv(input_file, encoding='utf-8')

df.columns = df.columns.str.strip() # 공백 제거
print(f"✅ 데이터 로드 완료: {len(df)}행")

# ==========================================
# 🕒 3. 시간대별 컬럼 찾기
# ==========================================
# '운행시간_00시' ~ '운행시간_23시' 패턴 찾기
time_cols = [col for col in df.columns if col.startswith('운행시간_') and col.endswith('시')]
time_cols.sort()

print(f"⏰ 찾은 시간 컬럼 ({len(time_cols)}개): {time_cols}")

if len(time_cols) == 0:
    raise ValueError("❌ 시간대 컬럼을 못 찾았습니다! 컬럼 이름을 다시 확인해주세요.")

# ==========================================
# 🔄 4. 데이터 구조 변환
# ==========================================
date_col = '기준_날짜'
node_col = '출발_정류장_ID' 

print(f"기준 컬럼 -> 날짜: {date_col}, 노드: {node_col}")

# ⭐⭐ [수정됨] 이름 충돌 방지: 기존 '운행시간' 컬럼 삭제 ⭐⭐
if '운행시간' in df.columns:
    print("🧹 데이터 정리를 위해 기존 '운행시간' 컬럼을 삭제합니다.")
    df = df.drop(columns=['운행시간'])

# 1. 세로로 길게 펴기
print("🔄 데이터를 시간순으로 정렬하는 중...")
# 이제 충돌 없이 '운행시간'이라는 이름으로 만들 수 있음
df_melted = df.melt(id_vars=[date_col, node_col], value_vars=time_cols, 
                    var_name='시간_str', value_name='운행시간')

# 2. 진짜 시간(datetime) 만들기
df_melted['시간_int'] = df_melted['시간_str'].str.extract('(\d+)').astype(int)

# 날짜 + 시간 합치기
df_melted['일시'] = pd.to_datetime(df_melted[date_col].astype(str), format='%Y%m%d') + \
                    pd.to_timedelta(df_melted['시간_int'], unit='h')

# 3. 정렬
df_melted = df_melted.sort_values(['일시', node_col])

# ==========================================
# 📊 5. 행렬 만들기
# ==========================================
print("📊 행렬로 변환 중...")
# pivot_table 사용 (중복 데이터는 평균값 사용)
df_pivot = df_melted.pivot_table(index='일시', columns=node_col, values='운행시간', aggfunc='mean')

# 결측치 채우기
df_pivot = df_pivot.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

data_matrix = df_pivot.values
print(f"✅ 최종 데이터 행렬 크기: {data_matrix.shape} (시간 x 구간수)")

# ==========================================
# ✂️ 6. 학습용 4차원 텐서 만들기
# ==========================================
print("✂️ 학습용 데이터로 자르는 중...")
x_list, y_list = [], []
num_samples = data_matrix.shape[0]

for i in range(num_samples - seq_len - horizon + 1):
    x = data_matrix[i : i+seq_len, :]       
    y = data_matrix[i+seq_len : i+seq_len+horizon, :] 
    x_list.append(x)
    y_list.append(y)

# (Samples, Time, Nodes, 1) 형태로 변환
x_arr = np.expand_dims(np.array(x_list), axis=-1)
y_arr = np.expand_dims(np.array(y_list), axis=-1)

print(f"📦 생성된 텐서 형태 - X: {x_arr.shape}, Y: {y_arr.shape}")

# ==========================================
# 💾 7. 저장
# ==========================================
os.makedirs(output_dir, exist_ok=True)

n_train = int(len(x_arr) * 0.7)
n_val = int(len(x_arr) * 0.2)

np.savez(f"{output_dir}/train.npz", x=x_arr[:n_train], y=y_arr[:n_train])
np.savez(f"{output_dir}/val.npz", x=x_arr[n_train:n_train+n_val], y=y_arr[n_train:n_train+n_val])
np.savez(f"{output_dir}/test.npz", x=x_arr[n_train+n_val:], y=y_arr[n_train+n_val:])

print(f"\n🎉 성공! {output_dir} 폴더에 학습 데이터가 저장되었습니다.")
# 나중에 train.py 실행할 때 필요한 노드 개수 출력
print(f"📢 [중요] train.py 실행 시 --num_nodes {data_matrix.shape[1]} 옵션을 사용하세요!")