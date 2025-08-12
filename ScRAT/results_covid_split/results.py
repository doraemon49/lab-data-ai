# python results_covid_split/results.py
file_path = 'logs_covid_split/manual_annotation_sample_cells_100_augment.txt'
print(file_path)
# 결과를 저장할 리스트
repeat_lines = []

# 파일을 읽고 특정 줄만 추출
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        if 'Test AUC 0.0' in line:
            repeat_lines.append(line.strip())

# 결과 출력
for line in repeat_lines:
    print(line)