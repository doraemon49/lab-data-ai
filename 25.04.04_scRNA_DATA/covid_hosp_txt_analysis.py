# python data/covid_hosp/txt_analysis.py
import pandas as pd

# 예시 파일 경로
file_path = "data/covid_hosp/E-MTAB-9357_txt/heathlab_dc_9_17_pbmc_gex_library_1_2.txt"

# 파일 읽기 (TSV 형식: 탭 구분자)
df = pd.read_csv(file_path, sep="\t")

# 행과 열 개수 출력
print(f"행(row; cell/barcode) 개수: {df.shape[0]}")
print(f"열(column; 유전자) 개수: {df.shape[1]}")

print(df.iloc[:10, :5])

"""
# 1_1
행(row; cell/barcode) 개수: 5719
열(column; 유전자) 개수: 24967
                     Unnamed: 0  A1BG  A1BG-AS1  A2M  A2M-AS1
0  AAACCTGAGAATTGTG-1-1:1_1:1-1   0.0       0.0  0.0      0.0
1  AAACCTGAGCAGCGTA-1-1:1_1:1-1   0.0       0.0  0.0      0.0
2  AAACCTGAGCTACCGC-1-1:1_1:1-1   0.0       0.0  0.0      0.0
3  AAACCTGAGTAGCGGT-1-1:1_1:1-1   0.0       0.0  0.0      0.0
4  AAACCTGCACAAGCCC-1-1:1_1:1-1   0.0       0.0  0.0      0.0
5  AAACCTGCACACAGAG-1-1:1_1:1-1   0.0       0.0  0.0      0.0
6  AAACCTGCATACAGCT-1-1:1_1:1-1   0.0       0.0  0.0      0.0
7  AAACCTGGTCTCCACT-1-1:1_1:1-1   0.0       0.0  0.0      0.0
8  AAACCTGGTCTTCTCG-1-1:1_1:1-1   0.0       0.0  0.0      0.0
9  AAACCTGTCATCGATG-1-1:1_1:1-1   0.0       0.0  0.0      0.0

1_2
행(row; cell/barcode) 개수: 3259
열(column; 유전자) 개수: 24967

                     Unnamed: 0      A1BG  A1BG-AS1  A2M  A2M-AS1
0  AAACCTGAGGCCCTCA-1-1:1_2:1-2  5.340616       0.0  0.0      0.0
1  AAACCTGAGTCAAGGC-1-1:1_2:1-2  0.000000       0.0  0.0      0.0
2  AAACCTGAGTGCCAGA-1-1:1_2:1-2  0.000000       0.0  0.0      0.0
3  AAACCTGCATGCCCGA-1-1:1_2:1-2  0.000000       0.0  0.0      0.0
4  AAACCTGCATTTCAGG-1-1:1_2:1-2  0.000000       0.0  0.0      0.0
5  AAACCTGGTATATCCG-1-1:1_2:1-2  0.000000       0.0  0.0      0.0
6  AAACCTGGTCCAGTAT-1-1:1_2:1-2  0.000000       0.0  0.0      0.0
7  AAACCTGTCCGTCATC-1-1:1_2:1-2  0.000000       0.0  0.0      0.0
8  AAACCTGTCTTGACGA-1-1:1_2:1-2  0.000000       0.0  0.0      0.0
9  AAACGGGAGGAATGGA-1-1:1_2:1-2  0.000000       0.0  0.0      0.0

2_1
행(row; cell/barcode) 개수: 7791
열(column; 유전자) 개수: 24967

2_2
행(row; cell/barcode) 개수: 7093
열(column; 유전자) 개수: 24967

...

51_1
행(row; cell/barcode) 개수: 2191
열(column; 유전자) 개수: 24967

51_2
행(row; cell/barcode) 개수: 2721
열(column; 유전자) 개수: 24967

...

100_1
행(row; cell/barcode) 개수: 1173
열(column; 유전자) 개수: 24967

100_2
행(row; cell/barcode) 개수: 1001
열(column; 유전자) 개수: 24967

...

1053BW
행(row; cell/barcode) 개수: 2604
열(column; 유전자) 개수: 24967

BP0219101
행(row; cell/barcode) 개수: 754
열(column; 유전자) 개수: 24967

CL2
행(row; cell/barcode) 개수: 7972
열(column; 유전자) 개수: 24967

Mix_donor1
행(row; cell/barcode) 개수: 2881
열(column; 유전자) 개수: 24967
"""
