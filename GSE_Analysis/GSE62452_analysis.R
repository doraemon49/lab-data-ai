test <- c(-1, 0, 1)
test

# install.packages("installr")
# library(installr)
# check.for.updates.R()
# install.R()
# .libPaths()

# 파일 경로 설정
GSE62452 <- "/home/kim89/GES62452.csv"

# CSV 파일 읽기
data <- read.csv(GSE62452, header = TRUE)

# Data 정상화 (Optional)
# RNA-seq일 경우, 데이터가 로그 변환되어 있다면 이 과정은 생략 가능
# For array data or non-log-transformed data:
# data <- log2(data + 1)

# 데이터 분포 확인
hist(data$X.E005.Tc8., main="Distribution of X.E005.Tc8.", xlab="Expression levels")
summary(data$X.E005.Tc8.)

# 데이터 확인
print(head(data))

#     GeneSymbol  ID_REF X.E005.Tc8. X.E005.Tn. X.E012.Tp2. X.E12.Tn. X.E021.Tc8.
# 1       OR4F17 7896740     1.98519    2.22654     2.20386   1.89503     2.22880
# 2 LOC100134822 7896742     4.27115    5.08438     4.40239   5.53027     5.13266
# 3       OR4F29 7896744     4.10218    4.26139     4.20362   4.47237     3.84170
# 4 LOC100287934 7896754     5.37197    5.68217     6.17415   4.83325     4.60658
# 5       FAM87B 7896756     2.66610    3.39605     2.86317   2.79406     2.82701
# 6    LINC01128 7896759     4.59326    4.53481     4.31149   4.60532     4.06294

####  limma   ######################################################
# 필요한 패키지 설치 및 로드
if (!requireNamespace("limma", quietly = TRUE)) {
    install.packages("limma")
}
library(limma)

# 데이터 불러오기
data <- read.csv(GSE62452, header = TRUE)
# GeneSymbol 및 ID_REF 열 제외한 발현량 데이터만 선택
# 세 번째 열부터가 발현 데이터이므로 [, -c(1, 2)]로 첫 번째, 두 번째 열을 제외
data_expression <- data[, -c(1, 2)]

# GeneSymbol에 중복이 있는지 확인 
sum(duplicated(data$GeneSymbol)) # 2193개 중복...
gene_symbols <- data$GeneSymbol  # GeneSymbol 열만 따로 저장
# GeneSymbol을 기준으로 발현량 평균 계산
data_aggregated <- aggregate(data_expression, by = list(GeneSymbol = gene_symbols), FUN = mean)
sum(duplicated(data_aggregated$GeneSymbol)) # 0 # 이제 중복된 GeneSymbol 없음

# 결과에서 GeneSymbol 열을 추출해 rownames로 설정
rownames(data_aggregated) <- data_aggregated$GeneSymbol

# 더 이상 필요 없는 GeneSymbol 열을 제거
data_numeric <- data_aggregated[, -1]
nrow(data_numeric) # 20002 # 22195개에서 줄어듦

# head(rownames(data_numeric))
# nrow(data_numeric) # 22195 rows
# head(colnames(data_numeric))
# ncol(data_numeric) # 130 columns

# 샘플 조건 정의
# group 정보는 각 샘플이 속한 조건을 지정하는 벡터로 정의
# 샘플의 순서대로, Tumor인지 Nomal인지 기입해준다.
group <- factor(c(
  "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N",
  "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N",
  "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N",
  "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N",
  "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "T", "N",
  "T", "N", "T", "N", "T", "N", "T", "N", "T", "N", "N", "T", "N", "T", "T", "T",
  "N", "T", "N", "T", "T", "N", "T", "N", "T", "T", "N", "T", "T", "N", "T", "T",
  "N", "T", "T", "T", "N", "N", "T", "N", "T", "N", "T", "T", "N", "T", "N", "T",
  "N", "T"
))
# Design Matrix 생성
design <- model.matrix(~ 0 + group)
# head(design)
#   groupN groupT
# 1      0      1
# 2      1      0


##############
# 이제, Linear Model fitting
fit <- lmFit(data_numeric, design)

# Contrasts 설정 
contrast.matrix <- makeContrasts(Tumor_vs_Nomal = groupT - groupN, levels = design)

# Contrasts 적용
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

# limma DEGs 결과 확인
deg_limma <- topTable(fit2, adjust.method = "BH", number = Inf)
head(deg_limma, 10) # 상위 10개의 DEGs 출력
#            logFC  AveExpr        t      P.Value    adj.P.Val        B
# LAMC2   2.646110 6.061879 11.32408 3.550988e-21 7.102686e-17 37.45794
# LAMB3   2.084578 5.486133 10.92686 3.525125e-20 3.525478e-16 35.22765
# TSPAN1  2.384041 5.796048 10.72418 1.136131e-19 7.574961e-16 34.09015

# adf.P.Val < 0.05 및  abs(log2 FC) > 1.5인 DEG 필터링
deg_limma_filtered <- deg_limma[deg_limma$adj.P.Val < 0.05 & abs(deg_limma$logFC) > 1.5, ]
nrow(deg_limma_filtered)  # 필터링된 DEG 개수 확인 # 88 # 20002개 중에서 선별됨.
rownames(deg_limma_filtered)
head(deg_limma_filtered)   # 필터링된 상위 DEGs 확인
#           logFC  AveExpr        t      P.Value    adj.P.Val        B
# LAMC2  2.646110 6.061879 11.32408 3.550988e-21 7.102686e-17 37.45794
# LAMB3  2.084578 5.486133 10.92686 3.525125e-20 3.525478e-16 35.22765
# TSPAN1 2.384041 5.796048 10.72418 1.136131e-19 7.574961e-16 34.09015


# 결과 파일로 저장
# write.csv(deg_limma_filtered, file = "deg_limma_results.csv")


####  limma - Paired   ######################################################
# 파일 경로 설정
GSE62452 <- "/home/kim89/GES62452.csv"

# CSV 파일 읽기
data <- read.csv(GSE62452, header = TRUE)

# 데이터 분포 확인
hist(data$X.E005.Tc8., main="Distribution of X.E005.Tc8.", xlab="Expression levels")
summary(data$X.E005.Tc8.)

# 데이터 확인
# print(head(data))

#     GeneSymbol  ID_REF X.E005.Tc8. X.E005.Tn. X.E012.Tp2. X.E12.Tn. X.E021.Tc8.
# 1       OR4F17 7896740     1.98519    2.22654     2.20386   1.89503     2.22880
# 2 LOC100134822 7896742     4.27115    5.08438     4.40239   5.53027     5.13266
# 3       OR4F29 7896744     4.10218    4.26139     4.20362   4.47237     3.84170

# 필요한 패키지 설치 및 로드
if (!requireNamespace("limma", quietly = TRUE)) {
    install.packages("limma")
}
library(limma)

# GeneSymbol 및 ID_REF 열 제외한 발현량 데이터만 선택
data_expression <- data[, -c(1, 2)]

# GeneSymbol에 중복이 있는지 확인 # 2193개 중복...
sum(duplicated(data$GeneSymbol))
gene_symbols <- data$GeneSymbol      # GeneSymbol 열만 따로 저장
# GeneSymbol을 기준으로 발현량 평균 계산
data_aggregated <- aggregate(data_expression, by = list(GeneSymbol = gene_symbols), FUN = mean)
sum(duplicated(data_aggregated$GeneSymbol)) # 0 # 이제 중복된 GeneSymbol 없음

# 결과에서 GeneSymbol 열을 추출해 rownames로 설정
rownames(data_aggregated) <- data_aggregated$GeneSymbol

# 더 이상 필요 없는 GeneSymbol 열을 제거
data_numeric <- data_aggregated[, -1]
nrow(data_numeric) # 20002 # 22195에서 줄어듦.

# Pair 정보를 읽기
# 필요한 패키지 설치 및 로드
if (!require("readxl")) install.packages("readxl")
library(readxl)
pair_data <- read_excel("/home/kim89/GSE62452_paired_sample_information.xlsx")
pair_info <- pair_data[,c("Sample", "Paired Sample ID")]
# nrow(pair_info) # 130 rows
# 샘플 이름을 맞추기 위해 열 이름을 수정
colnames(pair_info) <- c("Sample", "Pair") 

# 데이터를 pair 정보와 매칭
paired_samples <- pair_info[complete.cases(pair_info), ]  # 쌍이 있는 데이터만 선택
# nrow(paired_samples) # 120 rows
# colnames(data_numeric)에서 "X." 제거 및 마침표를 하이픈으로 변환
adjusted_colnames <- gsub("^X\\.", "", colnames(data_numeric))  # "X." 제거
adjusted_colnames <- gsub("\\.(?=\\w)", "-", adjusted_colnames, perl = TRUE)  # 중간 마침표만 하이픈으로 변환 (lookahead 사용)
adjusted_colnames <- gsub("\\.$", "", adjusted_colnames)  # 마지막 마침표 제거
# 샘플 이름을 맞추기 위해 수정된 열 이름으로 비교
matched_samples <- adjusted_colnames %in% paired_samples$Sample
# print(sum(matched_samples)) # 120개. 

# matched_samples의 TRUE인 열만 선택되어 paired_data에 저장
paired_data <- data_numeric[, matched_samples]
# length(paired_data) # pair이 있는 120개
# length(data_numeric) # 전체 130개

# 그룹과 쌍 정보 생성 # 샘플 이름에 "Tc","Tp" 또는 마지막에 "T"가 가 포함된 경우 그 샘플은 "Tumor"로, 그렇지 않은 경우는 "Normal"로 분류
group <- factor(ifelse(grepl("Tc|Tp|T$", paired_samples$Sample), "Tumor", "Normal"))
# group 데이터뿐만아니라, pair 데이터까지 고려함.
pair <- factor(paired_samples$Pair)
# head(group)
# [1] Tumor  Normal Tumor  Normal Tumor  Normal
# Levels: Normal Tumor
# head(pair)
# [1] 30 30 35 35 46 46
# 60 Levels: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ... 60

# 디자인 매트릭스 생성 (paired analysis) # 절편은 제외 ~0
design <- model.matrix(~ 0 + group + pair)
# head(design) # Tumor이면 'groupTumor' 값이 1. # pair n 의 값이 1
  # groupNormal groupTumor pair2 pair3 pair4 pair5 pair6 pair7 pair8 pair9 pair10 ... pair30 pair31 pair32 pair33 pair34 pair35
# 1          0          1     0     0     0     0     0     0     0     0      0 ...      1      0      0      0      0      0
# 2          1          0     0     0     0     0     0     0     0     0      0 ...      1      0      0      0      0      0
# 3          0          1     0     0     0     0     0     0     0     0      0 ...      0      0      0      0      0      1
# 4          1          0     0     0     0     0     0     0     0     0      0 ...      0      0      0      0      0      1

# Linear Model fitting
fit <- lmFit(paired_data, design)

# Contrasts 설정 (Tumor vs Normal 비교)
contrast.matrix <- makeContrasts(Tumor_vs_Normal = groupTumor - groupNormal, levels = design)

# Contrasts 적용
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

# DEGs 결과 확인
deg_limma_pair <- topTable(fit2, adjust.method = "BH", number = Inf)
head(deg_limma_pair, 10)  # 상위 10개의 DEGs 출력
nrow(deg_limma_pair)  # 20002

# adj.P.Val < 0.05 및 abs(logFC) > 1.5인 DEG 필터링
deg_limma_pair_filtered <- deg_limma_pair[deg_limma_pair$adj.P.Val < 0.05 & abs(deg_limma_pair$logFC) > 1.5, ]

nrow(deg_limma_pair_filtered)  # 필터링된 DEG 개수 확인 # 88 # 20002개 중에서 선별됨.
rownames(deg_limma_pair_filtered)
head(deg_limma_pair_filtered)  # 필터링된 상위 DEGs 확인
#            logFC  AveExpr        t      P.Value    adj.P.Val        B
# TSPAN1  2.511219 5.789988 13.48984 3.157358e-20 4.406567e-16 35.51121
# TMPRSS4 2.165359 4.779879 13.42411 3.970774e-20 4.406567e-16 35.29028
# LAMC2   2.757130 6.049321 13.09952 1.241606e-19 9.185815e-16 34.19055

# 결과 파일로 저장
# write.csv(deg_limma_pair, file = "deg_limma_paired_results_all.csv")
# write.csv(deg_limma_pair_filtered, file = "deg_limma_paired_results.csv")


############ 방법 1과 2의 공통 결과 ##########
# 공통된 유전자 이름 찾기
common_genes <- intersect(rownames(deg_limma_filtered), rownames(deg_limma_pair_filtered))
# 공통된 유전자의 데이터프레임 생성
common_data_limma <- deg_limma_filtered[common_genes, ]
# 공통된 유전자 이름 출력
print(common_genes)
length(common_genes) # 83개

# 결과 파일로 저장
# write.csv(common_genes, file = "deg_limma_common_results.csv")
# write.csv(common_data_limma, file = "deg_limma_common.csv")

# ggplot2 패키지 로드
library(ggplot2)

# common gene의 volcano plot
ggplot(common_data_limma, aes(x = logFC, y = -log10(adj.P.Val))) +
  geom_point(aes(color = logFC > 0), size = 4) +
  scale_color_manual(values = c("blue", "red")) + 
  theme_minimal() +
  labs(title = "Volcano Plot of Common DEGs",
       x = "Log2 (Fold Change)",
       y = "-log10 (Adjusted P-Value)") +
  geom_hline(yintercept = -log10(0.05), color = "red", linetype = "dashed") + # p-value 기준선 추가
  theme(plot.title = element_text(hjust = 0.5))

  # 상위 10개 유전자를 adj.P.Val 값으로 정렬 후 선택
top_genes <- head(common_data_limma[order(common_data_limma$adj.P.Val), ], 30)

# Volcano plot에 상위 10개 유전자 이름을 라벨링하는 코드
ggplot(common_data_limma, aes(x = logFC, y = -log10(adj.P.Val))) +
  geom_point(aes(color = logFC > 0), size = 4) +
  scale_color_manual(values = c("blue", "red")) + 
  theme_minimal() +
  labs(title = "Volcano Plot of Common DEGs",
       x = "Log2 (Fold Change)",
       y = "-log10 (Adjusted P-Value)") +
  geom_hline(yintercept = -log10(0.05), color = "red", linetype = "dashed") + # p-value 기준선 추가
  theme(plot.title = element_text(hjust = 0.5)) +
    theme(
    plot.title = element_text(hjust = 0.5, size = 35),  # 제목 글씨 크기
    axis.title.x = element_text(size = 25),             # x축 제목 글씨 크기
    axis.title.y = element_text(size = 25),             # y축 제목 글씨 크기
    axis.text.x = element_text(size = 25),              # x축 숫자 크기
    axis.text.y = element_text(size = 25)               # y축 숫자 크기
  ) +
  # 상위 10개 유전자에 라벨 추가
  geom_text(data = top_genes, aes(label = rownames(top_genes)), 
            vjust = 1, hjust = 1.2, size = 8, color = "black")

# # 히트맵 필요한 패키지 설치
# if (!requireNamespace("pheatmap", quietly = TRUE)) {
#   install.packages("pheatmap")
# }
# library(pheatmap)

# # 히트맵에 사용할 데이터를 준비합니다.
# # common gene들의 발현 데이터를 준비합니다.
# # 데이터는 common_data_limma에서 logFC 값들로 사용한다고 가정합니다.
# heatmap_data <- common_data_limma[, "logFC", drop = FALSE]  # 열로 구성된 logFC 데이터 선택
# rownames(heatmap_data) <- rownames(common_data_limma)       # rownames 설정
# # 스케일링을 제거하고 히트맵을 다시 그립니다
# pheatmap(heatmap_data, 
#          cluster_rows = TRUE,    # 행 기준 클러스터링 여부
#          cluster_cols = FALSE,   # 열 기준 클러스터링 여부 (열이 하나이므로 비활성화)
#          scale = "none",         # 스케일링 제거
#          show_rownames = TRUE,   # 유전자 이름 표시
#          color = colorRampPalette(c("blue", "white", "red"))(100),  # 색상 팔레트 설정
#          main = "Heatmap of Common Genes (logFC)")  # 히트맵 제목
# # ggplot2를 사용한 막대 그래프
# library(ggplot2)

# # logFC 막대 그래프
# ggplot(heatmap_data, aes(x = rownames(heatmap_data), y = logFC)) +
#   geom_bar(stat = "identity", fill = "steelblue") +
#   theme_minimal() +
#   labs(title = "Bar Plot of Common Genes (logFC)", x = "Genes", y = "Log2 Fold Change") +
#   theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))  # 유전자 이름 회전

# 예시 데이터: DAVID에서 추출한 GO term 데이터 프레임 (예: df)
# df는 GO Term, PValue, Count 등의 열을 포함한다고 가정

go_bp <- read.table("/home/kim89/DAVID_BP.txt", header = TRUE, sep = "\t")
go_cc <- read.table("/home/kim89/DAVID_CC.txt", header = TRUE, sep = "\t")
go_mf <- read.table("/home/kim89/DAVID_MF.txt", header = TRUE, sep = "\t")
go_all <- rbind(go_bp, go_cc, go_mf)
# Bubble Plot (버블 플롯):
# 점 플롯과 유사하지만, 점의 크기와 색상을 동시에 활용하여 유전자 수와 P-value를 함께 시각화합니다.
ggplot(go_all, aes(x = Category, y = reorder(Term, -Count), size = Count, color = PValue)) + 
    geom_point(alpha = 0.7) + 
    theme_minimal() +
    labs(title = "Bubble Plot for GO Term Results",
         x = "Category",
         y = "GO Term") +
    scale_color_gradient(low = "blue", high = "red") +
    scale_size(range = c(3, 10)) +  # 버블 크기 조절
    theme(
    plot.title = element_text(hjust = 0.5, size = 20),  # 제목 글씨 크기
    axis.title.x = element_text(size = 15),             # x축 제목 글씨 크기
    axis.title.y = element_text(size = 15),             # y축 제목 글씨 크기
    axis.text.x = element_text(size = 15),              # x축 숫자 크기
    axis.text.y = element_text(size = 15)               # y축 숫자 크기
  )

# 범주별로 구분하는 열 추가
go_bp$Category <- "BP"
go_cc$Category <- "CC"
go_mf$Category <- "MF"

# P-value와 FDR 기준으로 필터링
filtered_go_bp <- go_bp[go_bp$PValue < 0.05 & go_bp$FDR < 0.05, ]
filtered_go_cc <- go_cc[go_cc$PValue < 0.05 & go_cc$FDR < 0.05, ]
filtered_go_mf <- go_mf[go_mf$PValue < 0.05 & go_mf$FDR < 0.05, ]
filtered_go_all <- rbind(filtered_go_bp, filtered_go_cc, filtered_go_mf)

# 필요한 라이브러리 설치 및 불러오기
# install.packages("ggplot2")
library(ggplot2)

# 데이터 병합

# 시각화
ggplot(filtered_go_all, aes(x = reorder(Term, -Count), y = Count, fill = PValue)) + 
    geom_bar(stat = "identity") + 
    coord_flip() +  # 수평 바 차트
    theme_minimal() +
    labs(title = "DAVID GO Term Results (BP, CC, MF)",
         x = "GO Term",
         y = "Gene Count") +
    facet_wrap(~ Category, scales = "free") +  # 범주별로 나눠서 표시
    scale_fill_gradient(low = "blue", high = "red")

# Dot Plot (점 그래프):
ggplot(filtered_go_all, aes(x = reorder(Term, -Count), y = Count, color = PValue, size = Count)) + 
    geom_point() + 
    coord_flip() +  # 수평으로 정렬
    theme_minimal() +
    labs(title = "Dot Plot for GO Term Results",
         x = "GO Term",
         y = "Gene Count") +
    scale_color_gradient(low = "blue", high = "red")

# Bubble Plot (버블 플롯):
# 점 플롯과 유사하지만, 점의 크기와 색상을 동시에 활용하여 유전자 수와 P-value를 함께 시각화합니다.
ggplot(filtered_go_all, aes(x = Category, y = reorder(Term, -Count), size = Count, color = PValue)) + 
    geom_point(alpha = 0.7) + 
    theme_minimal() +
    labs(title = "Bubble Plot for GO Term Results",
         x = "Category",
         y = "GO Term") +
    scale_color_gradient(low = "blue", high = "red") +
    scale_size(range = c(3, 10)) +  # 버블 크기 조절
    theme(
    plot.title = element_text(hjust = 0.5, size = 20),  # 제목 글씨 크기
    axis.title.x = element_text(size = 15),             # x축 제목 글씨 크기
    axis.title.y = element_text(size = 15),             # y축 제목 글씨 크기
    axis.text.x = element_text(size = 15),              # x축 숫자 크기
    axis.text.y = element_text(size = 15)               # y축 숫자 크기
  )

library(pheatmap)
# GO term과 유전자 수를 기반으로 히트맵 생성
gene_count_matrix <- as.matrix(filtered_go_all$Count)
rownames(gene_count_matrix) <- filtered_go_all$Term
pheatmap(gene_count_matrix, cluster_rows = TRUE, cluster_cols = FALSE)

library(clusterProfiler)
cnetplot(your_go_result, categorySize = "gene_count", foldChange = NULL)

# KEGG
kegg_data <- read.table("/home/kim89/KEGG_PATHWAY.txt", header = TRUE, sep = "\t")
library(ggplot2)

# 점 그래프 (Dot plot)
ggplot(kegg_data, aes(x = reorder(Term, -Count), y = Count, color = PValue, size = Count)) + 
  geom_point() + 
  coord_flip() +  # 수평 정렬
  theme_minimal() +
  labs(title = "KEGG Pathway Enrichment Results",
       x = "KEGG Pathway",
       y = "Gene Count") +
  scale_color_gradient(low = "blue", high = "red") +  # 색상 그라데이션
  scale_size(range = c(3, 10)) +  # 점 크기 조절
   theme(
    plot.title = element_text(hjust = 0.5, size = 20),  # 제목 글씨 크기
    axis.title.x = element_text(size = 15),             # x축 제목 글씨 크기
    axis.title.y = element_text(size = 15),             # y축 제목 글씨 크기
    axis.text.x = element_text(size = 15),              # x축 숫자 크기
    axis.text.y = element_text(size = 15)               # y축 숫자 크기
  )
