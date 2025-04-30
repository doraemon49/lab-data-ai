#install.packages("installr")
#library(installr)
#check.for.updates.R()
#install.R()
#updateR()
#.libPaths()
#.libPaths("C:/Program Files/R/R-4.4.1/library")
#install.packages("Matrix")

# Enter commands in R (or R studio, if installed)
#install.packages('Seurat')
library(Seurat) # scRNAseq
#install.packages('dplyr')
library(dplyr) # 데이터 집계/처리용
library(patchwork) # 그래프 그리기
#install.packages('ggplot2')
library(ggplot2)

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("MAST")
BiocManager::install("MAST", force = TRUE)
library(MAST)
# 잘 안 되면,, 수동 설치 #devtools::install_github("RGLab/MAST")
BiocManager::install("limma")
BiocManager::install("DESeq2")

# Replace with the actual path to your TPM data
tpm_data <- read.csv("C:\\Users\\LG\\Documents\\MJU\\연구실\\24.09.20_scRNA 데이터 분석\\GSE86982_smartseq.tpm.csv", row.names = 1, header = TRUE)

# Check the row names or metadata to identify stages
head(rownames(tpm_data))  # View the row names to check for stage information
# Check the column names to view the stage information
head(colnames(tpm_data))  # This should show identifiers related to D26, D54, and other stages

# "26" 또는 "54"가 포함된 열 이름을 찾는 경우
d26_d54_cells  <- grepl("^X26D|^X54D", colnames(tpm_data))
tpm_d26_d54 <- tpm_data[, d26_d54_cells]  # 해당 세포만 포함한 데이터로 서브셋 만들기

# Convert to a matrix, as Seurat expects a matrix format
# You can specify min.cells and min.features depending on your dataset's needs
# Seurat 객체 생성 (D26과 D54 세포만 포함)
library(Seurat)
seurat_object <- CreateSeuratObject(counts = as.matrix(tpm_d26_d54), project = "TPM_D26_D54", min.cells = 3, min.features = 200)


############### Seurat객체의 QC 전후, 바이올린 플랏 ### 퀄리티가 안 좋은 세포들은 버리자.
# 메타데이터 추가: nFeature_RNA, nCount_RNA, percent.mt 계산
# 미토콘드리아 유전자 비율 계산 (MT로 시작하는 유전자 이름 패턴 사용)
seurat_object[["percent.mt"]] <- PercentageFeatureSet(seurat_object, pattern = "^MT")
# 세포 이름에서 D26 또는 D54 라벨 추가 (메타데이터에 추가)
seurat_object$stage <- ifelse(grepl("26", colnames(tpm_d26_d54)), "D26", 
                              ifelse(grepl("54", colnames(tpm_d26_d54)), "D54", NA))
head(seurat_object$stage)
tail(seurat_object$stage)
# 그룹별 바이올린 플롯 그리기
VlnPlot(seurat_object, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), group.by = "stage", ncol = 3)

# 그룹별 바이올린 플롯 (범위 설정 포함)


VlnPlot(seurat_object, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), group.by = "stage", ncol = 3) +
  ylim(0, 10000) + # nFeature_RNA 범위
  ylim(0, 2000000) + # nCount_RNA 범위
  ylim(0, 5) # percent.mt 범위




# D26 세포만 필터링
d26_subset <- subset(seurat_object, subset = stage == "D26")
d26_subset$stage <- "D26"
# D26 세포에 대해 그룹화된 바이올린 플롯 그리기
VlnPlot(d26_subset, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), group.by = "stage", ncol = 3) +
  ylim(0, 10000) + # nFeature_RNA 범위 설정
  ylim(0, 2000000) + # nCount_RNA 범위 설정
  ylim(0,20) # percent.mt 범위 설정


# QC 기준 적용
seurat_object_qc <- subset(seurat_object, subset = nFeature_RNA > 200 & nFeature_RNA < 10000 & percent.mt < 5)


# QC 적용 후 바이올린 플롯 (y축 범위 설정)
VlnPlot(seurat_object_qc, features = c( "nCount_RNA"), group.by = "stage", ncol = 1)  + ylim(0, 2000000)



############ Preprocessing :
# 리보솜 유전자 비율(percent.rb) 계산
seurat_object[["percent.rb"]] <- PercentageFeatureSet(seurat_object, pattern = "^RPS|^RPL")
# 미토콘드리아와 리보솜 수가 20% 이상인 세포 필터링
seurat_object <- subset(seurat_object, subset = percent.mt < 20 & percent.rb < 20)
# 데이터 정규화 (LogNormalize 방식 사용)
seurat_object <- NormalizeData(seurat_object, normalization.method = "LogNormalize", scale.factor = 10000)
# 가변적 특성(변동성이 큰 유전자) 찾기
seurat_object <- FindVariableFeatures(seurat_object, selection.method = "vst", nfeatures = 2000)
# 데이터 스케일링
seurat_object <- ScaleData(seurat_object)


#############MAST, limma, Deseq2
library(MAST)
# D26과 D54 세포에 대한 정확한 조건 설정
#seurat_object$stage <- ifelse(grepl("26", colnames(tpm_d26_d54)), "D26", 
#                              ifelse(grepl("54", colnames(tpm_d26_d54)), "D54", NA))
# stage 메타데이터의 값을 확인
head(seurat_object$stage)
tail(seurat_object$stage)
# stage 메타데이터를 ident으로 설정
Idents(seurat_object) <- seurat_object$stage
# 현재 정체성 확인
levels(Idents(seurat_object))

# MAST를 사용한 D26 vs D54 차등 발현 유전자(DEGs) 분석
degs_mast <- FindMarkers(seurat_object, ident.1 = "D26", ident.2 = "D54", test.use = "MAST")

# MAST 결과 상위 몇 개 유전자 확인
head(degs_mast)
# FDR 보정 p-값이 0.05 미만이고 log2 폴드 변화가 절대값 1.5보다 큰 DEG 필터링
degs_mast_filtered <- degs_mast[degs_mast$p_val_adj < 0.05 & abs(degs_mast$avg_log2FC) > 1.5, ]
head(degs_mast_filtered)

# 필터링된 DEG 개수 확인 # 849개
nrow(degs_mast_filtered)



# limma :
# First, limma 분석을 위한 raw counts 데이터 추출
counts_data <- GetAssayData(seurat_object, layer = "counts")

# Create a design matrix based on your cell groups (D26 vs D54)
group <- seurat_object$stage
design <- model.matrix(~ group)
colnames(design)
# Apply voom transformation and limma analysis
library(limma)
voom_data <- voom(counts_data, design)
fit <- lmFit(voom_data, design)
fit <- eBayes(fit)
degs_limma <- topTable(fit, number = Inf)

# 차등 발현 유전자의 개수 확인 # 19347
nrow(degs_limma)
# 상위 6개의 DEG 결과를 확인
head(degs_limma)
# FDR < 0.05 및 절대 log2 폴드 변화 > 1.5인 DEG 필터링
degs_limma_filtered <- degs_limma[degs_limma$adj.P.Val < 0.05 & abs(degs_limma$logFC) > 1.5, ]
nrow(degs_limma_filtered)  # 필터링된 DEG 개수 확인 ## 449
head(degs_limma_filtered)   # 필터링된 상위 DEGs 확인

# Deseq2 :
# Convert the Seurat object to a DESeq2 dataset
# 필요한 패키지 로드
if (!requireNamespace("DESeq2", quietly = TRUE)) {
  install.packages("DESeq2")
}
library(DESeq2)

# Seurat 객체에서 count 데이터 추출 (유전자 x 세포)
# 데이터를 정수형으로 변환
counts_data <- round(as.matrix(GetAssayData(seurat_object, layer = "counts")))

# 세포(stage) 정보 가져오기
meta_data <- data.frame(stage = factor(seurat_object$stage))

# DESeq2 객체 생성
dds <- DESeqDataSetFromMatrix(countData = counts_data, colData = meta_data, design = ~ stage)

# DESeq2 차등 발현 분석 실행
dds <- DESeq(dds)

# D26과 D54 간의 차등 발현 유전자 추출
degs_deseq2 <- results(dds, contrast = c("stage", "D26", "D54"))

# FDR < 0.05 및 절대 log2 폴드 변화 > 1.5로 필터링
# NA 값을 제거하고 FDR < 0.05 및 절대 log2 폴드 변화 > 1.5로 필터링
degs_deseq2_filtered <- degs_deseq2[!is.na(degs_deseq2$padj) & degs_deseq2$padj < 0.05 & 
                                      !is.na(degs_deseq2$log2FoldChange) & abs(degs_deseq2$log2FoldChange) > 1.5, ]

# 필터링된 DEG 확인
head(degs_deseq2_filtered)

# 필터링된 DEG 개수 확인 # 1299개
nrow(degs_deseq2_filtered)

###########
# MAST, limma, DESeq2에서 추출한 DEG의 rownames (유전자 이름)을 각각 가져옵니다.
degs_mast_genes <- rownames(degs_mast_filtered)  # MAST에서 필터링된 DEG
degs_limma_genes <- rownames(degs_limma_filtered)  # limma에서 필터링된 DEG
degs_deseq2_genes <- rownames(degs_deseq2_filtered)  # DESeq2에서 필터링된 DEG

# 세 가지 방법에서 공통된 DEG 찾기
common_degs <- Reduce(intersect, list(degs_mast_genes, degs_limma_genes, degs_deseq2_genes))

# 공통 DEG 개수 확인 # 184개개
length(common_degs)

# 공통된 DEGs에 대한 정보를 가져오기
common_degs_data <- degs_deseq2_filtered[common_degs, ]  # 여기서는 DESeq2 결과를 사용

# 결과 확인
head(common_degs_data)

# 필요한 패키지 로드
library(ggplot2)

# Volcano plot 그리기
ggplot(common_degs_data, aes(x = log2FoldChange, y = -log10(padj))) +
  geom_point(aes(color = log2FoldChange > 0), size = 1.5) +
  scale_color_manual(values = c("blue", "red")) +  # 다운레귤레이션(blue), 업레귤레이션(red)
  theme_minimal() +
  labs(title = "Volcano Plot of Common DEGs",
       x = "Log2 (Fold Change)",
       y = "-log10 (Adjusted P-Value)") +
  theme(plot.title = element_text(hjust = 0.5))

# 업레귤레이션된 유전자
up_regulated <- sum(common_degs_data$log2FoldChange > 0)

# 다운레귤레이션된 유전자
down_regulated <- sum(common_degs_data$log2FoldChange < 0)

cat("Up-regulated genes:", up_regulated, "\n")
cat("Down-regulated genes:", down_regulated, "\n")

#### + 라벨링
# 상위의 몇 개 유전자에 라벨 표시 (padj < 0.05 및 log2FoldChange > 1.5 기준 상위 유전자)
top_genes <- head(common_degs_data[order(common_degs_data$padj), ], 30)  # 상위 10개 유전자 선택

# 유전자 상태 (업레귤레이션, 다운레귤레이션, 비유의미) 분류
common_degs_data$significance <- ifelse(common_degs_data$padj < 0.05 & abs(common_degs_data$log2FoldChange) > 1.5, 
                                        ifelse(common_degs_data$log2FoldChange > 0, "Up", "Down"), "Not Sig")
# ggrepel 패키지 설치
if (!requireNamespace("ggrepel", quietly = TRUE)) {
  install.packages("ggrepel")
}

# 패키지 로드
library(ggrepel)

# Volcano plot에 유전자 라벨 추가
ggplot(common_degs_data, aes(x = log2FoldChange, y = -log10(padj))) +
  geom_point(aes(color = significance), size = 1.5) +
  scale_color_manual(values = c("Not Sig" = "grey", "Down" = "blue", "Up" = "red")) +
  geom_text_repel(data = top_genes, aes(label = rownames(top_genes)), size = 3, max.overlaps = 10) +  # 라벨 추가
  theme_minimal() +
  labs(title = "Volcano Plot of Common DEGs",
       x = "Log2 (Fold Change)",
       y = "-log10 (Adjusted P-Value)") +
  theme(plot.title = element_text(hjust = 0.5))

############ Hierarchical clustering
# 필요한 패키지 설치 및 로드
if (!requireNamespace("pheatmap", quietly = TRUE)) {
  install.packages("pheatmap")
}
library(pheatmap)

# 공통된 DEGs에 대한 발현 데이터 추출 (D26과 D54 세포만)
expression_data <- counts_data[common_degs, ]  # 공통 DEGs의 발현 데이터

# Seurat 객체에서 D26과 D54 세포의 메타데이터(stage 정보)만 추출
stage_data <- seurat_object$stage

# D26과 D54 시점의 세포들만 필터링
d26_d54_cells <- which(stage_data %in% c("D26", "D54"))
expression_data <- expression_data[, d26_d54_cells]

# Z-score 정규화: 유전자별로 표준화를 수행하여 비교할 수 있도록 함
expression_data <- t(scale(t(expression_data))) ## 대부분 0으로 가버려서, 결과 좋지 못함
# 발현 데이터 로그 변환 (정수형 값에서 변환해야 합니다)
# expression_data_log <- log2(expression_data + 1)

# stage 정보를 다시 저장
cell_stage <- stage_data[d26_d54_cells]


# 히트맵 그리기 (로그 변환된 데이터 사용)
pheatmap(expression_data_zScore, 
         annotation_col = data.frame(Stage = cell_stage),  # D26과 D54 시점 표시
         clustering_method = "ward.D2", 
         color = colorRampPalette(c("blue", "white", "red"))(100),  # 색상 팔레트 설정
         show_rownames = FALSE, 
         show_colnames = FALSE, 
         main = "Hierarchical Clustering of Common DEGs (Log Transformed)")

# 유전자별 regulation 정보 추가 (Up/Down 설정)
common_degs_data$regulation <- ifelse(common_degs_data$log2FoldChange > 0, "Up", "Down")

# regulation 정보를 annotation_row로 추가하기 위해 rownames 설정
annotation_row <- data.frame(Regulation = common_degs_data$regulation)
rownames(annotation_row) <- rownames(common_degs_data)

# 히트맵 그리기 (로그 변환된 데이터 사용, regulation 정보 추가)
pheatmap(expression_data_log, 
         annotation_col = data.frame(Stage = cell_stage),  # D26과 D54 시점 표시
         annotation_row = annotation_row,  # Up/Down Regulation 정보 추가
         clustering_method = "ward.D2", 
         color = colorRampPalette(c("blue", "white", "red"))(100),  # 색상 팔레트 설정
         show_rownames = FALSE, 
         show_colnames = FALSE, 
         main = "Hierarchical Clustering of Common DEGs with Regulation")

# 히트맵 그리기 (공통 DEGs의 발현 패턴 시각화)
# 히트맵의 색상 팔레트 조정
pheatmap(expression_data, 
         annotation_col = data.frame(Stage = cell_stage),  # D26과 D54 시점 표시
         clustering_method = "ward.D2", 
         scale = "row", 
         color = colorRampPalette(c("blue", "white", "red"))(100),  # 색상 팔레트 설정
         show_rownames = FALSE, 
         show_colnames = FALSE, 
         main = "Hierarchical Clustering of Common DEGs")

# 클러스터링 방법을 변경해 시도 (예: complete linkage)
pheatmap(expression_data, 
         annotation_col = data.frame(Stage = cell_stage),
         clustering_method = "complete",  # 다른 클러스터링 방법 시도
         scale = "row", 
         color = colorRampPalette(c("blue", "white", "red"))(100),
         show_rownames = FALSE, 
         show_colnames = FALSE, 
         main = "Hierarchical Clustering of Common DEGs")

# 클러스터링 방법을 변경해 시도 (예: average)
pheatmap(expression_data, 
         annotation_col = data.frame(Stage = cell_stage),
         clustering_method = "average",  # 다른 클러스터링 방법 시도
         scale = "row", 
         color = colorRampPalette(c("blue", "white", "red"))(100),
         show_rownames = FALSE, 
         show_colnames = FALSE, 
         main = "Hierarchical Clustering of Common DEGs")


################그냥 그려본 것들.
##### Output the common DEGs
common_degs <- data.frame(
  MAST = degs_mast[common_genes, ],
  limma = degs_limma[common_genes, ],
  DESeq2 = degs_deseq2[common_genes, ]
)

# View the common DEGs
head(common_degs)

# UMAP plot
seurat_object <- RunUMAP(seurat_object, dims = 1:10)
DimPlot(seurat_object, reduction = "umap", group.by = "stage")

# Heatmap of DEGs
DoHeatmap(seurat_object, features = common_genes)


############## 추가로 시도해본 것들

# Run Principal Component Analysis (PCA):
seurat_object <- RunPCA(seurat_object, features = VariableFeatures(object = seurat_object))

# Cluster the Cells:
seurat_object <- FindNeighbors(seurat_object, dims = 1:10)
seurat_object <- FindClusters(seurat_object, resolution = 0.5)

# Visualize the Data Using UMAP or t-SNE:
# UMAP
seurat_object <- RunUMAP(seurat_object, dims = 1:10)
DimPlot(seurat_object, reduction = "umap")
# t-SNE (optional)
seurat_object <- RunTSNE(seurat_object, dims = 1:10)
DimPlot(seurat_object, reduction = "tsne")

