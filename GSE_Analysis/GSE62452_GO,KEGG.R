# DESeq2 패키지 설치 (설치되어 있지 않은 경우)
if (!requireNamespace("DESeq2", quietly = TRUE)) {
  install.packages("BiocManager")
  BiocManager::install("DESeq2")
}



# Install necessary packages if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

#BiocManager::install('clusterProfiler') # 설치완료
#BiocManager::install("org.Hs.eg.db") # 설치 완료료

# Load the required libraries
library(clusterProfiler)
library(org.Hs.eg.db)

common_data_limma <- read.csv("C:\\Users\\LG\\Documents\\MJU\\연구실\\24.09.27_GSE62452\\deg_limma_common.csv", row.names = 1, header = TRUE)

# Extract gene symbols from your dataset `common_data_limma`
gene_symbols <- rownames(common_data_limma)

# Convert gene symbols to ENTREZ IDs using org.Hs.eg.db
eg <- bitr(gene_symbols, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)

# Save the conversion result to a CSV file
write.csv(eg, file = "C:\\Users\\LG\\Documents\\MJU\\연구실\\24.09.27_GSE62452\\common_deg_limma_entrez.csv")


# Display the result
head(eg)

# DAVID로 가서 결과 받아오자.

# DAVID에서 결과 받아옴
go_bp <- read.table("C:\\Users\\LG\\Documents\\MJU\\연구실\\24.09.27_GSE62452\\GO, KEGG\\DAVID_BP.txt", header = TRUE, sep = "\t")
go_cc <- read.table("C:\\Users\\LG\\Documents\\MJU\\연구실\\24.09.27_GSE62452\\GO, KEGG\\DAVID_CC.txt", header = TRUE, sep = "\t")
go_mf <- read.table("C:\\Users\\LG\\Documents\\MJU\\연구실\\24.09.27_GSE62452\\GO, KEGG\\DAVID_MF.txt", header = TRUE, sep = "\t")


# 범주별로 구분하는 열 추가
go_bp$Category <- "BP"
go_cc$Category <- "CC"
go_mf$Category <- "MF"

# P-value와 FDR 기준으로 필터링
filtered_go_bp <- go_bp[go_bp$PValue < 0.05 & go_bp$FDR < 0.05, ]
filtered_go_cc <- go_cc[go_cc$PValue < 0.05 & go_cc$FDR < 0.05, ]
filtered_go_mf <- go_mf[go_mf$PValue < 0.05 & go_mf$FDR < 0.05, ]
filtered_go_all <- rbind(filtered_go_bp, filtered_go_cc, filtered_go_mf)

library(clusterProfiler)

# GO 분석 수행 (BP는 Biological Process)
go_enrich <- enrichGO(gene = eg$ENTREZID, 
                      OrgDb = org.Hs.eg.db, 
                      keyType = "ENTREZID", 
                      ont = "BP", 
                      pvalueCutoff = 0.05)

# 네트워크 플롯 그리기
cnetplot(go_enrich, categorySize = "gene_count", color.params = list(foldChange = NULL))
