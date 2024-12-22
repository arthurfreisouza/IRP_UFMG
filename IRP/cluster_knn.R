rm(list =ls())
library("mlbench")
#install.packages("Clustering")

data("BreastCancer")

Bs <- na.omit(BreastCancer)
xall <- data.matrix(Bs[, (2:10)])
yall <- 1*data.matrix(Bs[, 11] == "benign")

xallord <- xall[order(yall), ]
yallord <- xall[order(yall), ]
mat_dist <- as.matrix(dist(xallord, diag = T, upper = T))

image(mat_dist)
rows <- which(is.na(xall))
print(rows)


retkm <- kmeans(x = xall, centers = 2)
rotulos <- retkm$cluster
cbind(rotulos, yallord)

sil <- silhouette(retkm$clustering, dist(bs, ""))
#sil <- silhouette(retkm$clustering, dist(bs))
windows() # RStudio sometimes does not display silhouette plots correctly
plot(sil)


