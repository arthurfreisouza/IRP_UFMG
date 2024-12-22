rm(list = ls())

pxkdenvar <- function(xrange, X, h) {
  knorm <- function(u, h) {
    K <- (1 / sqrt(2 * pi * h * h)) * exp(-0.5 * u^2)
    return(K)
  }
  
  N <- dim(X)[1]
  Nxrange <- dim(xrange)[1]
  nxrange <- dim(xrange)[2]
  px <- matrix(nrow = Nxrange, ncol = 1)
  
  for (i in 1:Nxrange) {
    if (nxrange >= 2) {
      xi <- xrange[i, ]
    } else {
      xi <- xrange[i]
    }
    kxixall <- 0
    for (j in 1:N) {
      xj <- X[j, ]
      u <- (sqrt(sum((xi - xj)^2))) / h
      kxixall <- kxixall + knorm(u, h)
    }
    px[i] <- kxixall
  }
  px <- px / N
  return(px)
}

# Variables and data generation
n <- 2
N1 <- 100
N2 <- 100

xc1 <- matrix(rnorm(N1 * n, sd = 0.8), nrow = N1, ncol = n) + matrix(c(2, 2), nrow = N1, ncol = n)
xc2 <- matrix(rnorm(N2 * n, sd = 0.8), nrow = N2, ncol = n) + matrix(c(4, 4), nrow = N2, ncol = n)

# Plot the data
plot(xc1[, 1], xc1[, 2], col = 'blue', xlim = c(0, 6), ylim = c(0, 6))
par(new = TRUE)
plot(xc2[, 1], xc2[, 2], col = 'red', xlim = c(0, 6), ylim = c(0, 6))

# Combine data
x <- rbind(xc1, xc2)
y <- rbind(matrix(-1, nrow = N1, ncol = 1), matrix(1, nrow = N2, ncol = 1))

# Debug check
print("Before source: y exists?")
print(exists("y"))

# Check if the source file clears the environment
# source("/home/arthur/Desktop/IRP/KDE/KDE.R")

# Check if `y` still exists after running source()
print("After source: y exists?")
print(exists("y"))

px <- matrix(nrow = nrow(x), ncol = 1)
h <- 0.1
YY <- y %*% t(y)  
d <- dist(x, upper = TRUE, diag = TRUE)
k <- exp((-d * d) / h^2)

kd <- k * YY
PI <- rowSums(kd)
plot(PI)

# Loop to compute nneg for multiple h values
seqh <- seq(0.01, 10, 0.01)
nneg <- matrix(nrow = length(seqh), ncol = 1)

ch <- 1
for (h in seqh) {
  YY <- y %*% t(y)  
  d <- as.matrix(dist(x, upper = TRUE, diag = TRUE))
  k <- exp((-d * d) / h^2)
  kd <- k * YY
  PI <- rowSums(kd)
  nneg[ch] <- sum(1 * (PI < 0))
  ch <- ch + 1
}

# Plot nneg vs seqh as a line
plot(seqh, nneg, type = "l", col = "blue", xlim = c(0, 1), ylim = c(0, max(nneg)), 
     xlab = "h", ylab = "nneg", main = "Line Plot of nneg vs. h")

# Updated h values
seqh <- seq(0.1, 0.4, 0.1)
meanpxi <- matrix(nrow = length(seqh), ncol = 1)

for (i in 1:length(seqh)) {
  ch <- 1  # Reset ch for each h
  h <- seqh[i]
  YY <- y %*% t(y)  
  d <- as.matrix(dist(x, upper = TRUE, diag = TRUE))
  k <- (1 / ((N1 * sqrt(2 * pi) * h)^2)) * exp((-d * d) / h^2)
  k11 <- k[1:N1, 1:N1]
  pxi <- rowSums(k11)
  meanpxi[i] <- mean(pxi)
  ch <- ch + 1
}

plot(seqh, meanpxi, type = 'l')

# 3D plotting setup
seqx1x2 <- seq(0, 6, 0.1)
npgrid <- length(seqx1x2)
M <- matrix(nrow = npgrid, ncol = npgrid)
h <- 0.05

# Generate grid for the 3D plot
seqx1x2 <- seq(0, 6, 0.1)
npgrid <- length(seqx1x2)
MZ <- matrix(nrow = npgrid, ncol = npgrid)
h <- 0.05  # Set bandwidth

for (i in 1:npgrid) {
  for (j in 1:npgrid) {
    x1 <- seqx1x2[i]
    x2 <- seqx1x2[j]
    x1x2 <- as.matrix(cbind(x1, x2))
    MZ[i, j] <- pxkdenvar(x1x2, xc1, h)  # Kernel density estimate for each grid point
  }
}

plot(xc1[, 1], xc1[, 2], col = 'blue', xlim = c(0, 6), ylim = c(0, 6))
par(new = TRUE)
plot(xc2[, 1], xc2[, 2], col = 'red', xlim = c(0, 6), ylim = c(0, 6))

contour(seqx1x2, seqx1x2, MZ, xlim = c(0,6), ylim = c(0,6))
