rm(list = ls())
library(rgl)

# Declaração dataset
N<-60
sd1<-0.4
sd2<-0.8
xc1<-matrix(rnorm(2*N, sd = sd1), ncol = 2, nrow = 30)+2
xc2<-matrix(rnorm(2*N, sd = sd2), ncol = 2, nrow = 30)+4
plot(xc1[,1], xc1[,2], xlim = c(0,6), ylim = c(0,6), xlab='', ylab='', col='red')
par(new=TRUE)
plot(xc2[,1], xc2[,2], xlim = c(0,6), ylim = c(0,6), xlab='', ylab='', col='blue')

# Nomenclatura: mean_(numero da variável, classe)
# Calculo médias e desvio padrão
m11<-mean(xc1[,1])
m12<-mean(xc2[,1])
m21<-mean(xc1[,2])
m22<-mean(xc2[,2])

s11<-sd(xc1[,1])
s12<-sd(xc2[,1])
s21<-sd(xc1[,2])
s22<-sd(xc2[,2])

# Normal function
# x -> input, m -> mean, r -> standart deviation
fnormal1var<-function(x,m,r)
{
  y<-(1/(sqrt(2*pi*r*r)))*exp(-0.5* ((x-m)/(r))^2)
  return(y)
}

# Calculo das gaussianas
xrange<-seq(0,6,0.05)
f11<-fnormal1var(xrange, m11, s11)
f12<-fnormal1var(xrange, m12, s12)
f21<-fnormal1var(xrange, m21, s21)
f22<-fnormal1var(xrange, m22, s22)

par(new=TRUE)
plot(xrange, f11, type='l', xlim = c(0,6), ylim = c(0,6), xlab='', ylab='', col='red')
par(new=TRUE)
plot(xrange, f12, type='l', xlim = c(0,6), ylim = c(0,6), xlab='', ylab='', col='blue')
par(new=TRUE)
plot(f21, xrange, type='l', xlim = c(0,6), ylim = c(0,6), xlab='', ylab='', col='red')
par(new=TRUE)
plot(f22, xrange, type='l', xlim = c(0,6), ylim = c(0,6), xlab='', ylab='', col='blue')

pc1 <- length(xc1)/(length(xc1) + length(xc2))
pc2 <- length(xc2)/(length(xc1) + length(xc2))

lseq<-length(xrange)
M1<-matrix(nrow=lseq, ncol=lseq)
M2<-matrix(nrow=lseq, ncol=lseq)

sup <- matrix(nrow=lseq, ncol=lseq)

for(i in 1:lseq){
  for(j in 1:lseq){
    M1[i,j]<-f11[i]*f21[j]
    M2[i,j]<-f12[i]*f22[j]
    sup[i,j] <- sign(M1[i,j] - M2[i,j])
  }
}

par(new=TRUE)
contour(xrange, xrange, sup, nlevels=1, xlim=c(0,6), ylim=c(0,6), col = 'purple')

persp3d(xrange, xrange, M1, xlim=c(-2,10), ylim=c(-2,10), xlab="x1", ylab="x2", zlab="(x1,x2)", theta = 60, phi = 40)
par(new=TRUE)
persp3d(xrange, xrange, M2, xlim=c(-2,10), ylim=c(-2,10), xlab="x1", ylab="x2", zlab="", theta = 60, phi = 40)
par(new=TRUE)
persp3d(xrange, xrange, sup, xlim=c(-2,10), ylim=c(-2,10), xlab="x1", ylab="x2", zlab="", theta = 60, phi = 40, col='yellow')
















