rm(list = ls())
library(rgl)

fnormal1var<-function(x,m,r) # eq 18 das notas de aula
{
  y<-(1/(sqrt(2*pi*r*r)))*exp(-0.5 * ((x-m)/(r))^2)
  return(y)
}


pxkde1var<-function(xrange,X,h)
{
  knorm<-function(u,h)
  {
    K <- (1/sqrt(2*pi*h*h)) * exp(-0.5*u^2)
    return(K)
  }
  #########################
  N<-length(X)
  px<-matrix(nrow=length(xrange),ncol=1)
  for (i in 1:length(xrange))
  {
    xi<-xrange[i]
    kxixall<-0
    for (j in 1:N)
    {
      xj<-X[j]
      u<-(sum(xi-xj))/h
      kxixall<-kxixall+knorm(u,h)
    }
    px[i]<-kxixall
  }
  px<-px/(N)
  return(px)
} 

pxkdenvar<-function(xrange,X,h)
{
  knorm<-function(u,h)
  {
    K <- (1/sqrt(2*pi*h*h)) * exp(-0.5*u^2)
    return(K)
  }
  N<-dim(X)[1]
  Nxrange<-dim(xrange)[1]
  nxrange<-dim(xrange)[2]
  px<-matrix(nrow=Nxrange,ncol=1)
  
  for (i in 1:Nxrange)
  {
    if (nxrange >= 2)
    {
      xi<-xrange[i,]
    }
    else
    {
      xi<-xrange[i]
    }
    kxixall<-0
    for (j in 1:N)
    {
      xj<-X[j,]
      u<-(sqrt(sum((xi-xj)^2)))/h
      kxixall<-kxixall+knorm(u,h)
    }
    px[i]<-kxixall
  }
  px<-px/(N)
  return(px)
} 
# N1<-120
# sd1<-0.4
# xc1<-rnorm(N1, 2, sd=sd1)
# 
# N2<-120
# sd2<-0.3
# xc2<-rnorm(N1, 5, sd=sd2)
# par(new=TRUE)
# 
# plot(xc1, rep(0,N2), col = 'black', xlim=c(0,6), ylim=c(0,1))
# 
# # Silverman 
# h<-1.06*sd(xc1)*length(xc1)^(-1/5)
# X<-mean(xc1)
# 
# # Normal distribution
# xrange <- seq(0, 6, length.out = 100)
# 
# # Normal x KDE 
# px1 <- fnormal1var(xrange, X, sd1)
# px1kdenvar <- pxkdenvar(as.matrix(xrange), as.matrix(xc1), h)
# 
# lines(xrange, px1, col='red', lwd = 1)
# lines(xrange, px1kdenvar, col = 'blue', lwd = 1)
# 
# 
# # For non-normal distribution
# xc3<-rbind(as.matrix(xc1),as.matrix(xc2))
# plot(xc3, rep(0,N1+N2), col = 'black', xlim=c(0,7), ylim=c(0,1))
# xrange <- seq(0, 7, length.out = 100)
# 
# # Normal x KDE
# px3normal <- fnormal1var(xrange, mean(xc3), sd(xc3))
# px3kdenvar <- pxkdenvar(as.matrix(xrange), xc3, h)
# 
# lines(xrange, px3normal, col='blue', lwd = 1)
# lines(xrange, px3kdenvar, col='red', lwd = 1)
# 
# # Multimodal
# N<-30
# m1<-c(2,2)
# m2<-c(4,4)
# m3<-c(2,4)
# m4<-c(4,2)
# 
# g1<-matrix(rnorm(N*2,sd=0.5), nrow = N, ncol=2) + matrix(m1, nrow=N, ncol=2, byrow=T)
# g2<-matrix(rnorm(N*2,sd=0.5), nrow = N, ncol=2) + matrix(m2, nrow=N, ncol=2, byrow=T)
# g3<-matrix(rnorm(N*2,sd=0.5), nrow = N, ncol=2) + matrix(m3, nrow=N, ncol=2, byrow=T)
# g4<-matrix(rnorm(N*2,sd=0.5), nrow = N, ncol=2) + matrix(m4, nrow=N, ncol=2, byrow=T)
# 
# xc <- rbind(g1,g2,g3,g4)
# 
# 
# # KDE parameters
# seqx1x2<-seq(0,6,0.1)
# lseq<-length(seqx1x2)
# MZ<- matrix(nrow=lseq, ncol=lseq)
# h<-0.5
# 
# cr <- 0
# for(i in 1:lseq){
#   for(j in 1:lseq){
#     cr <- cr + 1
#     x1<-seqx1x2[i]
#     x2<-seqx1x2[j]
#     x1x2<-as.matrix(cbind(x1,x2))
#     MZ[i,j]<-pxkdenvar(x1x2, xc, h)
#   }
# }
# 
# contour(seqx1x2, seqx1x2, MZ, xlim=c(0,6), ylim = c(0,6), xlab = "x1", ylab = "x2")
# par(new=TRUE)
# plot(xc[,1], xc[,2], col='red', xlim=c(0,6), ylim=c(0,6))
# persp3d(seqx1x2, seqx1x2, MZ, xlim = c(0,5), ylim = c(0,6), xlab = 'x1', ylab=x2, col='red')
# 
# 
# ################################################################# 2/12 ###########
# dev.off()
# xc1<-rbind(g1,g2)
# xc2<-rbind(g3,g4)
# xall<-rbind(xc1, xc2)
# 
# dev.new()
# plot(xc1[,1], xc1[,2], col = 'red', xlim = c(0,6), ylim=c(0,6))
# par(new=TRUE)
# plot(xc2[,1], xc2[,2], col = 'blue', xlim = c(0,6), ylim=c(0,6))
# 
# matD<-as.matrix(dist(xall, upper = T, diag = T))
# dev.new()
# image(matD)
# 
# seqh<-seq(0.0001, 5, 0.1)
# nseq<-length(seqh)
# 
# #Animação variação de h 
# for(i in 1:nseq){
#   h<-seqh[i]
#   matK<-exp(-(matD*matD)/h^2)
#   #image(matK)
#   #pxall<-colSums(matK)/(4*N*(sqrt(2*pi*h))^2)
#   
#   # 4 submatrizes de verossimilhança resultantes de matk
#   K11<-matK[(1:60),(1:60)] # Amostra da classe 1 em relação a classe 1
#   K12<-matK[(1:60),(61:120)] # amostra da classe 1 em relação a classe 2
#   K22<-matK[(61:120),(61:120)] # amostra da classe 2 em relação a classe 2
#   K21<-matK[(61:120),(1:60)] # amostra da classe 2 em relação a classe 1
#   
#   pxc1_c1 <- colSums(K11)/(4*N*(sqrt(2*pi*h))^2)
#   pxc1_c2 <- colSums(K12)/(4*N*(sqrt(2*pi*h))^2)
#   pxc2_c2 <- colSums(K22)/(4*N*(sqrt(2*pi*h))^2)
#   pxc2_c1 <- colSums(K21)/(4*N*(sqrt(2*pi*h))^2)
#   
#   #dev.new()
#   plot(pxc1_c1, pxc1_c2, xlim = c(0,0.04), ylim=c(0,0.04), col = 'red') 
#   par(new=T)
#   plot(pxc2_c1, pxc2_c2, xlim = c(0,0.04), ylim=c(0,0.04), col = 'blue')
#   Sys.sleep(0.2)
# }
# 
# h<-0.5
# matK<-exp(-(matD*matD)/h^2)
# #image(matK)
# pxall<-colSums(matK)/(4*N*(sqrt(2*pi*h))^2)
# 
# # 4 submatrizes de verossimilhança resultantes de matk
# K11<-matK[(1:60),(1:60)] # Amostra da classe 1 em relação a classe 1
# K12<-matK[(1:60),(61:120)] # amostra da classe 1 em relação a classe 2
# K22<-matK[(61:120),(61:120)] # amostra da classe 2 em relação a classe 2
# K21<-matK[(61:120),(1:60)] # amostra da classe 2 em relação a classe 1
# 
# pxc1_c1 <- colSums(K11)/(4*N*(sqrt(2*pi*h))^2)
# pxc1_c2 <- colSums(K12)/(4*N*(sqrt(2*pi*h))^2)
# pxc2_c2 <- colSums(K22)/(4*N*(sqrt(2*pi*h))^2)
# pxc2_c1 <- colSums(K21)/(4*N*(sqrt(2*pi*h))^2)
# 
# #dev.new()
# #plot(pxc1_c1, pxc1_c2, xlim = c(0,0.04), ylim=c(0,0.04), col = 'red') 
# #par(new=T)
# #plot(pxc2_c1, pxc2_c2, xlim = c(0,0.04), ylim=c(0,0.04), col = 'blue')
# 
# yall<-rbind(matrix(-1, nrow = 60, ncol = 1), matrix(1, nrow = 60, ncol = 1))
# maty<-yall %*% t(yall)
# image(maty)
# matyyK<- maty * matK
# plot(colSums(matyyK))
# #lines(0)














