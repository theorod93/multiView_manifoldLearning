#' In this script, we will create a function to simulate multi-view data
#' and test the performance of Multi-view Visualisation methods, namely:
#'

# Directory
# setwd("")

# Load image

# Libraries
library(MASS)
library(clusterGeneration)
library(tsne)
# Check README.txt

# Define function:
non_linear_mView_vis_simulation <- function(M, C, mean_matrix, p, num_samples){
  X <- vector("list", length = M)
  power_values <- c()
  div_values <- c()
  sum_values <- c()
  for (m in 1:M){
    # Simulate polynomial coefficients for non-linear functions
    power_val <- round(runif(min = 3, max = 5,1))
    div_val <- round(runif(min = 3, max = 50,1))
    sum_val <- round(runif(min = 4, max = 10,1))
    power_values <- c(power_values, power_val)
    div_values <- c(div_values, div_val)
    sum_values <- c(sum_values, sum_val)
    # Introduce view
    X[[m]] <- matrix(ncol = p[m])
    # Run for each cluster
    for (c in 1:C){
      Xtemp <- matrix(nrow = num_samples, ncol = p[m])
      Xinp <- matrix(nrow = num_samples, ncol = p[m])
      for (i in 1:p[m]){
        Xtemp[,i] <- runif(n = num_samples, min = 0, max = mean_matrix[m,c])

        #Generate y as a*e^(bx)+c

#        Xinp[,i] <- runif(1,0,20)*exp(Xtemp[,i])+runif(n=num_samples,0,5)
#        Xtemp[,i] <- rnorm(n = num_samples, mean = mean_matrix[m,c])
        Xinp[,i] <- (Xtemp[,i]+sum_val)^power_val/div_val + rnorm(n=num_samples)
      }
      X[[m]] <- rbind(X[[m]], Xinp)
    }
    X[[m]] <- X[[m]][-1,] #  Remove first instance
  }
  # Labels
  ColLabels<- c()
  for (c in 1:C){
    ColLabels <- c(ColLabels, rep(c, num_samples))
  }
  return(list("labels" = ColLabels, "X" = X, "power_values" = power_values,
              "sum_values" = sum_values, "div_values" = div_values))
}

# Scenario (B)
M = 4
C = 3
mean_matrix <- matrix(nrow = M, ncol = C)
mean_matrix[1,] <- c(1,1,2)
mean_matrix[2,] <- c(1,2,1)
mean_matrix[3,] <- c(2,1,1)
mean_matrix[4,] <- c(1,1,1)
p = c(100,100,100,1000)
num_samples = 100
simData_scenB <- non_linear_mView_vis_simulation(M = M,
                                                 C = C,
                                                 mean_matrix = mean_matrix,
                                                 p = p,
                                                 num_samples = num_samples)

# Save
write.table(simData_scenB$X[[1]], "mView_sim_scenB_X1.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(simData_scenB$X[[2]], "mView_sim_scenB_X2.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(simData_scenB$X[[3]], "mView_sim_scenB_X3.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(simData_scenB$X[[4]], "mView_sim_scenB_X4.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(simData_scenB$labels, "mView_sim_scenB_labels.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)
