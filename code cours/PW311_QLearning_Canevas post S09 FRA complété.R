### Résolution par Q-learning simple du problème PW311 vu en classe
### Solving by plain Q-learning the PW311 problem discussed in class

### Appeler par exemple avec // Call for example with
# probs <- c(.7,.8,.9,0); payoff <- c(0,10,20,30); discount <- 0.8
# numSteps <- 10000; startState <- 1; alpha <- 0.1
# startQFactors <- matrix(c(rep(50,3),-Inf,-Inf,payoff[2:4]),ncol=2)

### Et "set.seed(555)" si désiré, pour figer le départ.
### And "set.seed(555)" to fix the starting point, if desired.


PW311_QLearning <- function(probs,payoff,discount,numSteps,
                            startState,startQFactors,alpha){
  QFactors <- startQFactors
  state <- startState
  numStates <- length(probs)  
  
  for (k in 1:numSteps){
    if (state == 1){ # state 1, no choice is possible
      if (runif(1)>probs[state])  
        newState <- 2
      else
        newState <- 1
      action <- 1
      reward <- 0
    }
    else if (state == numStates){ #last state, pocket reward, go back to state 1
      newState <- 1
      action <- 2
      reward <- payoff[state]
    }
    else { # States 2 and 3, need to choose optimal action
      action <- which.max(QFactors[state,])
      if (action == 2){   # pocket reward and go back to state 1
        newState <- 1
        reward <- payoff[state]
      }
      else { # Action is 1, so random transition 
        if (runif(1) > probs[state]){
          newState <- newState + 1
          reward <- 0
        }
      }
    } # end if on states
    
    ## Seules deux lignes caviardées dans le canevas 
    ## Only lines hidden in the canevas
    qhat <- reward+discount*max(QFactors[newState,])
    QFactors[state,action] <- alpha*qhat+(1-alpha)*QFactors[state,action]
    state <- newState
  } # end for
  # Find optimal policy (for interior states)
  indexVet <- 2:(numStates-1)
  policy <- c(0,QFactors[indexVet,2]>QFactors[indexVet,1],1) +1
  out <- list(QFactors=QFactors,policy=policy)
  return(out)
}

