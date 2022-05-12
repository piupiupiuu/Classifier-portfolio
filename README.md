# Classifier-portfolio
We can view each classifier as an individual stock, and borrow the idea from investment portfolio to balance the bias-variance of classifiers (ensemble learning).  
We assume there's only two classifiers in our portfolio. The weight of classifier1 = w1, the weight of classifier2 = w2 and w1+w2=1
## 1. Bias:
bias of the first classifier:  
**bias(θ1_hat) = E(θ1_hat) - θ**  
bias of the second classifier:  
**bias(θ2_hat) = E(θ2_hat) - θ**  
Bias of the portfolio:   
**bias(w1θ1_hat + w2θ2_hat) = E(w1θ1_hat + w2θ2_hat) - θ**  
                          **= E(w1θ1_hat) + E(w2θ2_hat) - θ**  
                          **= E(w1θ1_hat) + E(w2θ2_hat) - (w1+w2)θ**  
                          **= w1E(θ1_hat) + w2E(θ2_hat) - w1θ - w2θ**  
                          **= w1(E(θ1_hat) - θ) + w2(E(θ2_hat) - θ)**  
                          **=w1bias(θ1_hat) + w2bias(θ2_hat)**  
