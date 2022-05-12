# Classifier-portfolio
We can view each classifier as an individual stock, and borrow the idea from investment portfolio to balance the bias-variance of classifiers (ensemble learning).  
We will assume there's only two classifiers in our portfolio. The weight of classifier1 = w1, the weight of classifier2 = w2 and w1+w2=1.  
Classifiers with low bias usually will have higher variance, thus we will also assume classifier 1 has a lower bias but higher variance, classifier 2 has a higher bias but lower variance. 
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

## 2. Variance:
standard deviation of the first classifier: **std(θ1_hat)** 
standard deviation of the second classifier: **std(θ2_hat)**  
standard deviation of the portfolio:  
**std(w1θ1_hat + w2θ2_hat)^2 = var(w1θ1_hat) + var(w2θ2_hat) + 2w1w2Cov(θ1_hat,θ2_hat)**  
                         **= w1^2var(θ1_hat) + w2^2var(θ2_hat) + 2w1w2Cov(θ1_hat,θ2_hat)**   
                         **= w1^2std(θ1_hat)^2 + w2^2std(θ2_hat)^2 + 2w1w2pstd(θ1_hat)std(θ2_hat)**   
                         **=< (w1std(θ1_hat) + w2std(θ2_hat))^2**  
This is where we get the benefit from the combination of the classifiers.

## 3. Efficient frontier：
![efficient frontier (2)](https://user-images.githubusercontent.com/96370219/168144604-89d8be17-ee73-4ca7-a265-51a03e1a4831.png)  
The dot line shows the bias and variance of the combinations of classifiers with correlation = 1. The curve shows the combinations of classifier 1 and 2 with correlation < 1. The combinations of classifiers on the green line has the same variance as the combinations on the black line, but with lower bias, thus we will only consider the combinations on the green line.  
![efficient frontier (3)](https://user-images.githubusercontent.com/96370219/168147280-67173bdf-38cb-4948-8169-42c3002655d4.png)  
Now we add another classifier with 0 variance but high variance (stable classifier). And we find a tangent to the green line which go through the stable classifier, we now have a new line which is the combination between the stable classifier and the combination of classifier 1 and 2. Compare to the original green line, for the same bias, the combinations on the new black line can achieve a lower variance. Note that, the new black line is also the linear combination between the stable classifier and the points on the green line with the lowest gradient (or the highest absolute gradient).

## 4. Algorithm:
w1 = weight of classifier 1, w2 = weight of classifier 2, w = weight of stable classifier, where w1 + w2 = 1  
1. Find the weight **w1 w2** which maximizes **the absolute slope = (bias(θ_stable_hat) - bias(w1θ1_hat + w2θ2_hat)) / std(w1θ1_hat + w2θ2_hat)** where **std(w1θ1_hat + w2θ2_hat) = (w1^2std(θ1_hat)^2 + w2^2std(θ2_hat)^2 + 2w1w2pstd(θ1_hat)std(θ2_hat))^0.5** and **bias(w1θ1_hat + w2θ2_hat)) = w1bias(θ1_hat) + w2bias(θ2_hat)**  
2. Based on the risk preference, choose the weight w to generate a portfolio: w*stable_classifier + (1-w)*(w1*classifier_1 + w2*classifier 2)




