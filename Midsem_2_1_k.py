import numpy as np
import matplotlib.pyplot as plt
import random

# Generate sequence v(n):
v = np.random.normal(loc = 0,scale = 1,size=(1,100000))
# Calculate x(n) (the desired outputs):
x = np.zeros((1,100000))
beta = 0.5
for i in range(100000):
    if i==0:
        x[0,i] = v[0,0]
    elif i==1:
        x[0,i] = v[0,1]
    else:
        x[0,i] = v[0,i] + beta*v[0,i-1]*v[0,i-2]

lr = 0.001
alpha = 0.6
w12 = []
w23 = []
w12.append(np.random.rand(6,16))                    # Weight matrix between layer 1 and layer 2 (6x16)
w23.append(np.random.rand(16,1))                    # Weight matrix between layer 2 and layer 3 (16x1)
count = 0
var = []                                        # Variances for each epoch
error = []                                      # MSE for each epoch
for t in range(1000):

    i = random.randint(6,97999)
    for ctr in range(1000):
        
        x_1 = np.array([[v[0,i+ctr],v[0,i-1+ctr],v[0,i-2+ctr],v[0,i-3+ctr],v[0,i-4+ctr],v[0,i-5+ctr]]])              # Input in layer 1 (1X6)
        v_2 = np.dot(x_1,w12[count])                                                         # Induced local field in layer 2 (1x16)
        y_2 = 1/(1+np.exp((-1)*v_2))                                                       # Output in layer 2 (1x16)
        v_3 = np.dot(y_2,w23[count])                                                       # Induced local field in layer 3 (1x1)
        y_3 = v_3                                                                          # Output in layer 3 (1x1)
        e = x[0:1,i+ctr:i+1+ctr] - y_3                                                               # Error (1x1)
        # Backward computation
        delta_3 = e                                                          # Delta in layer 3 (1x1)
        delta_2 = y_2 * (1 - y_2) * np.transpose((np.dot(w23[count],np.transpose(delta_3))))  # Delta in layer 2 (1x16)
        # Adjust synaptic weights
        if count == 0:
            w12_new = w12[count] + lr * np.dot(np.transpose(x_1),delta_2[0:1,0:16])
            w12.append(w12_new)
            w23_new = w23[count] + lr * np.dot(np.transpose(y_2),delta_3)
            w23.append(w23_new)
        else:
            w12_new = w12[count] + lr * np.dot(np.transpose(x_1),delta_2[0:1,0:16]) + alpha * (w12[count]-w12[count-1])
            w12.append(w12_new)
            w23_new = w23[count] + lr * np.dot(np.transpose(y_2),delta_3) + alpha * (w23[count]-w23[count-1])
            w23.append(w23_new)

        count = count + 1
    
    num = 1000
    set = np.zeros((1,num))
    
    
    for k in range(98999,98999+num):
        x_1 = np.array([[v[0,k],v[0,k-1],v[0,k-2],v[0,k-3],v[0,k-4],v[0,k-5]]]) 
        v_2 = np.dot(x_1,w12[count]) 
        y_2 = 1/(1+np.exp((-1)*v_2))
        v_3 = np.dot(y_2,w23[count])
        y_3 = v_3
        set[0,k-98999] = y_3[0,0]
    var.append(np.var(set))
    sum = 0
    for j in range(98999,98999+num):
        sum = sum +(set[0,j-98999]-x[0,j])**2
    error.append(sum)

x0 = []
var1 = []
error1 = []
for i in range(1000):
    if i%10 == 0:
        x0.append(i)
        var1.append(var[i])
        error1.append(error[i])


plt.plot(x0,var1)
plt.xlabel("Epoch")
plt.ylabel("Variance")
plt.title("Variance vs Epoch")
plt.show()
plt.plot(x0,error1)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE vs Epoch")
plt.show()

    







