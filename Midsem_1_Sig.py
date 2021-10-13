import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1,0,0,0],[1,0,1,1],[1,0,2,2],[1,1,0,1],[1,1,1,2],[1,1,2,0],[1,2,0,2],[1,2,1,0],[1,2,2,1]])
# Using sigmoid activation function
lr = 2.5
alpha = 0.1
w12 = []
w23 = [] 
w12.append(np.random.rand(3,4))                          # Weight matrix between layer 1 and 2 (3x4)
w23.append(np.random.rand(5,3))                          # Weight matrix between layer 2 and 3 (5x3)
count =0
e = np.ones((1,3))
for t in range(5000):
    np.random.shuffle(data)
    for i in range(9):
        # Forward computation
        # Calculating output
        x_1 = data[i:i+1,0:3]                         # Input in layer 1 (1x3)
        v_2 = np.dot(x_1,w12[count])                              # Induced local field in layer 2 (1x4)
        v_2 = np.concatenate((np.ones((1,1)),v_2),axis=1)
        y_2 = 1/(1+np.exp((-1)*v_2))                        # Output in layer 2 (1x5)
        v_3 = np.dot(y_2,w23[count])                              # Induced local field in layer 3 (1x3)
        y_3 = 1/(1+np.exp((-1)*v_3))                        # Output in layer 3 (1x3)    
        if data[i,3] == 0:
            d = np.array([[1,0,0]])                          # Desired signal (1x3)
        elif data[i,3] == 1:
            d = np.array([[0,1,0]])                          # Desired signal (1x3)
        elif data[i,3] == 2:           
            d = np.array([[0,0,1]])                          # Desired signal (1x3)
        else:
            print("Error!") 
        e = d - y_3                                         # Error in layer 3 (1x3)

        # Backward computation
        delta_3 = e * y_3 * (1 - y_3)                       # Delta in layer 3 (1x3)
        delta_2 = y_2 * (1 - y_2) * np.transpose((np.dot(w23[count],np.transpose(delta_3))))  # Delta in layer 2 (1x5)
        # Adjust synaptic weights
        if i == 0:
            w12_new = w12[count] + lr * np.dot(np.transpose(x_1),delta_2[0:1,1:5])
            w12.append(w12_new)
            w23_new = w23[count] + lr * np.dot(np.transpose(y_2),delta_3)
            w23.append(w23_new)
        else:
            w12_new = w12[count] + lr * np.dot(np.transpose(x_1),delta_2[0:1,1:5]) + alpha * (w12[count]-w12[count-1])
            w12.append(w12_new)
            w23_new = w23[count] + lr * np.dot(np.transpose(y_2),delta_3) + alpha * (w23[count]-w23[count-1])
            w23.append(w23_new)

        count = count + 1

# TEST MODEL:
correct = 0
for i in range(9):
    x_1 = data[i:i+1,0:3]
    v_2 = np.dot(x_1,w12[count])
    v_2 = np.concatenate((np.ones((1,1)),v_2),axis=1)
    y_2 = 1/(1+np.exp((-1)*v_2))
    v_3 = np.dot(y_2,w23[count])
    y_3 = 1/(1+np.exp((-1)*v_3))
    max = np.amax(y_3) 
    if (y_3[0,0] == max) and (data[i,3]==0):
        correct=correct+1
    if (y_3[0,1] == max) and (data[i,3]==1):
        correct=correct+1
    if (y_3[0,2] == max) and (data[i,3]==2):
        correct=correct+1
accuracy = float(correct)/float(9)


print("Accuracy: ")
print(accuracy)
print(w12[count])

plt.scatter([0,1,2],[0,2,1])
plt.scatter([0,1,2],[1,0,2])
plt.scatter([0,1,2],[2,1,0])
x = [-1.0,0.0,1.0,2.0,3.0,4.0,5.0]
y1 = []
y2 = []
y3 = []
y4 = []
for i in range(7):
    y1.append((-w12[count][0,0]-w12[count][1,0]*x[i])/w12[count][2,0])
for i in range(7):
    y2.append((-w12[count][0,1]-w12[count][1,1]*x[i])/w12[count][2,1])
for i in range(7):
    y3.append((-w12[count][0,2]-w12[count][1,2]*x[i])/w12[count][2,2])
for i in range(7):
    y4.append((-w12[count][0,3]-w12[count][1,3]*x[i])/w12[count][2,3])
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(x,y4)
plt.title("Decision Boundaries with Sigmoid Activation Function")
plt.show()
