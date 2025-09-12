#0203
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt


#1: load train data
with np.load("/home/inteee/opencvZoo2/data/0201_data50.npz") as X:
    x_train = X['x_train'].astype(np.float32)
    y_train = X['y_train'].astype(np.int32)
    height, width = X['size']
    
#2: k-nearest neighbours: create, train, and predict
#2-1
model = cv2.ml.KNearest_create() # 모델 세팅
ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train) # xtrain, 숫자샘프 배치, ytrain 

#2-2
step = 2
xx, yy = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))

x_test = np.float32(np.c_[xx.ravel(), yy.ravel()])
k = 20
ret, pred = model.predict(x_test, k) # pred.shape = (75000, 1)
pred = pred.reshape(xx.shape) # pred.shape = (250, 300)

#3: display data and result
#3-1
ax = plt.gca()
ax.set_aspect('equal')
#ax.axis('off')
#ax.xaxis.tick_bottom()
#ax.xaxis.tick_top()
#ax.invert_yaxis()

#3-2
class_colors = ['blue', 'red']
plt.contourf(xx, yy, pred, cmap = plt.cm.gray)
plt.contour(xx, yy, pred, colors = "red", linewidths = 1)


#3-3
for label in range(2):
    plt.scatter(x_train[y_train == label, 0], x_train[y_train == label, 1], 20, class_colors[label], 'o')


plt.show()