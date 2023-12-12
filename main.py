from ANN import *

eta=0.01

i=InputLayer(5,"relu")

h1=HiddenLayer(5,"relu")
h1.attach_after(i)
h1.set_weights("random")

o=OutputLayer(5,"none","MSE")
o.attach_after(h1)
o.set_weights("random")

ANN=[i,h1,o]

x=np.array([
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 1, 1],
  [0, 0, 1, 0, 0],
  [0, 0, 1, 0, 1],
  [0, 0, 1, 1, 0],
  [0, 0, 1, 1, 1],
  [0, 1, 0, 0, 0],
  [0, 1, 0, 0, 1],
  [0, 1, 0, 1, 0],
  [0, 1, 0, 1, 1],
  [0, 1, 1, 0, 0],
  [0, 1, 1, 0, 1],
  [0, 1, 1, 1, 0],
  [0, 1, 1, 1, 1],
  [1, 0, 0, 0, 0],
  [1, 0, 0, 0, 1],
  [1, 0, 0, 1, 0],
  [1, 0, 0, 1, 1],
  [1, 0, 1, 0, 0],
  [1, 0, 1, 0, 1],
  [1, 0, 1, 1, 0],
  [1, 0, 1, 1, 1],
  [1, 1, 0, 0, 0],
  [1, 1, 0, 0, 1],
  [1, 1, 0, 1, 0],
  [1, 1, 0, 1, 1],
  [1, 1, 1, 0, 0],
  [1, 1, 1, 0, 1],
  [1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1],
]
)
y=np.array([
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 1, 1],
  [0, 0, 1, 0, 0],
  [0, 0, 1, 0, 1],
  [0, 0, 1, 1, 0],
  [0, 0, 1, 1, 1],
  [0, 1, 0, 0, 0],
  [0, 1, 0, 0, 1],
  [0, 1, 0, 1, 0],
  [0, 1, 0, 1, 1],
  [0, 1, 1, 0, 0],
  [0, 1, 1, 0, 1],
  [0, 1, 1, 1, 0],
  [0, 1, 1, 1, 1],
  [1, 0, 0, 0, 0],
  [1, 0, 0, 0, 1],
  [1, 0, 0, 1, 0],
  [1, 0, 0, 1, 1],
  [1, 0, 1, 0, 0],
  [1, 0, 1, 0, 1],
  [1, 0, 1, 1, 0],
  [1, 0, 1, 1, 1],
  [1, 1, 0, 0, 0],
  [1, 1, 0, 0, 1],
  [1, 1, 0, 1, 0],
  [1, 1, 0, 1, 1],
  [1, 1, 1, 0, 0],
  [1, 1, 1, 0, 1],
  [1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1]
])
# x=np.array([
#     [0,0],
#     [0,1],
#     [1,0],
#     [1,1]
# ])
# y=np.array([
#     [0],
#     [1],
#     [1],
#     [0]
# ])

y

def gradient_descent(ANN,x,y,epochs):
  loss=[]
  for j in range(0,epochs):
    weightgradients=[0]*len(ANN)
    biasgradients=[0]*len(ANN)
    for k in range(0,len(x)):
      ANN[0].put_values(x[k])
      ANN[len(ANN)-1].set_actual(y[k])

      for layer in ANN:
        layer.forward()

      for i in range(len(ANN)-1,0,-1):
        ANN[i].backward()

      for i in range(1,len(ANN)):
        weightgradients[i]+=ANN[i].dLdW
        biasgradients[i]+=ANN[i].dLda.reshape(1,-1)

    loss.append(ANN[len(ANN)-1].loss())

    for i in range(1,len(weightgradients)):
      weightgradients[i]/=len(x)
      biasgradients[i]/=len(x)
    for i in range(1,len(ANN)):
      ANN[i].W=ANN[i].W-eta*(weightgradients[i])
      ANN[i].Bias=ANN[i].Bias-eta*(biasgradients[i])

    print(f"epoch: {j}, Loss: {ANN[len(ANN)-1].loss()}")
    # it+=1
  return ANN,loss


ANN,loss=gradient_descent(ANN,x,y,100000)
plt.plot(loss)
plt.show()

for j in range(0,len(x)):
  ANN[0].put_values(x[j])
  for layer in ANN:
    layer.forward()
  output=ANN[len(ANN)-1].output()
  it=0
  for i in output:
    temp=[]
    for k in i:
      if k>0.7:
        k=1
      elif k<=0.7:
        k=0
      temp.append(k)
  print(f"actual: {y[j]}, predicted: {temp}")
