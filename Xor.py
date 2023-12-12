from ANN import *

eta=0.1

i=InputLayer(2,"tanh")

h1=HiddenLayer(1,"tanh")
h1.attach_after(i)
h1.set_weights("random")

o=OutputLayer(1,"none","MSE")
o.attach_after(h1)
o.set_weights("random")

ANN=[i,h1,o]

x=np.array([
  [0,0], 
  [0,1],
  [1,0],
  [1,1]
]
)
y=np.array([
  [0], 
  [1], 
  [1], 
  [0]
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
    for k in range(0,len(x)):
      ANN[0].put_values(x[k])
      ANN[len(ANN)-1].set_actual(y[k])

      for layer in ANN:
        layer.forward()

      for i in range(len(ANN)-1,0,-1):
        ANN[i].backward()

      for i in range(1,len(ANN)):
        ANN[i].W-=eta*ANN[i].dLdW
        ANN[i].Bias-=eta*ANN[i].dLda.reshape(1,-1)

    loss.append(ANN[len(ANN)-1].loss())

    print(f"epoch: {j}, Loss: {ANN[len(ANN)-1].loss()}")
    # it+=1
  return ANN,loss


ANN,loss=gradient_descent(ANN,x,y,50000)
plt.plot(loss)
plt.show()

for j in range(0,len(x)):
  ANN[0].put_values(x[j])
  for layer in ANN:
    layer.forward()
  output=ANN[len(ANN)-1].output()

  print(f"actual: {y[j]}, predicted: {output}")
