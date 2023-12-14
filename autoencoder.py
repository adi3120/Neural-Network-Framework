#import here
import numpy as np
from ANN import *
import matplotlib.pyplot as plt

def visualize_binary_sequence(sequence, title, ax,len):
    ax.set_title(title)
    im = ax.imshow(sequence.reshape(len, -1), cmap='binary')
    plt.colorbar(im, ax=ax) 
    im.set_clim(0, 1)  


eta=0.001

i=InputLayer(5,"tanh")

h1=HiddenLayer(3,"tanh")
h1.attach_after(i)
h1.set_weights("random")

o=OutputLayer(5,"sigmoid","bincrossentropy")
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
  return ANN,loss


ANN,loss=gradient_descent(ANN,x,y,15000)
plt.plot(loss)
plt.show()

outputs=[]
hiddens=[]

for j in range(0,len(x)):
  ANN[0].put_values(x[j])
  for layer in ANN:
    layer.forward()
  output=ANN[len(ANN)-1].output()
  outputs.append(output)
  hidden=ANN[1].activations
  hiddens.append(hidden)


num_samples = len(x)
fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples)) 

for j in range(num_samples):
    visualize_binary_sequence(x[j], f"Input {j} - Actual: {y[j]}", axs[j][0],ANN[0].length)

    visualize_binary_sequence(outputs[j], f"Predicted Output", axs[j][1],ANN[2].length)

    visualize_binary_sequence(hiddens[j], f"Hidden Layer", axs[j][2],ANN[1].length)

plt.tight_layout()
plt.show()





plt.imshow(ANN[1].W,cmap='cividis')
plt.title("Encoder Weight Matrix")

plt.show()

plt.imshow(ANN[2].W,cmap='cividis')
plt.colorbar() 
plt.title("Decoder Weight Matrix")

plt.show()

