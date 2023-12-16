#import here
import numpy as np
import matplotlib.pyplot as plt
from ANN import *
from PIL import Image

input_image = Image.open('image1.jpg').convert('L').resize((20, 20))
output_image = Image.open('image1.jpg').convert('L').resize((20, 20))


# plt.imshow(input_image,cmap='gray')
# plt.show()

input_array = np.array(input_image) / 255.0
output_array = np.array(output_image) / 255.0

input_flat = input_array.flatten()
output_flat = output_array.flatten()

x = np.array([input_flat])
y = np.array([output_flat])


eta=0.001

i=InputLayer(x.shape[1],"none")

h1=HiddenLayer(x.shape[1],"relu")
h1.attach_after(i)
h1.set_weights("random")
h1.set_biases("random")

o=OutputLayer(x.shape[1],"sigmoid","MSE")
o.attach_after(h1)
o.set_weights("random")
o.set_biases("random")

ANN=[i,h1,o]

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


ANN,loss=gradient_descent(ANN,x,y,10000)
plt.plot(loss)
plt.show()

imagenew=ANN[2].activations.reshape(input_array.shape)

plt.imshow(imagenew,cmap='gray')
plt.show()
