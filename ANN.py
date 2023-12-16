#import here
import numpy as np
import matplotlib.pyplot as plt
import os


def SigmoidActivation(x):
  return 1/(1+np.exp(-np.clip(x, -500, 500)))
def ReLUActivation(x):
    return np.maximum(0, x)
def noActivation(x):
  return x
def TanhActivation(x):
  return np.tanh(x)
def SoftmaxActivation(x):
    exp_values = np.exp(x - np.max(x))
    activations=exp_values / np.sum(exp_values)
    return activations

def sigmoid_derivative(x):
    sigmoid_x =SigmoidActivation(x)
    return sigmoid_x * (1 - sigmoid_x)
def ReLUDerivative(x):
    return np.where(x <= 0, 0, 1)
def noActDerivative(x):
  return np.ones((1, len(x)))
def TanhDerivative(x):
    return 1 - np.tanh(x)**2

def softmax_derivative(x):
    n = np.size(x)
    exp_values = np.exp(x - np.max(x))
    softmax_output = exp_values / np.sum(exp_values)
    derivative = softmax_output * (np.identity(n) - softmax_output.T)
    return derivative
# def softmax_derivative(x):
#   n=np.size(x)
#   tmp=np.tile(x,n).reshape(n,n)
#   derivative=tmp*(np.identity(n)-np.transpose(tmp))
#   return derivative

    
def MSE_Derivative(activations,actual):
  y=actual
  activations=activations[0]
  dLdy=0
  for i in range(0,len(y)):
    dLdy+=(2/len(activations))*(activations[i]-y[i])
  return dLdy
def Binary_CrossEntropy_Derivative(activations, actual):
    activations = activations[0]
    epsilon = 1e-10
    activations = np.clip(activations, epsilon, 1 - epsilon)  # Avoiding division by zero
    return np.array((activations - actual) / (activations * (1 - activations) + epsilon)).reshape(-1, 1)
def CrossEntropy_Derivative(activations,actual):
    epsilon = 1e-10
    activations=activations[0]
    activations = np.clip(activations, epsilon, 1 - epsilon)  # Avoiding division by zero
    # print("Actual: ",actual)
    # print("Activations: ",activations)
    derivative=np.zeros_like(activations)
    tc=-1
    for i in range(0,len(actual)):
       if actual[i]==1:
          tc=i
          break
    derivative[tc] = -actual[tc] / activations[tc]
    derivative=derivative.reshape(-1,1)
    return derivative



def MSE_Loss(activations, actual):
    loss = 0
    activations=activations[0]
    for i in range(len(activations)):
        loss += (activations[i] - actual[i]) ** 2
    return (1 / len(activations)) * loss
def Binary_CrossEntropy_Loss(activations, actual):
    epsilon = 1e-10
    activations=activations[0]
    activations = np.clip(activations, epsilon, 1 - epsilon)  # Avoiding logarithm of zero
    loss = -np.sum(actual * np.log(activations) + (1 - actual) * np.log(1 - activations))
    return loss/len(activations)
def CrossEntropy_Loss(activations,actual):
    activations=activations[0]
    # print("actual: ",actual)
    # print("activations: ",activations)
    tc=-1
    for i in range(0,len(actual)):
       if actual[i]==1:
          tc=i
          break
    epsilon = 1e-10  # small value to prevent log(0)
    activations = np.clip(activations, epsilon, 1 - epsilon)  # clip to avoid log(0) or log(1)
    ce_loss = -(actual[tc] * np.log(activations[tc]))
    return ce_loss




class Layer():
  def __init__(self,n):
    self.length=n

class InputLayer():
  def __init__(self,n,actfn="none"):
    self.length=n
    self.inputs=np.array([])
    self.next=None
    self.actname=""
    if actfn=="sigmoid":
      self.actfn=SigmoidActivation
      self.actname="sigmoid"
    elif actfn=="relu":
      self.actfn=ReLUActivation
      self.actname="relu"
    elif actfn=="tanh":
      self.actfn=TanhActivation
      self.actname="tanh"
    elif actfn=="none":
      self.actfn=noActivation
      self.actname="none"

  def put_values(self,values):
    if len(values)==self.length:
      self.inputs=np.array([values])
    else:
      print("Error: The values you are trying to insert are more then the allocated size of input vector")
  def forward(self):
    self.pre_activations=self.inputs
    self.pre_activations = np.squeeze(self.pre_activations)
    self.activations=self.actfn(self.pre_activations)

class HiddenLayer():
  def __init__(self,n,actfn="none"):
    self.length=n
    self.Bias=None
    self.next=None
    self.actname=""
    if actfn=="sigmoid":
      self.actfn=SigmoidActivation
      self.actname="sigmoid"
    elif actfn=="relu":
      self.actfn=ReLUActivation
      self.actname="relu"
    elif actfn=="tanh":
      self.actfn=TanhActivation
      self.actname="tanh"
    elif actfn=="none":
      self.actfn=noActivation
      self.actname="none"

  def attach_after(self,layer):
    self.previous=layer
    self.previous.next=self

  def set_weights(self,method="random"):
    if self.length>0:

      if method=="normal_random":
        self.W = np.random.randn(self.length, self.previous.length)
      elif method=="uniform_random":
        self.W = np.random.rand(self.length, self.previous.length)
      elif method=="xavier":
        self.W = (1/self.length**0.5)*np.random.randn(self.length, self.previous.length)
      elif method=="one":
        self.W = np.ones((self.length, self.previous.length))
      elif method == "he":
        self.W = np.random.randn(self.length, self.previous.length) * np.sqrt(2.0 / (self.previous.length*self.length))
      elif method == "lecun":
        limit = np.sqrt(1.0 / self.previous.length)
        self.W = np.random.uniform(-limit, limit, (self.length, self.previous.length))

  def set_biases(self,method="random"):
      if self.length>0 and self.previous!=None:

        if method=="normal_random":
            self.Bias = np.random.randn(1, self.length)
        elif method=="uniform_random":
            self.Bias = np.random.rand(1, self.length)
        elif method == "zeros":
            self.Bias = np.zeros((1, self.length))
        elif method == "constant":
            self.Bias = np.full((1, self.length), 0.1)  # Set bias to a constant value
        elif method == "xavier":
            self.Bias = np.random.randn(1, self.length) * np.sqrt(1 / self.length)
        elif method == "lecun":
            self.Bias = np.random.randn(1, self.length) * np.sqrt(1 / self.length)
        elif method == "he":
            self.Bias = np.random.randn(1, self.length) * np.sqrt(1 / self.length)
        # Add other bias initialization methods here...



  def forward(self):
    self.pre_activations=np.dot(self.W,self.previous.activations)+self.Bias
    self.pre_activations = np.squeeze(self.pre_activations)
    self.activations=self.actfn(self.pre_activations)

  def backward(self):
    nextdLda=self.next.dLda
    dadh=self.next.W
    if self.actname=="relu":
      dhda=ReLUDerivative(self.pre_activations)
    elif self.actname=="sigmoid":
      dhda=sigmoid_derivative(self.pre_activations)
    elif self.actname=="tanh":
      dhda=TanhDerivative(self.pre_activations)
    elif self.actname=="none":
      dhda=noActDerivative(self.pre_activations)
    dadW=self.previous.activations

    dhda=dhda.reshape(-1, 1)
    dadW=dadW.reshape(-1, 1)
    dLdh=np.dot(dadh.T,nextdLda)
    dLda=np.multiply(dLdh,dhda)
    dLdW=np.dot(dLda,dadW.T)
    self.dLda=dLda
    self.dLdW=dLdW







class OutputLayer():
  def __init__(self,n,outputfn="none",lossfn="MSE"):
    self.length=n
    self.previous=None
    self.W=None
    self.next=None
    self.Bias=None
    self.Loss_grad_W=None
    self.lossfnname=""

    self.actname=""
    if outputfn=="sigmoid":
      self.outputfn=SigmoidActivation
      self.actname="sigmoid"
    elif outputfn=="relu":
      self.outputfn=ReLUActivation
      self.actname="relu"
    elif outputfn=="none":
      self.outputfn=noActivation
      self.actname="none"
    elif outputfn=="softmax":
      self.outputfn=SoftmaxActivation
      self.actname="softmax"
    elif outputfn=="tanh":
      self.outputfn=TanhActivation
      self.actname="tanh"

    if lossfn=="MSE":
      self.lossfn=MSE_Loss
      self.lossfnname="MSE"
    elif lossfn=="bincrossentropy":
      self.lossfn=Binary_CrossEntropy_Loss
      self.lossfnname="bincrossentropy"
    elif lossfn=="crossentropy":
      self.lossfn=CrossEntropy_Loss
      self.lossfnname="crossentropy"


  def attach_after(self,layer):
    self.previous=layer
    self.previous.next=self

  def set_weights(self,method="random"):
      if self.length>0 and self.previous!=None:

        if method=="normal_random":
            self.W = np.random.randn(self.length, self.previous.length)
        elif method=="uniform_random":
            self.W = np.random.rand(self.length, self.previous.length)
        elif method=="xavier":
          self.W = (1/self.length**0.5)*np.random.randn(self.length, self.previous.length)
        elif method=="one":
          self.W = np.ones((self.length, self.previous.length))
        elif method == "he":
          self.W = np.random.randn(self.length, self.previous.length) * np.sqrt(2.0 / self.previous.length)
        elif method == "lecun":
            limit = np.sqrt(1.0 / self.previous.length)
            self.W = np.random.uniform(-limit, limit, (self.length, self.previous.length))

  def set_biases(self,method="random"):
      if self.length>0 and self.previous!=None:

        if method=="normal_random":
            self.Bias = np.random.randn(1, self.length)
        elif method=="uniform_random":
            self.Bias = np.random.rand(1, self.length)
        elif method == "zeros":
            self.Bias = np.zeros((1, self.length))
        elif method == "constant":
            self.Bias = np.full((1, self.length), 0.1)  # Set bias to a constant value
        elif method == "xavier":
            self.Bias = np.random.randn(1, self.length) * np.sqrt(1 / self.length)
        elif method == "lecun":
            self.Bias = np.random.randn(1, self.length) * np.sqrt(1 / self.previous.length)
        # Add other bias initialization methods here...

  def forward(self):
      self.pre_activations = np.dot(self.W, self.previous.activations) + self.Bias
      self.activations = self.outputfn(self.pre_activations)


  def output(self):
    return self.activations

  def set_actual(self,actual):
    self.actual=actual

  def loss(self):
    return self.lossfn(self.activations,self.actual)

  def backward(self):
    if self.lossfnname=="MSE":
      dLdy=MSE_Derivative(self.activations,self.actual)
    elif self.lossfnname=="bincrossentropy":
      dLdy=Binary_CrossEntropy_Derivative(self.activations,self.actual)
    elif self.lossfnname=="crossentropy":
      dLdy=CrossEntropy_Derivative(self.activations,self.actual)

    if self.actname=="sigmoid":
      dyda = sigmoid_derivative(self.pre_activations[0])
    elif self.actname=="relu":
      dyda = ReLUDerivative(self.pre_activations[0])
    elif self.actname=="none":
      dyda = noActDerivative(self.pre_activations[0])
    elif self.actname=="softmax":
      dyda = softmax_derivative(self.pre_activations[0])
    elif self.actname=="tanh":
      dyda=TanhDerivative(self.pre_activations)
      
    # print("dLdy: ",dLdy)
    # print("dLdy shape: ",dLdy.shape)
    # print("dyda: ",dyda)
    # print("dyda shape: ",dyda.shape)

    dadW=self.previous.activations


    dadW=dadW.reshape(-1,1)
    
    if self.actname!="softmax":
      dyda=dyda.reshape(-1,1)
      dLda=dLdy*dyda
    else:
      dyda=dyda
    #   dLda=np.dot(dyda,dLdy)
      dLda=self.activations-self.actual
      dLda=dLda.reshape(-1,1)
    # print("dLda: ",dLda)
    # print("dLda shape: ",dLda.shape)
      
    dLdW = np.dot(dLda, dadW.T)

    self.dLdy=dLdy
    self.dyda=dyda
    self.dLda=dLda
    self.dLdW=dLdW

def gradient_descent_threshold(ANN,x,y,eta,thresh):
  loss=[]
  ANN[0].put_values(x[0])
  ANN[len(ANN)-1].set_actual(y[0])

  for layer in ANN:
    layer.forward()
  j=0
  while ANN[-1].loss()>thresh:
    if len(loss)>2 and loss[-1]>loss[-2]:
       break
    for k in range(0,len(x)):
      ANN[0].put_values(x[k])
      ANN[len(ANN)-1].set_actual(y[k])

      for layer in ANN:
        layer.forward()
    #   print("Activations: ",ANN[-1].activations)
    #   print("Output: ",ANN[-1].output())

      for i in range(len(ANN)-1,0,-1):
        ANN[i].backward()

      for i in range(1,len(ANN)):
        ANN[i].W-=eta*ANN[i].dLdW
        ANN[i].Bias-=eta*ANN[i].dLda.reshape(1,-1)

    loss.append(ANN[len(ANN)-1].loss())
    j+=1
    print(f"epoch: {j}, Loss: {ANN[len(ANN)-1].loss()}")
  return ANN,loss

def gradient_descent_epoch(ANN,x,y,eta,epochs):
  loss=[]
  for j in range(0,epochs):
    corrects=0
    for k in range(0,len(x)):
      ANN[0].put_values(x[k])
      ANN[len(ANN)-1].set_actual(y[k])

      for layer in ANN:
        layer.forward()
    #   print("Predicted: ",ANN[-1].output())
    #   print("Actual: ",y[k])

      actual_label = np.argmax(y[k])
      output = ANN[-1].output()
      predicted_label = np.argmax(output)
    #   print("Actual: ",actual_label)
    #   print("output: ",output)
    #   print("predicted_label: ",predicted_label)

      if predicted_label==actual_label:
         corrects+=1
    
    #   print("Input no: ",k)
    #   print("Activations: ",ANN[-1].activations)
    #   print("Output: ",ANN[-1].output())

      for i in range(len(ANN)-1,0,-1):
        ANN[i].backward()

      for i in range(1,len(ANN)):
        ANN[i].W-=eta*ANN[i].dLdW
        ANN[i].Bias-=eta*ANN[i].dLda.reshape(1,-1)

    loss.append(ANN[len(ANN)-1].loss())
    print(f"epoch: {j}, Loss: {ANN[len(ANN)-1].loss()}, Accuracy: {100*corrects/len(y)}")
  return ANN,loss
