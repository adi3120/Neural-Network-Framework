#import here
import numpy as np
import matplotlib.pyplot as plt
import os

def SigmoidActivation(x):
  return 1/(1+np.exp(-x))
def ReLUActivation(x):
    return np.maximum(0, x)
def noActivation(x):
  return x
def TanhActivation(x):
    return np.tanh(x)
def SoftmaxActivation(x):
    exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

def sigmoid_derivative(x):
    sigmoid_x = 1 / (1 + np.exp(-x))
    return sigmoid_x * (1 - sigmoid_x)
def ReLUDerivative(x):
    return np.where(x <= 0, 0, 1)
def noActDerivative(x):
  return np.ones((1, len(x)))
def TanhDerivative(x):
    return 1 - np.tanh(x)**2
def softmax_derivative(x):
    s = np.exp(x)
    return s / np.sum(s, axis=1, keepdims=True) * (1 - s / np.sum(s, axis=1, keepdims=True))

def MSE_Loss(activations, actual):
    loss = 0
    activations=activations[0]
    # print("Activations: ",activations)
    for i in range(len(activations)):
        loss += (activations[i] - actual[i]) ** 2
    return (1 / len(activations)) * loss

def CrossEntropy_Loss(activations, actual):
    activations=activations[0]
    epsilon = 1e-15  # Small value to prevent log(0)
    clipped_activations = np.clip(activations, epsilon, 1 - epsilon)
    loss = -np.sum(actual * np.log(clipped_activations))
    return loss / len(activations)


def MSE_Derivative(activations,actual):
  y=actual
  activations=activations[0]

  dLdy=0

  for i in range(0,len(y)):
    dLdy+=(2/len(activations))*(activations[i]-y[i])
  return dLdy
def CrossEntropy_Derivative(activations, actual):
    activations=activations[0]
    return activations - actual

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
    self.Bias=np.random.randn(1,self.length)
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

      if method=="random":
        self.W = np.random.randn(self.length, self.previous.length)
      elif method=="one":
        self.W = np.ones((self.length, self.previous.length))

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
    self.Bias=np.random.randn(1,self.length)
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

    if lossfn=="MSE":
      self.lossfn=MSE_Loss
      self.lossfnname="MSE"

    elif lossfn=="crossentropy":
      self.lossfn=CrossEntropy_Loss
      self.lossfnname="crossentropy"


  def attach_after(self,layer):
    self.previous=layer
    self.previous.next=self

  def set_weights(self,method="random"):
      if self.length>0 and self.previous!=None:
        if method=="random":
          self.W = np.random.randn(self.length, self.previous.length)
        elif method=="one":
          self.W = np.ones((self.length, self.previous.length))

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


    dadW=self.previous.activations

    dyda=dyda.reshape(-1,1)
    dadW=dadW.reshape(-1,1)

    dLdW = np.dot(dLdy*dyda, dadW.T)

    self.dLdy=dLdy
    self.dyda=dyda
    self.dLda=dLdy*dyda
    self.dLdW=dLdW

