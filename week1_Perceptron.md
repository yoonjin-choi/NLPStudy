# Perceptron
 - 인공신경망(Neuron 의 동작 방식)의 한 종류
   <img src="img/deep1.PNG">
  
 - weight(가중치)는 각각의 입력신호와의 계산을 하고 신호의 총합이 threshold를 넘으면 1, 넘지 못하면 0 or -1을 출력한다. 
 - input X * w + Bias -> ouput 
 - output은 activation function (예 sigmoid)을 거쳐 최종적 결정.
 * 퍼셉트론의 출력값은 1 or 0(또는 -1) 이므로
  -> **linear classifier**  즉, 선형 분류 모형 -> And, OR 문제는 해결가능
  
 * **but, XOR 문제 해결이 불가능.** -> multi layer 이 필요하다.
 <img src="img/xor2.gif">
 
-BackPropogation (조금 더 찾아 정리)
output과 정답간 loss 최소화 하기 위해 weight 업데이트 뒷 단에서 부터 

# RELU activation
* Sigmoid 의 문제점
 backpropogation 알고리즘에서 activation 값이 graident를 곱하는데
 뒷 단에서 부터 gradient 가 소멸됨 -> Vanishing Gradient
 Layer이 많이 사용 되었을 경우 더 문제가 됨.
 
* ReLU fuction
		                    f(x) = max(0,x)
	-  음수영역에서는 gradient 가 0이므로 문제 발생 가능.
* pyTorch code
	- torch.nn.sigmoid(x) 대신 torch.nn.relu(x) 사용 가능
	- torch.nn.leaky_relu(x,0.01) - x가 음수인 부분에서 완화시켜주는 함수.

* Optimizer in PyTorch
