# NLP Week3
 ## Activation function ( 활성함수)
 - 활성화 함수를 사용하는 이유는?
	 - input에 대한 output이 linear하지 않으므로 선형분류기를 비선형시스템으로 바꾼다.
	 - MLP(multiple Layer Perceptron)은 활성화 함수를 이용한 non linear을 여러 layer로 쌓는 것.
1. Sigmoid
- 출력값은 0~1
- Vanishing Gradient 문제 & exponential 연산 비용이 큰 단점.
2. Tanh
- Sigmoid보다 조금 더 발전된 모델
-  Vanishing Graident 문제 존재
3. ReLU
- input이 0보다 클 땐 input 값, 0보다 작으면 0값
- 입력값이 0 미만일 경우, 기울기 =0 이므로 학습 불가능
## 2.  RNN과 LSTM(Long-Short term memory)
- RNN 의 단점
	- RNN은 비교적 짧은 sequence에 대해서만 효과를 보이는 단점.
	- 시점(time step)이 점점 길어질수록, 앞의 정보가 뒤로 충분히 전달되지 못한다.
	- (예) **모스크바**에 여행을 왔는데 건물도 예쁘고 먹을 것도 맛있었어. 그런데 글쎄 직장 상사한테 전화가 왔어. 어디냐고 묻더라구 그래서 나는 말했지. 저 여행왔는데요. 여기 ___'' 
	
		->  '모스크바'는 제일 앞에 위치하고 있고, RNN이 충분한 기억력을 가지고 있지 못한다면 다음 단어를 엉뚱하게 예측
	- $$h_{t}=\tanh\left (  W_{x}x_{t}+W_{h}h_{t-1}+b \right )$$
	 > xt와 ht-1 두 입력이 각각의 가중치와 곱해져 하이퍼볼릭탄젠트함수의 입력으로 사용되고, 출력값은 은닉층의 출력인 은닉상태가 됨.

 - LSTM(장단기 메모리)
	 - 은닉층의 메모리 셀에 입력 게이트, 망각 게이트, 출력 게이트를 추가하여 불필요한 기억 삭제, 기억해야할 것을 지정.
	- RNN에 비해 조금 더 복잡, cell state $C_{t}$ 값이 추가됨.
	- RNN에 비해 긴 시퀀스의 입력 처리에 탁월하다. 
	![enter image description here](https://wikidocs.net/images/page/22888/vaniila_rnn_and_different_lstm_ver2.PNG)

	- Forget gate : 
	$$ f_{t}=\sigma \left ( W_{xf}x_{t}+ W_{hf}h_{t-1} + b_{f}\right ) $$
		> 현재시점의 x값과 이전시점 은닉상태(h)가 시그모이드 함수를 지나면 0과 1사이의 값이 나오는데, 0에 가까울수록 많이 삭제된 상태, 1에 가까울 수록 정보를 온전히 기억한 것이며 이를 통해 셀 상태($C_{t}$)를 구함.
	- Input gate :   
		$$i_{t}=\sigma \left ( W_{xi}x_{t}+ W_{hi}h_{t-1} + b_{i}\right )$$
		$$g_{t}=\tanh  \left ( W_{xg}x_{t}+ W_{hg}h_{t-1} + b_{g}\right)$$
	  > 시그모이드 함수를 지나 0 ~1 사이 값, tanh 함수를 지나 -1 ~1 사이 값 두개가 나오고, 이 두값을 이용하여 선택된 기억할 정보의 양을 결정한다.
	- Cell state(셀 상태, 장기상태) :   $$C_{t}=f_{t}\odot C_{t-1}+ i_{t}\odot g_{t}$$
	  >입력게이트에서 구한 두 값 $i_{t}$ 와 $g_{t}$ 두 값에 대해 원소별 곱( (Hardmard)아다마르곱:같은 위치의 성분끼리 곱하는 것) 이 이번에 선택된 기억할 값.
	  
	   >입력게이트에서 선택된 기억 + 삭제 게이트의 기억 = 현재 시점 t의 셀 상태이며, 
	   이 값은 다음 t+1 시점의 LSTM 셀로 넘어간다.

		>삭제 게이트의 출력값 $f_{t}$가 0이라면, 지난 시점의 셀 상태($C_{t-1}$)은 영향력이 0이 되며, 오직 입력게이트의 결과 값만이 현재 시점의 셀 상태 $C_{t}$를 결정, 반대로 입력 게이트 $i_{t}$의 값이 0이라면, 현재 시점의 셀 상태  $C_{t}$는 지난 시점의 셀 상태 $C_{t-1}$의 영향만을 받는다.
	   
	- Output gate: 
	$$ o_{t}=\sigma  \left ( W_{xo}x_{t}+ W_{ho}h_{t-1} + b_{o}\right )$$
	$$h_{t}= o_{t} \odot\tanh(c_{t})$$
		> 현재 시점의 셀 상태의 값이 하이퍼볼릭탄젠트 함수를 지나 -1과 1사이의 값이 되고, 해당 값은 출력 게이트의 값 $o_{t}$ 와 연산되면서, 값이 걸러지는 효과가 발생하여 은닉 상태 (단기상태) $h_{t}$가 된다. 단기 상태의 값은 또한 출력층으로도 향한다.

## 3 . Seq2Seq Model
-  seq2seq은 번역기에서 주로 사용하는 모델.
 
   ![](https://wikidocs.net/images/page/24996/%EC%8B%9C%ED%80%80%EC%8A%A4%ED%88%AC%EC%8B%9C%ED%80%80%EC%8A%A4.PNG)
	
- 	seq2seq 은 **Encoder**와 **Decoder**로 구성.
- Encoder는 입력문장을 모두 입력받은 후, 이 모든 단어의 정보를 압축하여 하나의 **context vector***로 만듬.
- Decoder로 전송된 context vector는 디코더에서 번역하여 단어 한개씩 순차적 출력.

  ![enter image description here](https://mblogthumb-phinf.pstatic.net/MjAyMDAxMjVfMTgz/MDAxNTc5ODgzODgxOTY4.9mZGRV9J_yzovNJzQ5gI03lyJvtUSlq2hDVkOaZB_Q0g.z_GZNj_cL2WJkk0YcwWSPFwSv0lb-FPzsGPL2GUzyvEg.PNG.sooftware/image.png?type=w800)

- decoder는 기본적으로 RNN Langauge Model.
- Decoder의 첫 입력으로 문장의 시작을 의미하는 심볼( < s >, < sos >) 이 들어가고 , Decoder의 첫번째 RNN 셀은 context vector와 < s > 두 개의 입력을 바탕으로 새로운 hidden state $h_{1}$ 계산 후, Affine(feed forward) 계층과 softmax계층을 거쳐 다음 등장할 확률이 높은 단어('안녕하세요')를 예측.
- 다음 RNN셀은 전 timestep에서 계산된 $h_{1}$과 예측한 단어('안녕하세요')를 입력으로 2번째 예측 수행.
- 문장의 끝을 의미하는 symbol(< /s >, < eos > 등) 이 예측될 때까지 반복.

## 4. Attention Mechanism
 - Seq2Seq의 한계점
	 1. encoder에서 하나의 고정된 크기 벡터에 정보를 압축하여 정보 손실 발생.
	 2.  RNN의 고질적 문제인 Vanishing Gradient Problem.
	 
- Attention mechanism ?
	seq2seq 구조에서 Encoder에서 계산한 여러 Hiddenstate 중 마지막 Hiddenstate만을 Decoder에서 사용되지 않는데 , **사용되지 않은 Hidden State**를 이용한다.
	즉, Decoder에서 출력을 예측하는 매 timestep 마다 Encoder의 Hidden state를 다시 한 번 참조하는데,
	해당 timestep에서 예측해야하는 결과와 연관있는 부분을 판단하여 비율을 더 **집중** 시킨다.
> 어느 인코더의 **Hidden State**를 얼마나 참고할지 결정하는 것이 핵심!
- **Dot-product Attention** (가장 기본적 Attention)의 Step
 1. 현 시점 디코더의 Hidden State와 인코더의 모든 Hidden state들과 각각 내적(dot product) 수행, 그 결과는 scalar 값 => Attention Score(어텐션 스코어)   
 
![enter image description here](https://blogfiles.pstatic.net/MjAyMDAxMjVfODYg/MDAxNTc5ODg3OTY2MjI2.Z7Lu2gOy8dU_B143Iyzs-yvQVIknQyHpfAwfXJYrsLEg.hZle2mnoLTKWAhapEiE9nCQisfUe4PbzxRhODi8Vdtsg.PNG.sooftware/image.png)
 2. 그렇게 나온 scalar에 대해 softmax 함수 적용하여 attention 의 분포(각 encoder hidden state의 중요도) 구함.

![enter image description here](https://blogfiles.pstatic.net/MjAyMDAxMjVfMTIg/MDAxNTc5ODg4NDkxMDQz.miNLNdmdj0t3Ll12purypbOIE6PWRFijlxAF4ci5K28g.c-UT98v0QJumGmehmlwGkQ0bQxxV_jCKOCjOVH17ZcYg.PNG.sooftware/image.png)
	 (예) e0 : 2%, e0 : 18%, e0 : 50%, e0 : 25%, e0 : 5%

3. attention의 분포를 각 encoder의 hidden state와 곱해준다.
 ![enter image description here](https://blogfiles.pstatic.net/MjAyMDAxMjVfMjAy/MDAxNTc5ODg5MDU0ODE0.4Xwd762CrmUZ6sWY_IFoddKsPOrDm4v55c-b0JU-PXEg.cFn2Pmagh6s_lLgfXbkPdY2_O4gCOLqSKQf3yHwds6Mg.PNG.sooftware/image.png)

4. 3번에서 곱을 통해 얻어진 hidden state들을 전부 더해준 것 => Context Vector(컨텍스트 벡터)
5. 4에서 구한 Context vector와 현 시점 decoder의 hidden state와 연결(concatenate)하여 벡터 구함.
6.  5에서 구한 벡터를 바탕으로 최종 값 계산.

  ![enter image description here](https://blogfiles.pstatic.net/MjAyMDAxMjVfMTE5/MDAxNTc5ODg3ODE3MjE2.qoAGM2HdLouYzpNv8zjR9tjKAatY1h_MyOh6KLWCN70g.XhC_2dkTliEKeD4DiDAe6HqBV_rVnjqhQbXs3pYxf2wg.PNG.sooftware/image.png)

- 다양한 종류의 Attention 
	 - Attention Score을 계산하는 방식(1번 단계)에서 차이가 있음!
	 
![enter image description here](https://postfiles.pstatic.net/MjAyMDAxMjVfMjM2/MDAxNTc5ODg5NjkzMjAy.-nH6VxdzQQU4ZSQY-09AVaq_8WKfy2Ox1LhpsQeuQ0Ag.jTicYxSxPvM6AeN1AGEHjuPCc5uOAOP-tp7qS18isoog.PNG.sooftware/image.png?type=w773)
	

> 참조 :[seq2seq Model](https://m.blog.naver.com/PostView.nhn?blogId=sooftware&logNo=221784419691&proxyReferer=https://www.google.com/)
, [LSTM](https://wikidocs.net/45101)
[Attention mechanism 출처](https://blog.naver.com/sooftware/221784472231)
