# NLP
 ## 1.  자연어  vs  인공어 
- 자연어 : 일상적으로 사용 하는 언어
	ex) 러시아어, 한국어, 일본어 등
-  인공어 : 자연어와 구분하여 사람의 의도, 목적으로 만들어진 언어
	ex) 프로그래밍 언어, 은어, 소설의 새로운 언어
## 2. 자연어 처리
-  자연어의 의미를 분석해서 컴퓨터로 다양한 문제를 해결하는 것.
-  수신받은 메세지를 컴퓨터가 해독(Decode) 하는 것
## 3 . 자연어 처리 과정
1.  Preprocessing - 전처리
		개행문자, 특수문자, 중복표현, 불용어(의미없는용어) 제거 
2.  Tokenizaing
		  자연어를 어떤 단위( 어절, 형태소 등) 로 살펴볼 것인가 
3.  Lexical analysis
4.  Syntax Analysis - 구문분석
		 문장을 구성하기 위한 문법, 규칙 구성
5.  Semantic Analysis - 의미분석
		 문법 규칙은 맞지만, 의미가 옳지 않은 것 찾아냄.
 > 자연어 처리는 '분류' 의 문제이다. 

## 4. Word Embeddings
 - word embedding ? 
	-  단어의 특징에 맞춰 단어를 n차원 벡터로 mapping 하는 것
	-   관련성 높은 주변 단어를 통해 의미적 정보를 효율적으로 계산하여 단어별 특징을 추출
	  1. Spare Representation(one - hot encoding) 
		  - 1, 0 으로 표현하여 컴퓨터가 이해할 수 있도록 하는 방식
		  -  각 단어마다 한 차원으로 표현하기에 저장 공간이 매우 비효율적이며 단어의 유사정도를 나타내지 못한다는 문제.
		  - 그것을 해결하는 것이 아래의 Dense representation.
		  -  (예)
			  cat => [ 1, 0, 0, 0, .....0]
			  dog=> [0, 1, 0,0, ......0 ]
			  pig => [0, 0, 1, 0 ......0 ]
	 2. Dense Representation  ( Distributed representation) 
		 - 각 vector의 값들이 특정한(유의미한) 실수값을 가지도록 표현.
		 -  벡터가 유사하면 유사한 속성을 가진다고 생각.
		 - 벡터 연산을 통해 단어에 대한 추론 가능
		 -  (ex 1)
			 cat => [ 0.37, 0.23, 0.07, 0.32, .....0.54]
			 dog=> [0.16,-0.05, 0.2,-0.11, ......0.15 ]
		- 	 (ex 2)
			'한국' - '서울' + '도쿄' = '일본' 
			 
	3. Word2vec( word to vector) 알고리즘
	 - 자연어의 의미를 벡터 공간에 임베딩 할 수 있도록 만든 알고리즘.
	 - 2가지 방식
		1. CBOW(  continous bag of words) : 주변 단어(context)를 통해 단어(target word)를 예측하는 모델
		2. Skip- gram model : CBOW와 방향만 반대.

## 5. Word 2 vec Algorithm
### CBOW model


![enter image description here](https://shuuki4.files.wordpress.com/2016/01/cbow.png?w=400&h=200)
 
> Input layer, Hidden layer, Output layer로 구분되어 있는데, input Layer에서 hidden Layer로 갈 때, 모든 단어들( 1XV 크기의 벡터) 이 공통적으로 사용하는 V X N크기의 Matrix W가 있고 , output layer로 갈 때는 NXV 크기의 Matrix W' 가 있다.  Input에서는 단어를 one-hot encoding으로 입력해준 후, 여러개의 단어에 각각 Weight를 곱하고 그 벡터들의 평균을 구해 Hidden Layer로 보낸다. 여기에 W' 를 곱해 output layer로 보낸 후, softmax 계산 하여 진짜 단어의 one - hot encoding과 비교하여 에러를 계산한다.

##  6. 순환 신경망(RNN, Recurrent Neural Network)
- Feed Forward 신경망은 은닉층에서 활성화 함수를 지난 값은 오직 출력층 방향으로만 향함.
- 그러나 RNN은 은닉층 노드(연두색 네모)의 활성화 함수를 통해 나온 결과값을 출력층 방향과 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 가짐.

 ![enter image description here](https://wikidocs.net/images/page/22886/rnn_image2_ver3.PNG)

- RNN에 대한 수식 정의
![enter image description here](https://wikidocs.net/images/page/22886/rnn_image4_ver2.PNG)

> 설명참조 : [RNN 신경망 설명참조](https://wikidocs.net/22886)
