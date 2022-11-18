# RACER: Co-speech Gesture Synthesis by Reinforcement Learning with Contrastive Pre-trained Reward

The video demo for “Co-speech Gesture Synthesis by Reinforcement Learning with Contrastive Pre-trained Reward (CVPR2023 OpenReview)” is available at: 

The character is from Mixamo (<https://www.mixamo.com>).

> There is a growing demand of automatically synthesizing co-speech gestures for virtual characters. However, it remains a challenge due to the complex relation between input speeches and target gestures. Existing works focus on learning a mapping from speeches to gestures in a supervised manner, ignoring the fact that co-speech gesture synthesis is a sequential decision making problem by nature. In this paper, we propose a novel reinforcement learning (RL) framework called **RACER** to generate sequences of gestures that maximize the overall satisfactory. **RACER** employs a vector quantized variational autoencoder to learn compact representations of gestures and a GPT-based policy architecture to generate coherent sequence of gestures autoregressively. In particular, we propose a contrastive pre-training approach to calculate the rewards, which integrates contextual information into action evaluation and successfully captures the complex relationships between multi-modal speech-gesture data. To our knowledge, **RACER** is the first co-speech gesture synthesizing frameworsk that is trained solely by RL. Experimental results show that our method significantly outperforms existing baselines in terms of both objective metrics and subjective human judgements.


## Introduction
RACER is a novel RL based approach to learn the optimal gesture synthesis policy, which models the co-speech gesture synthesis problem as a Markov decision process. The overview of our proposed framework RACER is shown in:



!(https://github.com/RLracer/RACER/blob/main/Module/Overview.pdf)


Given a piece of speech audio and the initial gesture code $a_0$, the Q-network represented by transformer layers autoregressively calculates the Q-values and selects a sequence of actions ($a_1,\cdots,a_T$). The action sequence will then be transformed to quantitative features by querying the codebook and finally be decoded to motion sequences by the decoder of VQ-VAE.







At each time step $t$, the state $s$ consists of the generated action tokens $(a_1,\dots,a_{t-1})$ and input audio. 
Unlike existing methods which directly learn a mapping from audio features to the continuous high-dimensional motion space, RACER encodes and quantizes the motion into a finite codebook $\mathcal{Z} =\{\bm{z_i}\}^{N}_{i=1} $ by VQ-VAE, where $N$ is the size of codebook and each code $\bm{z_i}$ represents a gesture lexeme feature. The details of action design are introduced in . 
We use a GPT-like unidirectional model as the Q-network that autoregressively outputs action tokens following a greedy strategy. An action token $a$ will be mapped to a gesture lexeme feature $z$ and then be decoded to a specific gesture motion. Moreover, we propose a contrastive speech-gesture pre-training model to compute the immediate rewards for the actions, which will be elaborated in . 

In addition, we will introduce how to train the Q-network in a fully offline manner in \cref{sec:approachOffline}.

|    | **Residual Block**  |
|  ---  | :----:  |
|  | Input: **0**; Argument: p, d|
|**1** | ReLU, Conv(512, 512, 3, 1, p, d)|
|**2** |ReLU, Conv(512, 512, 1, 1, 0, 1)|
|   |Output: **0** + **2**|

|    | **VQ-VAE Encoder**  |
|  ---  | :----:  |
|  | **Input: 0; Argument:** *J*|
|**1** |Conv(*J* × 3, 512, 4, 2, 1, 1)|
|**2**|**RB**(p = 1, d = 1)|
|**4** |**RB**(p = 3, d = 3)|
|**3** |Conv(512, 512, 4, 2, 1, 1)|
|**4** |**RB**(p = 3, d = 3)|
|**5** |Conv(512, 512, 4, 2, 1, 1)|
|**6** |**RB**(p = 9, d = 9)|
|**7** |Conv(512, 512, 3, 1, 1, 1)|
|   |Output: **7**|

|    | **VQ-VAE Decoder**  |
|  ---  | :----:  |
|  | **Input: 0; Argument:** *J*|
|**1** |Conv(512, 512, 3, 1, 1, 1)|
|**6** |**RB**(p = 9, d = 9)|
|**2** |Conv(512, 512, 3, 1, 1, 1)|
|**2**|**RB**(p = 3, d = 3)|
|**3** |Conv(512, 512, 4, 2, 1, 1)|
|**4** |**RB**(p = 1, d = 3)|
|**5** |Conv(512, 512, 4, 2, 1, 1)|
|   |Output: **7** |


