# Co-speech Gesture Synthesis by Reinforcement Learning with Contrastive Pre-trained Reward

Video demo for Co-speech Gesture Synthesis by Reinforcement Learning with Contrastive Pre-trained Reward, CVPR2023 OpenReview

The character is from Mixamo (<https://www.mixamo.com>).

>
There is a growing demand of automatically synthesizing co-speech gestures for virtual characters. However, it remains a challenge due to the complex relation between input speeches and target gestures. Existing works focus on learning a mapping from speeches to gestures in a supervised manner, ignoring the fact that co-speech gesture synthesis is a sequential decision making problem by nature. In this paper, we propose a novel reinforcement learning (RL) framework called RACER to generate sequences of gestures that maximize the overall satisfactory. RACER employs a vector quantized variational autoencoder to learn compact representations of gestures and a GPT-based policy architecture to generate coherent sequence of gestures autoregressively. In particular, we propose a contrastive pre-training approach to calculate the rewards, which integrates contextual information into action evaluation and successfully captures the complex relationships between multi-modal speech-gesture data. To our knowledge, RACER is the first co-speech gesture synthesizing framework that is trained solely by RL. Experimental results show that our method significantly outperforms existing baselines in terms of both objective metrics and subjective human judgements.
