# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding
from negative_sampling_layer import NegativeSamplingLoss


class HCBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus, hop_window_size, hop_count, hop_decay):
        V, H = vocab_size, hidden_size
        ### 추가된 params
        self.hop_count = hop_count
        self.window_size = window_size
        self.hop_decay = hop_decay
        self.hop_window_size = hop_window_size
        self.count_decay = sum((self.hop_decay**n) * self.hop_window_size for n in range(1, self.hop_count + 1))
        ### 추가된 params (end)

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V + 1, H).astype('f') # 0벡터를 위한 열 하나 추가
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # 계층 생성
        self.in_layers = []
        for i in range(2 * window_size + hop_count * hop_window_size):
            layer = Embedding(W_in)  # Embedding 계층 사용
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 모든 가중치와 기울기를 배열에 모은다.
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        ### decay 구현
        decay = self.hop_decay
        for i, layer in enumerate(self.in_layers):
            if i < (2 * self.window_size):
                h += layer.forward(contexts[:, i])
            else:
                h += layer.forward(contexts[:, i]) * decay
                check = i - (2 * self.window_size)
                if check != 0 and check % self.hop_window_size == 0:
                    decay **= decay 
        h *= 1 / (2 * self.window_size + self.count_decay)
        ### decay 구현 (end)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        ### decay 고려
        decay = self.hop_decay
        for i, layer in enumerate(self.in_layers):
            if i < (2 * self.window_size):
                dout *= 1 / (2 * self.window_size + self.count_decay)
            else:
                dout *= decay / (2 * self.window_size + self.count_decay)
                check = i - (2 * self.window_size)
                if check != 0 and check % self.hop_window_size == 0:
                    decay **= decay
            layer.backward(dout)
        ### decay 고려 (end)
        return None
