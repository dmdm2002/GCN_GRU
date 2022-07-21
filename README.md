# GCN_GRU

1. original code and paper
code : https://gitlab.com/hackshields/rna-paper
paper : https://www.sciencedirect.com/science/article/pii/S1319157821002871

2. model
sequence data를 GCN에 input으로 학습한 후 aggregator로는 본 논문에서 제시한 방법중 Conv를 채택하여 학습을 진행한다.
그이후 GRU를 이용하여 학습을 진행하게 된다.
