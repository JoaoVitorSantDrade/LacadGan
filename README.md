# ProGan
 Implementação de uma Progressive GAN

Feitos:
Salvamento da Model

Corrigido erro ao salvar imagem por tamanho (Ficava desformatado)

Implementação de TTUR (Two Time Scale Update Rule) - Paper (FEITO - 15/08/2022)

Implementação de Minibatch Descrimination - (FEITO - 20/07/22)

Differentiable Augmentation - (FEITO - 26/09/2022)

Correção do DifAugmentation - (FEITO - 20/10/2022)

Criação do dockerfile e configuração - (FEITO - 21/11/2022)

Adicionar verificação antes de tentar salvar/ler arquivo (Evitar jogar uma exception) - (FEITO - 21/11/2022)

Salvar Epoch e Step em binário - (FEITO - 21/11/2022)

Correção do UTILs.py - (FEITO - 23/11/2022)

Adicionar Ruido nas imagens reais e sintéticas (Generator) para melhorar estabilidade das imagens geradas - Paper

Implementação da ADAIN (Adapative Instance Normalization)

Implementação do Restart Learning rate do descriminador - Paper - Começo em 01/02/23

Implementação do Stochastic Weight Averaging (SWA) - (FEITO - 18/04/2023) 

Resolvido problema com o datasetAugmentation.py - (FEITO - 19/04/2023)
 Perguntas:

 Como treinar uma GAN com dataset de tamanho limitado? - Feito
 StyleGAN funciona p/ gerar glomérulos diferentes?

 To Do:
 
 Gradient Acumulation - Não funcionou (18/04/2023)
 Gradient Checkpointing
 Implementação de Feature Matching
 Implementar Feature learning (Fazer a GAN diferenciar entre os tipos de imagem gerada)
 Implementação de Multi Scale Gradient - Paper


Observações:
Quanto maior o learning rate, maiores são as features que ele captura. Irei testar com um learning rate muito baixo e mais épocas, assim talvez ele consiga pegar as features menores das células

https://www.casualganpapers.com/page/6/
https://github.com/LACAD/gan-amiloisose.git
PhD thesis, chapter 3: https://doi.org/10.17863/CAM.53748.

https://arxiv.org/pdf/2102.06108.pdf
https://arxiv.org/pdf/2103.11093.pdf
https://arxiv.org/pdf/2210.09655.pdf
https://arxiv.org/pdf/1812.04948.pdf
https://arxiv.org/pdf/1704.00028.pdf
https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
https://arxiv.org/pdf/2103.04922.pdf
https://arxiv.org/pdf/2102.07074.pdf
https://arxiv.org/pdf/1710.10196.pdf
https://arxiv.org/pdf/1904.03189.pdf
https://ieeexplore.ieee.org/document/8721631
https://torchmetrics.readthedocs.io/en/stable/image/kernel_inception_distance.html
Wavelet Loss
https://arxiv.org/pdf/2209.02316.pdf

OLHAR MSG-GAN