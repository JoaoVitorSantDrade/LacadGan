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

 Perguntas:

 Como treinar uma GAN com dataset de tamanho limitado? - Feito
 StyleGAN funciona p/ gerar glomérulos diferentes?

 To Do:
 
 Implementação de Feature Matching
 Implementar Feature learning (Fazer a GAN diferenciar entre os tipos de imagem gerada)
 Implementação de Multi Scale Gradient - Paper
 Adicionar Ruido nas imagens reais e sintéticas (Generator) para melhorar estabilidade das imagens geradas - Paper
 Implementação da ADAIN (Adapative Instance Normalization)
 Implementação do Restart Learning rate do descriminador - Paper


https://www.casualganpapers.com/page/6/
https://github.com/LACAD/gan-amiloisose.git