# IntroVAE
A pytorch implementation of Paper ["IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis"](http://papers.nips.cc/paper/7291-introvae-introspective-variational-autoencoders-for-photographic-image-synthesis)

## Prerequisites
* Python 2.7 or Python 3.6
* PyTorch

## Run

Use the default parameters except changing the hyper-parameter, i.e., *m_plus*, *weight_rec*, *weight_kl*, and *weight_neg*, for different image resolution settings. Noted that setting *num_vae* nonzero means pretraining the model in the standard VAE manner, which may helps improve the trainging stablitity and convergency. 

The default parameters for CelebA-HQ faces at 256x256 and 1024x1024 resolutions are provided in the file 'run_256.sh' and 'run_1024.sh', respectively. Other setting are allowed as discussed in the appendix of the published paper. 

## Results
