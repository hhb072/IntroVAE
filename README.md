# IntroVAE
A pytorch implementation of Paper ["IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis"](http://papers.nips.cc/paper/7291-introvae-introspective-variational-autoencoders-for-photographic-image-synthesis)

## Prerequisites
* Python 2.7 or Python 3.6
* PyTorch

## Run

Use the default parameters except changing the hyper-parameter, i.e., *m_plus*, *weight_rec*, *weight_kl*, and *weight_neg*, for different image resolution settings. Noted that setting *num_vae* nonzero means pretraining the model in the standard VAE manner, which may helps improve the training stablitity and convergency. 

The default parameters for CelebA-HQ faces at 256x256 and 1024x1024 resolutions are provided in the file 'run_256.sh' and 'run_1024.sh', respectively. Other settings are allowed as discussed in the appendix of the published paper. 

## Results

![](https://github.com/hhb072/IntroVAE/blob/master/sample.jpg)

## Citation

If you use our codes, please cite the following paper:

 	@inproceedings{huang2018introvae,
	  title={IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis},
	  author={Huang, Huaibo and Li, Zhihang and He, Ran and Sun, Zhenan and Tan, Tieniu},
	  booktitle={Advances in Neural Information Processing Systems},
	  pages={10236--10245},    
	  year={2018}
	}

**The released codes are only allowed for non-commercial use.**
