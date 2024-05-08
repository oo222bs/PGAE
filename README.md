# Paired Gated Autoencoders

[Learning Flexible Translation Between Robot Actions and Language Descriptions](https://link.springer.com/chapter/10.1007/978-3-031-15931-2_21)

Last updated: 8 May 2024.

Copyright (c) 2022, Ozan Özdemir <<ozan.oezdemir@uni-hamburg.de>>

## Requirements
- Python 3
- Pytorch
- NumPy
- Tensorboard

## Implementation
PGAE - Pytorch Implementation

## Training Example
```
$ cd src
$ python main_pgae.py
```
- main_pgae.py: trains the PGAE model
- pgae.py: defines the PGAE architecture
- channel_separated_cae: defines the channel separated CAE
- standard_cae: defines the standard CAE
- config.py: training and network configurations
- data_util.py: for reading the data
- generation.py: translates instructions to actions
- recognition.py: translates actions to descriptions
- extraction.py: extracts shared representations
- reproduction.py: reproduces the actions
- lang2lang.py: reproduces the descriptions
- inference.py: tests the model on the self actions
- inference_opp: tests the model on the opposite-robot actions

## Citation

**PGAE**
```bibtex
@Article{OKWLW22a, 
 	 author =  {Özdemir, Ozan and Kerzel, Matthias and Weber, Cornelius and Lee, Jae Hee and Wermter, Stefan},  
 	 title = {Learning Flexible Translation between Robot Actions and Language Descriptions}, 
 	 journal = {Artificial Neural Networks and Machine Learning – ICANN 2022},
 	 number = {},
 	 volume = {},
 	 pages = {246--257},
 	 year = {2022},
 	 month = {Sep},
 	 publisher = {Springer International Publishing},
 	 doi = {10.1007/978-3-031-15931-2_21}, 
 }
```
