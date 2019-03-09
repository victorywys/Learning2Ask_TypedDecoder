# Learning2Ask_TypedDecoder
---
The code for the paper [Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders](https://arxiv.org/pdf/1805.04843.pdf) on ACL2018

Cite this paper:
```
@inproceedings{wang2018learning,
  title={Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders},
  author={Wang, Yansen and Liu, Chenyi and Huang, Minlie and Nie, Liqiang},
  booktitle={ACL},
  year={2018}
}
```

You can download our data [here](http://coai.cs.tsinghua.edu.cn/hml/dataset/). Press ctrl+f and search for "Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders" and you can find the link to our dataset.

## Usage:
**IMPORTANT NOTE: Our code is not compatible with new versions of tensorflow, so please use tensorflow-1.0.0 to run our codes.**

### STD:
Command ```python main.py {--[option1]=[value1] --[option2]=[value2] ... }```

Options(=[default_value]):
* ```--is_train=True``` Set to True for training and False from inference.
* ```--symbols=20000``` Size of vocabulary.
* ```--embed_units=100``` Size of word embedding.
* ```--units=512``` Size of each model layer.
* ```--layers=4``` Number of layers in the model.
* ```--batch_size=50``` Batch size to use during training. 
* ```--data_dir=./data``` Data directory.
* ```--train_dir=./train``` Training directory.
* ```--per_checkpoint=1000``` How many steps to do per checkpoint.
* ```--check_version=0``` The version for continuing training or for inferencing. Set to 0 if you don't want to continue from an existed checkpoint.
* ```--log_parameters=True``` Set to True to show the parameters.
* ```--inference_path=""``` Set filename of inference, empty for screen input.
* ```--PMI_path=./PMI``` PMI directory.
* ```--keywords_per_sentence=20``` How many keywords will be included. We don't need to set this flag in STD.
* ```--question_data=True``` **(Deprecated, please set to True)** An unused option in the final version.

The file ```train.sh``` and ```infer.sh``` contain example commands for training and inference. You can use them with the ```sh``` command.

### HTD:
Command ```python main.py {--[option1]=[value1] --[option2]=[value2] ... }```

Options(=[default_value]):
* ```--is_train=True``` Set to True for training and False from inference.
* ```--symbols=20000``` Size of vocabulary.
* ```--embed_units=100``` Size of word embedding.
* ```--units=512``` Size of each model layer.
* ```--layers=4``` Number of layers in the model.
* ```--batch_size=50``` Batch size to use during training. **Please set to 1 during inference or the PMI mechanism can't work properly.**
* ```--data_dir=./data``` Data directory.
* ```--train_dir=./train``` Training directory.
* ```--per_checkpoint=1000``` How many steps to do per checkpoint.
* ```--check_version=0``` The version for continuing training or for inferencing.
* ```--log_parameters=True``` Set to True to show the parameters.
* ```--inference_path=""``` Set filename of inference, empty for screen input.
* ```--PMI_path=./PMI``` PMI directory.
* ```--keywords_per_sentence=20``` How many keywords will be included.
* ```--question_data=True``` **(Deprecated, please set to True)** An unused option in the final version.

The file ```train.sh``` and ```infer.sh``` contain example commands for training and inference. You can use them with the ```sh``` command.

### FAQ
##### Where can I find "vector.txt"?
We're sorry that due to our regulations, we can't share the word vectors pretrained. You can make your own "vector.txt" in this format:
```
[word1] 1.0 -2.0 5.0
[word2] 3.14 2.72 -1.41
[word3] 0.86 -1.71 -0.04
... ...
```
and set ```--embed_units==[vector dimension]```. In this case, you should set ```--embed_units==3```

Here's part of our word vectors:
```
冉津 -0.007428 -0.018109 0.017502 0.127934 0.090787 -0.008699 -0.181448 -0.117719 -0.130669 0.007109 -0.048784 -0.083871 -0.041926 -0.016476 0.026685 -0.094259 -0.097639 0.049795 0.077781 -0.027308 -0.000205 0.117830 -0.033821 -0.088984 0.150127 -0.065157 0.018675 -0.105137 0.001134 -0.026754 0.026742 -0.127951 -0.006684 -0.080394 0.003453 -0.031691 -0.013896 0.051936 0.034658 0.079686 0.026027 0.130313 0.011976 -0.154662 -0.065610 0.079444 -0.036182 -0.042820 0.040647 -0.009277 -0.094344 0.352311 -0.100773 -0.167505 -0.071562 0.182705 0.087977 -0.077308 0.121469 -0.076466 0.045806 0.029080 -0.120310 0.112574 0.027545 0.130245 0.060847 -0.087550 -0.072264 -0.061106 0.045996 -0.048654 0.036791 -0.324380 -0.129975 -0.151802 0.055080 0.108745 0.072554 0.063584 -0.183879 -0.088556 -0.189840 -0.028041 -0.130920 -0.110319 -0.043854 -0.124681 0.027615 -0.096786 0.024738 -0.112449 -0.041501 -0.016814 -0.026927 0.213262 0.127977 -0.085883 -0.056919 0.074451
冉徽 0.014493 -0.009604 -0.056103 0.137076 0.136810 0.003288 -0.162282 -0.142987 -0.111230 -0.007172 -0.036456 -0.059875 -0.034977 -0.000799 0.010098 -0.087427 -0.089052 0.052306 0.095106 -0.078993 -0.038151 0.072410 -0.069268 -0.057892 0.117272 -0.029470 0.013380 -0.051824 -0.039586 -0.041293 0.059040 -0.148370 -0.015987 -0.074139 0.048661 -0.056333 0.022390 0.077231 -0.010541 0.071275 0.015923 0.151031 0.013858 -0.166912 -0.053901 0.057671 -0.070033 -0.044730 0.011594 0.016944 -0.148096 0.327251 -0.109722 -0.195073 -0.074526 0.209270 0.096594 -0.008418 0.120976 -0.057380 0.039540 0.050772 -0.150347 0.127315 -0.023129 0.164845 0.086893 -0.053719 -0.042148 -0.030370 0.064161 -0.070620 0.031359 -0.297059 -0.092481 -0.101616 0.105090 0.139352 0.058642 0.080823 -0.226540 -0.081144 -0.161620 -0.055791 -0.109781 -0.082259 -0.023754 -0.115139 0.023207 -0.117227 0.025099 -0.098476 -0.039537 0.056101 0.011074 0.201935 0.127134 -0.081476 -0.025416 0.024106
冉红平 0.001560 -0.005889 0.025941 0.063590 0.079942 0.007259 -0.176020 -0.105751 -0.107272 0.005988 -0.078503 -0.030769 -0.029349 -0.039878 0.007160 -0.075574 -0.121881 0.030458 0.070573 -0.030429 -0.009549 0.063056 -0.024280 -0.122451 0.073607 -0.017913 0.002592 -0.099109 0.039369 -0.054562 0.044947 -0.135777 -0.023722 -0.065398 0.039630 -0.058899 -0.034931 0.051255 0.051398 0.016336 -0.003559 0.133971 0.088922 -0.220131 0.006107 0.022170 -0.056472 -0.061360 0.019423 0.018444 -0.161037 0.362732 -0.118108 -0.157995 -0.071416 0.118341 0.083489 -0.036985 0.103561 -0.086170 0.029961 0.045517 -0.165905 0.122532 0.004158 0.116590 0.024232 -0.052038 -0.053199 -0.038042 0.089462 -0.087992 0.033044 -0.303940 -0.160299 -0.150656 0.062613 0.156578 0.015454 0.124571 -0.198247 -0.060708 -0.183756 -0.080255 -0.135093 -0.155833 0.029361 -0.091097 0.032860 -0.103119 0.099081 -0.114630 -0.045474 0.023771 -0.044274 0.185795 0.088140 -0.072055 -0.031876 0.074563
......
```

If you do not have pretrained vectors, just leave a blank ```vector.txt``` file for the program to load. The program will automatically initialize all the word vectors for those not appeared in this file.
