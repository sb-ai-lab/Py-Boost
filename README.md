# Py-boost: a research tool for exploring GBDTs

Modern gradient boosting toolkits are very complex and are written in low-level programming languages. As a result,

* It is hard to customize them to suit one’s needs 
* New ideas and methods are not easy to implement
* It is difficult to understand how they work

Py-boost is a Python-based gradient boosting library which aims at overcoming the aforementioned problems. 

**Authors**: [Anton Vakhrushev](https://kaggle.com/btbpanda), [Leonid Iosipoi](http://iosipoi.com/).


## Py-boost Key Features

**Simple**. Py-boost is a simplified gradient boosting library but it supports all main features and hyperparameters available in other implementations.

**Fast with GPU**. Despite the fact that Py-boost is written in Python, it works only on GPU and uses Python GPU libraries such as CuPy and Numba.

**Easy to customize**. Py-boost can be easily customized even if one is not familiar with GPU programming (just replace np with cp).  What can be customized? Almost everuthing via custom callbacks. Examples: Row/Col sampling strategy, Training control, Losses/metrics, Multioutput handling strategy, Anything via custom callbacks


## Installation

Before installing py-boost via pip you should have cupy installed. You can use:

`pip install -U cupy-cuda110 py-boost`

**Note**: replace with your cuda version! For the details see [this guide](https://docs.cupy.dev/en/stable/install.html)


## Quick tour

Py-boost is easy to use since it has similar to scikit-learn interface. For usage example please see:

* [Tutorial_1_Basics](https://github.com/sberbank-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_1_Basics.ipynb) for simple usage examples
* [Tutorial_2_Advanced_multioutput](https://github.com/sberbank-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_2_Advanced_multioutput.ipynb) for advanced multioutput features
* [Tutorial_3_Custom_features](https://github.com/sberbank-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_3_Custom_features.ipynb) for examples of customization

More examples are comming soon



## Other Sber AI Lab Projects
LightAutoML: https://github.com/sberbank-ai-lab/LightAutoML  
AutoWoE: https://github.com/sberbank-ai-lab/AutoMLWhitebox  
RePlay: https://github.com/sberbank-ai-lab/RePlay  


