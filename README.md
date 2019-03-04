# How to use
* [download](https://pan.baidu.com/s/1B3x5BN5Nm1Um6yq1eJU6qA) model file, pwd: 747l
* just use like this 
    * first
        ````python 
        bn = BertNer(gpu_no=0, log_dir='log/', verbose=True, ner_model=r'bert_ner_model\\')
    * second
         ````python 
        bn.predict(['小张'])
    * third 
        > you will get result like this [[['张', 'PER']]], every element of list is ner of content
     
# Parameter
| name | type | detail |
|--------------------|------|-------------|
gpu_no | int | which gpu will be use to init bert ner graph
log_dir | str | log dir 
verbose | bool| whether show tensorflow log
ner_model | str| bert ner model path