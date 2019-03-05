# How to use
* [download](https://pan.baidu.com/s/1B3x5BN5Nm1Um6yq1eJU6qA) model file, pwd: 747l
* just use like this 
    * first
        ````python 
        bn = BertNer(gpu_no=0, log_dir='log/', verbose=True, ner_model=r'bert_ner_model\\')
    * second
         ````python 
         bn.predict(['两年前，来自上海的“高龄产妇”周月（化名），在香港顺产生下了一个活泼可爱的女儿', '湖北农民万其珍应下叔叔万述荣的临终嘱托，成为万家第三代义渡艄公'])
    * third 
        > you will get result like this [[['上海', 'LOC'], ['周月', 'PER'], ['香港', 'LOC']], [['湖北', 'LOC'], ['万其珍', 'PER'], ['万述荣', 'PER']]], every element of list is ner of content
     
# Parameter
| name | type | detail |
|--------------------|------|-------------|
gpu_no | int | which gpu will be use to init bert ner graph
log_dir | str | log dir 
verbose | bool| whether show tensorflow log
ner_model | str| bert ner model path