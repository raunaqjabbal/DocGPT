# DocGPT


## How to install for Windows

In a folder that contains contents of this Repository: 

```
pip install poetry
```

```
git clone https://github.com/cofactoryai/textbase
cd textbase
poetry shell
poetry install
```

Enter the virtual environment

``` 
cd ..
pip install -r requirements.txt
```

Run this in a python script:
```
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name= "thenlper/gte-small"

encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    # model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)

import pickle

with open('gpt/doc_embedding.pickle', 'wb') as pkl:
    pickle.dump(embedding, pkl)

```
