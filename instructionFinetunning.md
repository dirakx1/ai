```python
import itertools
import jsonlines

from datasets import load_dataset
from pprint import pprint

from llama import BasicModelRunner
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[1], line 2
          1 import itertools
    ----> 2 import jsonlines
          4 from datasets import load_dataset
          5 from pprint import pprint


    ModuleNotFoundError: No module named 'jsonlines'



```python
pip install jsonlines
```

    Collecting jsonlines
      Downloading jsonlines-4.0.0-py3-none-any.whl (8.7 kB)
    Requirement already satisfied: attrs>=19.2.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from jsonlines) (22.1.0)
    Installing collected packages: jsonlines
    Successfully installed jsonlines-4.0.0
    Note: you may need to restart the kernel to use updated packages.



```python
import itertools
import jsonlines

from datasets import load_dataset
from pprint import pprint

from llama import BasicModelRunner
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[1], line 4
          1 import itertools
          2 import jsonlines
    ----> 4 from datasets import load_dataset
          5 from pprint import pprint
          7 from llama import BasicModelRunner


    ModuleNotFoundError: No module named 'datasets'



```python
pip install datasets
```

    Collecting datasets
      Downloading datasets-2.18.0-py3-none-any.whl (510 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m510.5/510.5 kB[0m [31m1.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting aiohttp
      Downloading aiohttp-3.9.3-cp310-cp310-macosx_10_9_x86_64.whl (397 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m397.9/397.9 kB[0m [31m6.9 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hRequirement already satisfied: dill<0.3.9,>=0.3.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from datasets) (0.3.6)
    Collecting pyarrow>=12.0.0
      Downloading pyarrow-15.0.2-cp310-cp310-macosx_10_15_x86_64.whl (27.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m27.2/27.2 MB[0m [31m29.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: pandas in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from datasets) (1.5.3)
    Requirement already satisfied: tqdm>=4.62.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from datasets) (4.64.1)
    Collecting xxhash
      Downloading xxhash-3.4.1-cp310-cp310-macosx_10_9_x86_64.whl (31 kB)
    Requirement already satisfied: numpy>=1.17 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from datasets) (1.23.5)
    Requirement already satisfied: packaging in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from datasets) (22.0)
    Collecting multiprocess
      Downloading multiprocess-0.70.16-py310-none-any.whl (134 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m5.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting fsspec[http]<=2024.2.0,>=2023.1.0
      Downloading fsspec-2024.2.0-py3-none-any.whl (170 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m170.9/170.9 kB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: requests>=2.19.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from datasets) (2.28.1)
    Requirement already satisfied: filelock in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from datasets) (3.12.4)
    Requirement already satisfied: pyyaml>=5.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from datasets) (6.0)
    Collecting pyarrow-hotfix
      Downloading pyarrow_hotfix-0.6-py3-none-any.whl (7.9 kB)
    Collecting huggingface-hub>=0.19.4
      Downloading huggingface_hub-0.22.1-py3-none-any.whl (388 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m388.6/388.6 kB[0m [31m13.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: attrs>=17.3.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from aiohttp->datasets) (22.1.0)
    Collecting aiosignal>=1.1.2
      Using cached aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
    Collecting yarl<2.0,>=1.0
      Downloading yarl-1.9.4-cp310-cp310-macosx_10_9_x86_64.whl (81 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m81.2/81.2 kB[0m [31m3.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting frozenlist>=1.1.1
      Downloading frozenlist-1.4.1-cp310-cp310-macosx_10_9_x86_64.whl (53 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.8/53.8 kB[0m [31m1.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting multidict<7.0,>=4.5
      Downloading multidict-6.0.5-cp310-cp310-macosx_10_9_x86_64.whl (30 kB)
    Collecting async-timeout<5.0,>=4.0
      Using cached async_timeout-4.0.3-py3-none-any.whl (5.7 kB)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from huggingface-hub>=0.19.4->datasets) (4.4.0)
    Requirement already satisfied: charset-normalizer<3,>=2 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests>=2.19.0->datasets) (2.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests>=2.19.0->datasets) (1.26.14)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests>=2.19.0->datasets) (2022.12.7)
    Requirement already satisfied: idna<4,>=2.5 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests>=2.19.0->datasets) (3.4)
    Collecting dill<0.3.9,>=0.3.0
      Downloading dill-0.3.8-py3-none-any.whl (116 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m4.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: python-dateutil>=2.8.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from pandas->datasets) (2022.7)
    Requirement already satisfied: six>=1.5 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from python-dateutil>=2.8.1->pandas->datasets) (1.16.0)
    Installing collected packages: xxhash, pyarrow-hotfix, pyarrow, multidict, fsspec, frozenlist, dill, async-timeout, yarl, multiprocess, huggingface-hub, aiosignal, aiohttp, datasets
      Attempting uninstall: fsspec
        Found existing installation: fsspec 2022.11.0
        Uninstalling fsspec-2022.11.0:
          Successfully uninstalled fsspec-2022.11.0
      Attempting uninstall: dill
        Found existing installation: dill 0.3.6
        Uninstalling dill-0.3.6:
          Successfully uninstalled dill-0.3.6
      Attempting uninstall: huggingface-hub
        Found existing installation: huggingface-hub 0.10.1
        Uninstalling huggingface-hub-0.10.1:
          Successfully uninstalled huggingface-hub-0.10.1
    Successfully installed aiohttp-3.9.3 aiosignal-1.3.1 async-timeout-4.0.3 datasets-2.18.0 dill-0.3.8 frozenlist-1.4.1 fsspec-2024.2.0 huggingface-hub-0.22.1 multidict-6.0.5 multiprocess-0.70.16 pyarrow-15.0.2 pyarrow-hotfix-0.6 xxhash-3.4.1 yarl-1.9.4
    Note: you may need to restart the kernel to use updated packages.



```python
import itertools
import jsonlines

from datasets import load_dataset
from pprint import pprint

from llama import BasicModelRunner
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[1], line 7
          4 from datasets import load_dataset
          5 from pprint import pprint
    ----> 7 from llama import BasicModelRunner
          8 from transformers import AutoTokenizer, AutoModelForCausalLM
          9 from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


    ModuleNotFoundError: No module named 'llama'



```python
pip3 install llama
```


      Cell In[3], line 1
        pip3 install llama
             ^
    SyntaxError: invalid syntax




```python
pip install llama
```

    Collecting llama
      Using cached llama-0.1.1.tar.gz (387 kB)
      Preparing metadata (setup.py) ... [?25lerror
      [1;31merror[0m: [1msubprocess-exited-with-error[0m
      
      [31m×[0m [32mpython setup.py egg_info[0m did not run successfully.
      [31m│[0m exit code: [1;36m1[0m
      [31m╰─>[0m [31m[6 lines of output][0m
      [31m   [0m Traceback (most recent call last):
      [31m   [0m   File "<string>", line 2, in <module>
      [31m   [0m   File "<pip-setuptools-caller>", line 34, in <module>
      [31m   [0m   File "/private/var/folders/bm/nf7hjdj17pjg8yr71z4z3btw0000gn/T/pip-install-zsf613ta/llama_1543b598b6b44db8bcbaf6f913a5d8b3/setup.py", line 6, in <module>
      [31m   [0m     execfile('llama/version.py')
      [31m   [0m NameError: name 'execfile' is not defined
      [31m   [0m [31m[end of output][0m
      
      [1;35mnote[0m: This error originates from a subprocess, and is likely not a problem with pip.
    [1;31merror[0m: [1mmetadata-generation-failed[0m
    
    [31m×[0m Encountered error while generating package metadata.
    [31m╰─>[0m See above for output.
    
    [1;35mnote[0m: This is an issue with the package mentioned above, not pip.
    [1;36mhint[0m: See above for details.
    [?25hNote: you may need to restart the kernel to use updated packages.



```python
import itertools
import jsonlines

from datasets import load_dataset
from pprint import pprint

from llama import BasicModelRunner
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[1], line 7
          4 from datasets import load_dataset
          5 from pprint import pprint
    ----> 7 from llama import BasicModelRunner
          8 from transformers import AutoTokenizer, AutoModelForCausalLM
          9 from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


    ModuleNotFoundError: No module named 'llama'



```python
pip install llama2
```

    Collecting llama2
      Downloading llama2-0.0.1.dev0-py3-none-any.whl (1.3 kB)
    Requirement already satisfied: torch in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from llama2) (1.12.1)
    Requirement already satisfied: typing_extensions in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from torch->llama2) (4.4.0)
    Installing collected packages: llama2
    Successfully installed llama2-0.0.1.dev0
    Note: you may need to restart the kernel to use updated packages.



```python
pip install transformers
```

    Requirement already satisfied: transformers in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (4.24.0)
    Requirement already satisfied: tqdm>=4.27 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from transformers) (4.64.1)
    Requirement already satisfied: packaging>=20.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from transformers) (22.0)
    Requirement already satisfied: pyyaml>=5.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from transformers) (6.0)
    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from transformers) (0.11.4)
    Requirement already satisfied: numpy>=1.17 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from transformers) (1.23.5)
    Requirement already satisfied: requests in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from transformers) (2.28.1)
    Requirement already satisfied: filelock in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from transformers) (3.12.4)
    Requirement already satisfied: huggingface-hub<1.0,>=0.10.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from transformers) (0.22.1)
    Requirement already satisfied: regex!=2019.12.17 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from transformers) (2022.7.9)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.10.0->transformers) (4.4.0)
    Requirement already satisfied: fsspec>=2023.5.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.10.0->transformers) (2024.2.0)
    Requirement already satisfied: charset-normalizer<3,>=2 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests->transformers) (2.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests->transformers) (1.26.14)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests->transformers) (2022.12.7)
    Requirement already satisfied: idna<4,>=2.5 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests->transformers) (3.4)
    Note: you may need to restart the kernel to use updated packages.



```python
import itertools
import jsonlines

from datasets import load_dataset
from pprint import pprint

from llama import BasicModelRunner
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[1], line 7
          4 from datasets import load_dataset
          5 from pprint import pprint
    ----> 7 from llama import BasicModelRunner
          8 from transformers import AutoTokenizer, AutoModelForCausalLM
          9 from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


    ModuleNotFoundError: No module named 'llama'



```python
pip install llama

```

    Collecting llama
      Using cached llama-0.1.1.tar.gz (387 kB)
      Preparing metadata (setup.py) ... [?25lerror
      [1;31merror[0m: [1msubprocess-exited-with-error[0m
      
      [31m×[0m [32mpython setup.py egg_info[0m did not run successfully.
      [31m│[0m exit code: [1;36m1[0m
      [31m╰─>[0m [31m[6 lines of output][0m
      [31m   [0m Traceback (most recent call last):
      [31m   [0m   File "<string>", line 2, in <module>
      [31m   [0m   File "<pip-setuptools-caller>", line 34, in <module>
      [31m   [0m   File "/private/var/folders/bm/nf7hjdj17pjg8yr71z4z3btw0000gn/T/pip-install-p9o3leh3/llama_2b11fe2bb4de4023a4e0202498e204c0/setup.py", line 6, in <module>
      [31m   [0m     execfile('llama/version.py')
      [31m   [0m NameError: name 'execfile' is not defined
      [31m   [0m [31m[end of output][0m
      
      [1;35mnote[0m: This error originates from a subprocess, and is likely not a problem with pip.
    [1;31merror[0m: [1mmetadata-generation-failed[0m
    
    [31m×[0m Encountered error while generating package metadata.
    [31m╰─>[0m See above for output.
    
    [1;35mnote[0m: This is an issue with the package mentioned above, not pip.
    [1;36mhint[0m: See above for details.
    [?25hNote: you may need to restart the kernel to use updated packages.



```python
pip install lamini
```

    Collecting lamini
      Downloading lamini-2.1.3-117-py3-none-any.whl (43 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.2/43.2 kB[0m [31m291.1 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: jsonlines in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from lamini) (4.0.0)
    Requirement already satisfied: pandas in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from lamini) (1.5.3)
    Requirement already satisfied: numpy in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from lamini) (1.23.5)
    Collecting azure-storage-blob
      Downloading azure_storage_blob-12.19.1-py3-none-any.whl (394 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m394.5/394.5 kB[0m [31m1.5 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: tqdm in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from lamini) (4.64.1)
    Requirement already satisfied: aiohttp in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from lamini) (3.9.3)
    Requirement already satisfied: requests in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from lamini) (2.28.1)
    Collecting lamini-configuration[yaml]
      Downloading lamini_configuration-0.8.3-py3-none-any.whl (22 kB)
    Requirement already satisfied: scikit-learn in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from lamini) (1.2.1)
    Requirement already satisfied: attrs>=17.3.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from aiohttp->lamini) (22.1.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from aiohttp->lamini) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from aiohttp->lamini) (1.9.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from aiohttp->lamini) (1.3.1)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from aiohttp->lamini) (1.4.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from aiohttp->lamini) (4.0.3)
    Requirement already satisfied: cryptography>=2.1.4 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from azure-storage-blob->lamini) (39.0.1)
    Collecting azure-core<2.0.0,>=1.28.0
      Downloading azure_core-1.30.1-py3-none-any.whl (193 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m193.4/193.4 kB[0m [31m5.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting isodate>=0.6.1
      Downloading isodate-0.6.1-py2.py3-none-any.whl (41 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m41.7/41.7 kB[0m [31m1.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: typing-extensions>=4.3.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from azure-storage-blob->lamini) (4.4.0)
    Requirement already satisfied: pyyaml<7.0,>=6.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from lamini-configuration[yaml]->lamini) (6.0)
    Requirement already satisfied: python-dateutil>=2.8.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from pandas->lamini) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from pandas->lamini) (2022.7)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests->lamini) (2022.12.7)
    Requirement already satisfied: idna<4,>=2.5 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests->lamini) (3.4)
    Requirement already satisfied: charset-normalizer<3,>=2 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests->lamini) (2.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from requests->lamini) (1.26.14)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from scikit-learn->lamini) (2.2.0)
    Requirement already satisfied: scipy>=1.3.2 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from scikit-learn->lamini) (1.10.0)
    Requirement already satisfied: joblib>=1.1.1 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from scikit-learn->lamini) (1.1.1)
    Requirement already satisfied: six>=1.11.0 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from azure-core<2.0.0,>=1.28.0->azure-storage-blob->lamini) (1.16.0)
    Collecting typing-extensions>=4.3.0
      Downloading typing_extensions-4.10.0-py3-none-any.whl (33 kB)
    Requirement already satisfied: cffi>=1.12 in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from cryptography>=2.1.4->azure-storage-blob->lamini) (1.15.1)
    Requirement already satisfied: pycparser in /Users/rafaelortiz/anaconda3/lib/python3.10/site-packages (from cffi>=1.12->cryptography>=2.1.4->azure-storage-blob->lamini) (2.21)
    Installing collected packages: typing-extensions, lamini-configuration, isodate, azure-core, azure-storage-blob, lamini
      Attempting uninstall: typing-extensions
        Found existing installation: typing_extensions 4.4.0
        Uninstalling typing_extensions-4.4.0:
          Successfully uninstalled typing_extensions-4.4.0
    Successfully installed azure-core-1.30.1 azure-storage-blob-12.19.1 isodate-0.6.1 lamini-2.1.3 lamini-configuration-0.8.3 typing-extensions-4.10.0
    Note: you may need to restart the kernel to use updated packages.



```python
import itertools
import jsonlines

from datasets import load_dataset
from pprint import pprint

from llama import BasicModelRunner
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```


```python
instruction_tuned_dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
```


    Downloading readme:   0%|          | 0.00/7.47k [00:00<?, ?B/s]



```python
m = 5
print("Instruction-tuned dataset:")
top_m = list(itertools.islice(instruction_tuned_dataset, m))
for j in top_m:
  print(j)
```

    Instruction-tuned dataset:
    {'instruction': 'Give three tips for staying healthy.', 'input': '', 'output': '1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.', 'text': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.'}
    {'instruction': 'What are the three primary colors?', 'input': '', 'output': 'The three primary colors are red, blue, and yellow.', 'text': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat are the three primary colors?\n\n### Response:\nThe three primary colors are red, blue, and yellow.'}
    {'instruction': 'Describe the structure of an atom.', 'input': '', 'output': 'An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom.', 'text': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nDescribe the structure of an atom.\n\n### Response:\nAn atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom.'}
    {'instruction': 'How can we reduce air pollution?', 'input': '', 'output': 'There are a number of ways to reduce air pollution, such as shifting to renewable energy sources, encouraging the use of public transportation, prohibiting the burning of fossil fuels, implementing policies to reduce emissions from industrial sources, and implementing vehicle emissions standards. Additionally, individuals can do their part to reduce air pollution by reducing car use, avoiding burning materials such as wood, and changing to energy efficient appliances.', 'text': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nHow can we reduce air pollution?\n\n### Response:\nThere are a number of ways to reduce air pollution, such as shifting to renewable energy sources, encouraging the use of public transportation, prohibiting the burning of fossil fuels, implementing policies to reduce emissions from industrial sources, and implementing vehicle emissions standards. Additionally, individuals can do their part to reduce air pollution by reducing car use, avoiding burning materials such as wood, and changing to energy efficient appliances.'}
    {'instruction': 'Describe a time when you had to make a difficult decision.', 'input': '', 'output': 'I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client’s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team’s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client’s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities.', 'text': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nDescribe a time when you had to make a difficult decision.\n\n### Response:\nI had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client’s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team’s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client’s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities.'}



```python
prompt_template_with_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

prompt_template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
```


```python
processed_data = []
for j in top_m:
  if not j["input"]:
    processed_prompt = prompt_template_without_input.format(instruction=j["instruction"])
  else:
    processed_prompt = prompt_template_with_input.format(instruction=j["instruction"], input=j["input"])

  processed_data.append({"input": processed_prompt, "output": j["output"]})

```


```python
pprint(processed_data[0])
```

    {'input': 'Below is an instruction that describes a task. Write a response '
              'that appropriately completes the request.\n'
              '\n'
              '### Instruction:\n'
              'Give three tips for staying healthy.\n'
              '\n'
              '### Response:',
     'output': '1.Eat a balanced diet and make sure to include plenty of fruits '
               'and vegetables. \n'
               '2. Exercise regularly to keep your body active and strong. \n'
               '3. Get enough sleep and maintain a consistent sleep schedule.'}



```python
with jsonlines.open(f'alpaca_processed.jsonl', 'w') as writer:
    writer.write_all(processed_data)
```


```python
dataset_path_hf = "lamini/alpaca"
dataset_hf = load_dataset(dataset_path_hf)
print(dataset_hf)
```


    Downloading readme:   0%|          | 0.00/388 [00:00<?, ?B/s]


    Downloading data: 100%|██████████████████████████████████████████████████████████████████| 12.7M/12.7M [00:01<00:00, 7.46MB/s]



    Generating train split:   0%|          | 0/52002 [00:00<?, ? examples/s]


    DatasetDict({
        train: Dataset({
            features: ['input', 'output'],
            num_rows: 52002
        })
    })



```python
instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
instruct_output = instruct_model("Tell me how to train my dog to sit")
print("Instruction-tuned output (Llama 2): ", instruct_output)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[9], line 2
          1 instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
    ----> 2 instruct_output = instruct_model("Tell me how to train my dog to sit")
          3 print("Instruction-tuned output (Llama 2): ", instruct_output)


    File ~/anaconda3/lib/python3.10/site-packages/lamini/runners/base_runner.py:41, in BaseRunner.__call__(self, prompt, system_prompt, output_type, max_tokens, callback, metadata)
         32 def __call__(
         33     self,
         34     prompt: Union[str, List[str]],
       (...)
         39     metadata: Optional[List] = None,
         40 ):
    ---> 41     return self.call(
         42         prompt,
         43         system_prompt,
         44         output_type,
         45         max_tokens,
         46         callback,
         47         metadata,
         48     )


    File ~/anaconda3/lib/python3.10/site-packages/lamini/runners/base_runner.py:61, in BaseRunner.call(self, prompt, system_prompt, output_type, max_tokens, callback, metadata)
         50 def call(
         51     self,
         52     prompt: Union[str, List[str]],
       (...)
         57     metadata: Optional[List] = None,
         58 ):
         59     input_objects = self.create_final_prompts(prompt, system_prompt)
    ---> 61     return self.lamini_api.generate(
         62         prompt=input_objects,
         63         model_name=self.model_name,
         64         max_tokens=max_tokens,
         65         output_type=output_type,
         66         callback=callback,
         67         metadata=metadata,
         68     )


    File ~/anaconda3/lib/python3.10/site-packages/lamini/api/lamini.py:70, in Lamini.generate(self, prompt, model_name, output_type, max_tokens, max_new_tokens, callback, metadata)
         62 if isinstance(prompt, str) or (isinstance(prompt, list) and len(prompt) == 1):
         63     req_data = self.make_llm_req_map(
         64         prompt=prompt,
         65         model_name=model_name or self.model_name,
       (...)
         68         max_new_tokens=max_new_tokens,
         69     )
    ---> 70     result = self.completion.generate(req_data)
         71     if output_type is None:
         72         if isinstance(prompt, list) and len(prompt) == 1:


    File ~/anaconda3/lib/python3.10/site-packages/lamini/api/utils/completion.py:15, in Completion.generate(self, params)
         14 def generate(self, params):
    ---> 15     resp = make_web_request(
         16         self.api_key, self.api_prefix + "completions", "post", params
         17     )
         18     return resp


    File ~/anaconda3/lib/python3.10/site-packages/lamini/api/rest_requests.py:119, in make_web_request(key, url, http_method, json)
        116 def make_web_request(key, url, http_method, json=None):
        117     headers = {
        118         "Content-Type": "application/json",
    --> 119         "Authorization": "Bearer " + key,
        120     }
        121     if http_method == "post":
        122         resp = requests.post(url=url, headers=headers, json=json)


    TypeError: can only concatenate str (not "NoneType") to str



```python
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
```


    tokenizer_config.json:   0%|          | 0.00/396 [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/2.11M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/99.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/567 [00:00<?, ?B/s]



    pytorch_model.bin:   0%|          | 0.00/166M [00:00<?, ?B/s]



```python
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer
```


```python
finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_path)
print(finetuning_dataset)
```


    Downloading readme:   0%|          | 0.00/577 [00:00<?, ?B/s]


    Downloading data: 100%|████████████████████████████████████████████████████████████████████| 615k/615k [00:00<00:00, 2.11MB/s]
    Downloading data: 100%|███████████████████████████████████████████████████████████████████| 83.7k/83.7k [00:00<00:00, 317kB/s]



    Generating train split:   0%|          | 0/1260 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/140 [00:00<?, ? examples/s]


    DatasetDict({
        train: Dataset({
            features: ['question', 'answer', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 1260
        })
        test: Dataset({
            features: ['question', 'answer', 'input_ids', 'attention_mask', 'labels'],
            num_rows: 140
        })
    })



```python
test_sample = finetuning_dataset["test"][0]
print(test_sample)

print(inference(test_sample["question"], model, tokenizer))
```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.


    {'question': 'Can Lamini generate technical documentation or user manuals for software projects?', 'answer': 'Yes, Lamini can generate technical documentation and user manuals for software projects. It uses natural language generation techniques to create clear and concise documentation that is easy to understand for both technical and non-technical users. This can save developers a significant amount of time and effort in creating documentation, allowing them to focus on other aspects of their projects.', 'input_ids': [5804, 418, 4988, 74, 6635, 7681, 10097, 390, 2608, 11595, 84, 323, 3694, 6493, 32, 4374, 13, 418, 4988, 74, 476, 6635, 7681, 10097, 285, 2608, 11595, 84, 323, 3694, 6493, 15, 733, 4648, 3626, 3448, 5978, 5609, 281, 2794, 2590, 285, 44003, 10097, 326, 310, 3477, 281, 2096, 323, 1097, 7681, 285, 1327, 14, 48746, 4212, 15, 831, 476, 5321, 12259, 247, 1534, 2408, 273, 673, 285, 3434, 275, 6153, 10097, 13, 6941, 731, 281, 2770, 327, 643, 7794, 273, 616, 6493, 15], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'labels': [5804, 418, 4988, 74, 6635, 7681, 10097, 390, 2608, 11595, 84, 323, 3694, 6493, 32, 4374, 13, 418, 4988, 74, 476, 6635, 7681, 10097, 285, 2608, 11595, 84, 323, 3694, 6493, 15, 733, 4648, 3626, 3448, 5978, 5609, 281, 2794, 2590, 285, 44003, 10097, 326, 310, 3477, 281, 2096, 323, 1097, 7681, 285, 1327, 14, 48746, 4212, 15, 831, 476, 5321, 12259, 247, 1534, 2408, 273, 673, 285, 3434, 275, 6153, 10097, 13, 6941, 731, 281, 2770, 327, 643, 7794, 273, 616, 6493, 15]}
    
    
    I have a question about the following:
    
    How do I get the correct documentation to work?
    
    A:
    
    I think you need to use the following code:
    
    A:
    
    You can use the following code to get the correct documentation.
    
    A:
    
    You can use the following code to get the correct documentation.
    
    A:
    
    You can use the following



```python
instruction_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
```


    config.json:   0%|          | 0.00/717 [00:00<?, ?B/s]



    pytorch_model.bin:   0%|          | 0.00/282M [00:00<?, ?B/s]


    Some weights of GPTNeoXForCausalLM were not initialized from the model checkpoint at lamini/lamini_docs_finetuned and are newly initialized: ['gpt_neox.layers.0.attention.bias', 'gpt_neox.layers.0.attention.masked_bias', 'gpt_neox.layers.4.attention.masked_bias', 'gpt_neox.layers.5.attention.masked_bias', 'gpt_neox.layers.5.attention.bias', 'gpt_neox.layers.1.attention.bias', 'gpt_neox.layers.4.attention.bias', 'gpt_neox.layers.3.attention.bias', 'gpt_neox.layers.2.attention.masked_bias', 'gpt_neox.layers.2.attention.bias', 'gpt_neox.layers.1.attention.masked_bias', 'gpt_neox.layers.3.attention.masked_bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
print(inference(test_sample["question"], instruction_model, tokenizer))
```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.


    Yes, Lamini can generate technical documentation or user manuals for software projects. This can be achieved by providing a prompt for a specific technical question or question to the LLM Engine, or by providing a prompt for a specific technical question or question. Additionally, Lamini can be trained on specific technical questions or questions to help users understand the process and provide feedback to the LLM Engine. Additionally, Lamini



```python

```
