# Fine tunning LLMS
*  Figure out your task
*  Collect data related to your task inputs / outputs (structured as such)
*  you can generate data if you dont have
*  Finetune an small model (400m - 1b)
*  vary the ammout of data
*  evaluate your llm to know whats going well vs no
*  collect more data to improve.
*  increase data complexity
*  Increase model size for performance. 
*  More data and computing, but more exactitude.
*  stop hallucinations, concistence,performance
*  Privacy (on premises, vpc, prevent leakage)
*  Fine tunning and smaller llm can reduce cost for applications that require more requests.
*  Low latency

## Finetuning libs
* Pytorch (meta)
* Huggingface
* LLama library (Lamini)
* The pile. opensource pretrainning data.
* pre-trainning -- base model -- Finetune -- finetuned model.
* Finetuning can also have the same kind of unlabeled data  or also labeled or structured data.
## Finetunning steps
* Data prep -- trainning -- evaluation (iterative)

# Instruction finetunning
* Type of finetunning
* Turns GPT to ChatGPT (teaches the model to recieve instructions)

# Data preparation
* preparing data for trainning (fine tunning)
* higher quality, real, diversity, more.
## steps:
* 1. collect instruction-response pairs
  2. concatenate pairs
  3. tokenize: Pad, truncate (right size going into the model) , convert text to numbers and numbers to text (encode, decode) , there are different tokenizers for each model. 
  4. split into train / test data.
 
# Trainning the model. 
* using Lamini library.

# Evaluation and iteration 
* human evaluation -- prefered method
* test dataset
* Evaluation method (eleutherai)
  ** ARC set of grade school questions
  ** HellaSwag a set of common sense
  ** MMLU multitaks metric coveric elemental math, american history, computer science
  ** TruthfulQA evaluates a module propensity to propagate falsehoods found online
* Another evaluation method is error analysis: cathegorize errors for example misspelling, too-long, repetitive.
# Finetunning techniques. 
* Harder tasks could require larger models
* Combinations of task could also be harder
* Having an agent
* model sizes need more computing resources.
* PEFT: parameter eficient finetunning (more efficience on trainning) or example LoRa (low range adaptation).
* Freezing layers
* Reinformcement Learning with Human feeadback or RLHF (DPO directe preference optimize optimization)
* For larger llama models: Qlora and freeze layers Or use third parthy vendors like Together.ai
* You might need 20-30 iteration to have better results.  

  


   


  

