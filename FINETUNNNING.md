# Fine tunning LLMS
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
* pre-trainning -- base model -- Finetune -- finetune model.
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
  3. tokenize: Pad, truncate (right size going into the model) 
  4. split into train / test data.
  

