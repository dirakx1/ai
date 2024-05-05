# Whats is the needed GPU MEM for running a LLM
* fitting models into the ram:  precision x # model parameters) .

# Quntization
* Allows to run more powerful models in less RAM
* allows train larger model
* Reduce energy consumption
* basically allows to train models with less precision but the output has the same performance.

# TGI (HF)
* Text generation inference.
* allows to serve many LLMs, for example llama 70b
* Rust gRPC inrterface to GPUs
* Used by Hugging Face

# vLLM
* Uses paged attention
* Seems the easiest one if you want to serve your own model. 

# deep speed mii
# adding lora and slora adapters to serve llms (HF)
* S-lora thousands of lora adapters (still needs inprovements?

# Serving Llama with vendors
* grog cloud
* others..

* 



