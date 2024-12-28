It comes from MLOPs framework

* Automation, orchestrestation
  
* Mllops for LLMs (LLMops) is different than LLM system design
## LLMops: 
* experimenting FM,
* prompt desing and management,
* supervised-tunning,
* monitoring,
* evaluation

## LLM system desing: 
* Chainnig LLMs,
* grounding (make sure that LLM has the relevant information for the output that you desired),
* track history
* Model customization: data prep, tunning, evaluate.
## LLmops pipeline
* Data preparation and versioning --- Pipeline desing (supervised tunning) --- Artifact (file with configuration and workflow) --- then pipeline execution - deploy llms -- propmting and predictions --- responsible aI (check the repsonse) 
 * Key concepts: Orquestration (step by step definition) and automation
### File formats for train and evaluation
*  JSONL (small to medium datasrts)
*  TFrecord (bin)
*  Parquet file (bin)
*  Versionning artifacts


## Automation and orchestration with pipelines. 
* Kubeflow pipelines or Apache Ariflow
* Tranning - evaluation (varous times)
* Production data -- Trainned model -- prediction -- evaluation -- Can change train data

## ops
* Package, deploy and versionning
* Model momitoring: metrics and safety
* Inference scalability: Load test, controlled rollout etc
* Latency: Permitted latency -- smaller models - faster processig (gpus, tpus) -- deploy regionally
* prediction (prompts)
* evaluation  (safety scores, citation, performance metrics..)
* Guartrials.ai



  
    
