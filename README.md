
# Transformer QA

<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be used in an existing business or service. (You should not propose a system in which a new business or service would be developed around the machine learning system.) Describe the value proposition for the machine learning system. What’s the (non-ML) status quo used in the business or service? What business metric are you going to be judged on? (Note that the “service” does not have to be for general users; you can propose a system for a science problem, for example.)
-->
### Value Proposition
We propose a machine learning system that augments software engineering workflows by enabling accurate, context-aware question answering over large codebases and project histories. Specifically, our system builds a domain-specific retrieval-augmented generation (RAG) model that can answer technical and procedural questions about the HuggingFace Transformers project using a high-quality, internally generated QA dataset. Our target users are developers of HuggingFace Transformers and programmers who use the Transformers library in their daily development. With our system, users can ask questions about the library in natural language, enabling fast and efficient development.

**Status quo**: In current open-source software development workflows, developers must manually search through source code, commit history, GitHub issues, and documentation to understand past decisions, code functionality, and bug resolutions. This process is time-consuming, error-prone, and inefficient—especially for new contributors or in large, fast-evolving repositories.

**Value proposition**: Our ML system enables efficient knowledge retrieval by providing natural language answers grounded in the repository’s development history. It can help developers quickly understand code behavior, rationale behind changes, and historical discussions without manual digging. This significantly reduces onboarding time, accelerates debugging, and improves overall team productivity.

**Business metric**: The system will be evaluated on developer productivity metrics, such as reduced time-to-resolution for support queries, decreased onboarding time for new contributors, and accuracy/acceptance rate of answers in live usage (based on user feedback and manual review). We also monitor hallucination rates and latency to ensure system reliability in a real-world setting.
### Contributors

<!-- Table of contributors and their roles. First row: define responsibilities that are shared by the team. Then each row after that is: name of contributor, their role, and in the third column you will link to their contributions. If your project involves multiple repos, you will link to their contributions in all repos here. -->

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| Zeqiao Huang                 | model training |  https://github.com/Jasonzzzz28/ECE-GY-9183-Project/commits?author=Hzq1941048295                                  |
| Siyuan Zhang                 |  model serving and monitoring |    https://github.com/Jasonzzzz28/ECE-GY-9183-Project/commits?author=Jasonzzzz28                     |
| Junzhi Chen                   |       data pipeline          |           https://github.com/Jasonzzzz28/ECE-GY-9183-Project/commits?author=szjiozi                  |
| Chuangyu Xu                  |      continuous X pipeline           |       https://github.com/Jasonzzzz28/ECE-GY-9183-Project/commits/main/?author=xcy103                  |



### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. Must include: all the hardware, all the containers/software platforms, all the models, all the data. -->
<img width="1224" alt="image" src="https://github.com/user-attachments/assets/a6689540-e94f-4328-b3a1-f3da591cbee7" />


### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. Name of data/model, conditions under which it was created (ideally with links/references), conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| Transformers QA dataset   |     Created by ourselves (see detail in data pipeline section)               |    Training and evaluation               |
| Stanford Question Answering Dataset (SQuAD)  |      Downloaded from [Huggingface](https://huggingface.co/datasets/rajpurkar/squad), details see [this paper](https://arxiv.org/abs/1606.05250)             |       Training            |
| Llama-3.1-8B-Instruct |       Downloaded from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-14B), details see [this paper](https://arxiv.org/abs/2412.15115)             |      Base QA model             |
| BGE-M3          |       Downloaded from [Hugging Face](https://huggingface.co/BAAI/bge-m3), details see [this paper](https://arxiv.org/pdf/2402.03216)             |      Embedding model for RAG             |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), how much/when, justification. Include compute, floating IPs, persistent storage. The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| 2 VMs | 2 for entire project duration                     |     One for model training, the other for model serving       |
| `gpu_a100`     | 2 A100 80GB for 4 hour block twice a week    |    For our 14b model, we need approximatly 120GB RAM for training with fp16.    |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |    One for development, the other for testing           |
| Object Store    | 1 for entire project duration |    For storing the QA data pair and the user feedback data          |


### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the diagram, (3) justification for your strategy, (4) relate back to lecture material, (5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, and which optional "difficulty" points you are attempting. -->


Problem Definition
We frame commit-based question–answering as a conditional language modeling task: given a user’s natural-language question plus a JSON‐encoded commit context, the model must generate the appropriate answer text.

Inputs

![image1](model_training/7861747060471_.pic.jpg)


A stringified Python dict with keys like commit_hash, author, date, message.

Output (Target Variable)

The answer string, tokenized and appended with an EOS token.
During training, we maximize the likelihood of the ground‐truth answer continuation given the prompt.

Customer Use Case

Interactive Chatbot: engineers query “what changed in this commit?” and immediately get a human‐readable summary.


Model Choice

Base Model: meta-llama/Meta-Llama-3.1-8B-Instruct, a high‐capacity instruction‐tuned causal LM well‐suited to free‐form text generation.
LoRA Adapters: to drastically reduce fine‐tuning costs (only ~0.17% of parameters trained) and enable rapid re‐training on small incremental data.
4-bit Quantization: using bitsandbytes to fit the 8B‐parameter model onto a single 24 GiB GPU without sacrificing throughput.

Envirnoment Set up:
1-docker-compose-data.yaml
    This Compose file is a “data initialization” stack that runs a one-off container to populate a Docker volume with your raw           dataset (e.g. the qa_from_commits_formatted.json file). Typical contents:
    Service: often called init-data
    Image: a lightweight Linux or Alpine image
    Volume mounts:
    Mount your local data directory (e.g. ./model_training) into the container
    Mount a named volume (e.g. transformer-qa) where data will be copied
    Command: a cp or data‐loading script that copies files from the source mount into the named volume
    Restart policy: usually on-failure or none so it only runs once

2-Dockerfile.jupyter-torch-mlflow-cuda
    This Dockerfile builds a custom Jupyter environment with CUDA, PyTorch, and MLflow support. Key steps:
    Base image: NVIDIA’s CUDA runtime (e.g. nvidia/cuda:12.1.0-base-ubuntu22.04)
    System dependencies: installs Python, Jupyter Lab, git, and any OS packages needed for GPU drivers
    Python environment: installs PyTorch with CUDA via pip or conda, plus MLflow, Hugging Face Transformers, Datasets, and notebook     extensions
    Workspace setup: creates a /home/jovyan/work directory and sets it as the Jupyter working directory
    Entrypoint: configures CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
    Ports: exposes 8888 for Jupyter
    
3-docker-compose-mlflow.yaml
    This Compose file launches your MLflow tracking server (and optionally its backing services). Typical services:
    mlflow
    Runs mlflow server --backend-store-uri postgresql://… --default-artifact-root s3://… --host 0.0.0.0 --port 5000
    Environment variables:
    MLFLOW_TRACKING_URI=http://mlflow:5000
    Database URL, credentials
    Volume mounts:
    Local directory for artifact storage 
I. Training Pipeline Architecture

    graph TD
    A[Raw Codebase Data] --> B(Data Pipeline)
    B --> C{Versioned QA Dataset}
    C --> D[Model Training Cluster]
    D --> E[Fine-tuned LLM]
    E --> F[Evaluation on Curated QA Set]
    F -->|Accuracy < 85%| B
II. Core Implementation
  2.1 Model Selection & Training Strategy

    base model:

      python"
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Optimized for code understanding
            trust_remote_code=True,
            attn_implementation="flash_attention_2"  # Memory optimization
        )"

    Retraining Protocol:
      Trigger Conditions:
      Parameter-Efficient Fine-tuning:
  
  2.2 Distributed Training Configuration
    Hardware Matrix:
Stage	GPU Type	Count	Memory Optimizations
Pretraining	A100-80G	8	FSDP + Activation Checkpoint
Fine-tuning	A10G	4	DDP + Gradient Accumulation
Evaluation	T4	1	FP16 Quantization
    Performance Validation:



III. Training Infrastructure
![image](model_training/7851747059623_.pic.jpg)

  3.1 MLFlow Experiment Tracking

        with mlflow.start_run():
    mlflow.log_params({(https://github.com/user-attachments/assets/4d24bba1-8a8e-47a6-aacc-2d69e3b82370)

        "max_seq_length": 4096,
        "grad_accum_steps": 4
    })
    mlflow.pytorch.log_model(
        pipeline, 
        "model",
        registered_model_name="code_qwen_qa"
    )

    



IV. Cross-Team Integration
  4.1 Data Pipeline Interface
    Input Specification:
        Example structure of the data:
          ```python
          {
            "id": 0,
            "category": "source code",
            "context": "code name and its docstring",
            "question": "What does this function do?",
            "answer": "answer"
          }
          ```
  4.2 Model Serving Interface
    Output Schema:
                  
                  {
              "answer": "Use Spring Batch's ItemProcessor", 
              "confidence": 0.87,
              "source": {
                "commit": "a1b2c3d",
                "issue_id": 456
              }
            }
V. Advanced Features
  5.1 Hybrid Parallel Training
  
      python"
        strategy = FSDP(
        auto_wrap_policy={TransformerEncoderLayer},
        cpu_offload=CPUOffload(offload_params=True),
        mixed_precision=torch.float16
        )"
    
  5.2 Hyperparameter Tuning
VI. Validation Metrics


Difficult point: 
As the internal codebase evolves through frequent Git commits, pull requests, and issue resolutions, the QA dataset generated from these sources exhibits:

Semantic Drift: New API patterns/deprecations altering answer correctness

Context Fragmentation: Code snippet dependencies changing across versions

Label Noise Proliferation: Auto-generated QA pairs becoming misaligned with ground truth




<br>
<br>

#### Model serving, evaluation and monitoring platforms
<img width="1159" alt="image" src="https://github.com/user-attachments/assets/074419e5-6975-4ba4-8b34-627354fa86c2" />
<img width="1159" alt="image" src="https://github.com/user-attachments/assets/3de7f37a-a07e-4ebe-90b9-ef1c2d222a74" />

***Environment Setup For Model Serving, Evaluation and Monitoring***

Added the following files for setting up and running the entire serving and evaluation system:

- **How To Run The Serving, Evaluation and Monitoring System**
  ```bash
  docker compose -f ~/Transformer-QA/docker/docker-compose-tqa.yaml up -d
  ```

- **Application Setup**:
  - [docker/docker-compose-tqa.yaml](docker/docker-compose-tqa.yaml): Defines the services and configurations for deploying the application using Docker Compose.
  - [Dockerfile](Dockerfile): Used to build the Docker image for the application.
  - [requirements.txt](requirements.txt): Lists the Python dependencies required for the application.

- **FastAPI Server Setup**:
  - [fastapi_server/Dockerfile](fastapi_server/Dockerfile): Dockerfile for building the FastAPI server image.
  - [fastapi_server/requirements.txt](fastapi_server/requirements.txt): Lists the dependencies specific to the FastAPI server.

- **Evaluation and Monitoring Setup**:
  - [evaluation_monitor/docker/docker-compose.yml](evaluation_monitor/docker/docker-compose.yml): Docker Compose file for setting up evaluation and monitoring services.
  - [evaluation_monitor/docker/Dockerfile](evaluation_monitor/docker/Dockerfile): Dockerfile for building the evaluation and monitoring service image.
  - [evaluation_monitor/requirements.txt](evaluation_monitor/requirements.txt): Lists the dependencies required for evaluation and monitoring services.
  - [prometheus.yml](docker/prometheus.yml): Configuration file for Prometheus monitoring.


***Model Serving***

**Serving from an API Endpoint** 

- Backend: We will deploy our fine-tuned Llama-3.1-8B-Instruct model using FastAPI Server, exposing it through a REST API. The API will accept JSON-formatted requests containing user questions and return predictions in JSON format. For more details, refer to the [fastapi_server folder](fastapi_server/).

- Frontend: A Flask-based web interface will handle user interactions and send requests to the FastAPI backend for real-time predictions. For more details, refer to the [templates folder](templates/), [static folder](static/), [app.py file](app.py).

- Endpoint Accessibility: The API will be hosted on Chameleon Cloud and accessible via a configurable URL.

**Requirements** 

- Model Size: The expected size of the trained and optimized Llama-3.1-8B-Instruct model is approximately 16GB (8 billion parameters * 2 bytes per parameter in FP16 precision). This model will be stored in persistent storage on Chameleon (as defined in Unit 8).

- Throughput (Batch Inference): We anticipate a relatively low batch inference requirement. We aim for 6 - 12 QPS.

- Latency (Online Inference): For real-time question answering, we aim to achieve a latency of 1s - 1.5s per request. While not strictly real-time, this latency is crucial for a responsive and interactive user experience.

- Concurrency (Cloud Deployment): Our cloud deployment on Chameleon must support a concurrency of 8 simultaneous requests to handle concurrent users asking questions. This represents a minimal concurrency requirement for our prototype.

**Model Optimizations** [See detailed experiments here](workspace/serving_optimizations.ipynb)

- Model Format: Convert to ONNX for broader optimization support.

- Graph Optimization: Utilize ONNX Runtime's graph optimizations.

- Quantization: Implement FP16 quantization, allowing a maximum accuracy loss of 0.01 (on validation set).

- Execution Provider: Benchmark and select the faster performer between CUDA and TensorRT execution providers within ONNX Runtime.


**System Optimizations** [See detailed experiments here](workspace/serving_optimizations.ipynb)

- Model Server: Utilize Fast API Server for efficient execution of the optimized Llama-3.1-8B-Instruct model.

- Scaling: Deploy the model across 2 GPUs, with 2 instances running on each GPU.

- Concurrency: Configure each instance to handle a concurrency of 2, resulting in a total system concurrency of 8.

- Batching Strategy: Enable dynamic batching for efficient request aggregation.

**Selected Optimizations** 
- Option 1: Utilize HF transformers framework with PyTorch FP16 quantization for inference
- Option 2: Utilize vLLM framework with FP16 quantization for inference

<br>

***Evaluation and monitoring***

**Offline Evaluation** [See detailed implementation](evaluation_monitor/offline_eval/)

Automated testing after training:

- Standard/Domain Use Cases: Evaluate on the Transformers documentation test set.

- Population/Slice Analysis: Analyze performance on different question categories and complexities.

- Known Failure Modes: Test for adversarial prompts, hallucinations, and reasoning limitations.

- Unit Tests: Verify basic factual questions.

- Metrics: Track F1, Exact Match, Precision, and Recall.

- Model Registration: Register new model only if it exceeds performance thresholds.

**Staging Load Test** [See detailed implementation](evaluation_monitor/load_test/load_test.py)

Using load testing frameworks, like Locust, simulate concurrent users and measure:

- QPS (Target: 6-12)

- Latency (Target: 500ms - 1.5s)

- Error Rate

- Resource Utilization (CPU/GPU)

- Optimize resources based on results

**Canary Online Evaluation** [See detailed implementation](evaluation_monitor/online_eval/simulate_traffic.py)

We will conduct online evaluations in a canary environment by role-playing distinct user personas (novice, experienced developer, researcher). We'll generate questions representative of each persona and assess the model's accuracy(EM, F1) on these simulated interactions, in addition to standard load test metrics (QPS, latency, error rate).

For example:

- Novice User: "What is a Transformer?" "How do I load a pre-trained model?"

- Experienced Developer: "How do I fine-tune a Trainer with FSDP?" "What are the expected input formats for pipeline?"

- Researcher: "What are the key differences between BERT and RoBERTa?" "How does the attention mechanism work in Longformer?"
  
- Transformer Enthusiast: "Which Transformer is stronger: Optimus Prime or Megatron?"

**Monitoring**
<img width="1280" alt="image" src="https://github.com/user-attachments/assets/be6ba4a9-ccf9-4a93-b594-3d3dcb9e0bba" />


We use Prometheus and Grafana to monitor system performance, resource utilization, and application metrics. [See detailed dashboard configuration here](evaluation_monitor/grafana/dashboard.json). 

**Close the Loop** 

<img width="832" alt="image" src="https://github.com/user-attachments/assets/836ad595-a895-4763-8b78-f591edff722f" />

We will gather feedback about the quality of the model's predictions through:

- Periodic review and annotation of model outputs by human annotators to identify and correct errors, biases, or areas for improvement. (This is achieved through label studio. [See detailed configuration here](evaluation_monitor/label_studio/))

**Business Evaluation** 

- Tracking Active Users: Monitoring the number of daily/weekly active users interacting with the QA bot.

- Measuring Task Completion Rates: Analyzing whether users are able to more quickly resolve coding problems or understand the codebase with the help of the QA bot. This might be inferred from usage patterns, like time spent viewing documentation after a QA interaction.

- Conducting User Surveys: Collecting qualitative feedback from users regarding their satisfaction with the bot and how it impacts their workflow.

- Monitoring Support Ticket Volume: Observing if the QA bot leads to a reduction in the number of support tickets related to the Transformers library, indicating improved self-service and code understanding.

**Monitor for model degradation**

- Human Annotator Analysis: Track correction frequency to identify recurring model errors.

- Canary Deployments: Test new models with a small portion of live traffic while comparing performance against the current model to prevent regressions.

- Automatic Retraining: Set performance thresholds (e.g., rating drops or error increases) to trigger retraining with fresh labeled data.

<br>
<br>

#### Data pipeline

**Persistent storage**

Run [notebook](https://github.com/Jasonzzzz28/Transformer-QA/blob/main/data/provision_data.ipynb) in Chameleon Jupyter environment to provision object storage on Chameleon, which can be attached to and detached from our infrastructure. The object store will store offline data for our project.

**Offline data**

In this project, we aim to construct a high-quality internal QA dataset using the [HuggingFace Transformers Project](https://github.com/huggingface/transformers) as the internal knowledge base. The goal is to extract meaningful question-answer pairs from various components of the project to support the training of a retrieval-augmented generation (RAG) QA system tailored to this domain.

The dataset will be built by parsing and processing multiple information sources within the repository, including:

- Source code: Function and class definitions with docstrings and inline comments will be parsed to generate technical "what" and "how" questions.

- Git commit history: Commit messages and associated diffs will be analyzed to produce "why" and "what changed" style QA pairs.

- Real world QA data: QA data from [stackoverflow](https://stackoverflow.com/questions/tagged/huggingface)

The QA dataset will be generated in the following ways: 
- Rule-based automatic generation (e.g., "What does this function do?", "Why was this line changed?").
- LLM generation by strict source-alignment prompts. All QA examples will include metadata linking back to their origin (e.g., file path, commit hash, issue number) to maintain traceability.

Example structure of the data:
```python
{
    "question": "How does commit e1f379b alter the existing implementation?",
    "answer": "Fixing the example in generation strategy doc (#37598)\n\nUpdate generation_strategies.md\n\nThe prompt text shown in the example does not match what is inside the generated output. As the generated output always include the prompt, the correct prompt should be \"Hugging Face is an open-source company\".",
    "context": "{'commit_hash': 'e1f379bb090390acb62d36b7037cd93950ed322b', 'author': 'Xiaojian Ma', 'date': '2025-04-18T12:50:17-07:00', 'message': 'Fixing the example in generation strategy doc... \".'}",
    "source": "git_commit"
}
```

The offline data can be seen at [https://github.com/Jasonzzzz28/Transformer-QA/tree/main/data/offline_data](https://github.com/Jasonzzzz28/Transformer-QA/tree/main/data/offline_data) as well as in the object store. 

**Data pipelines**

Extract raw data from source code, git commit history and real world QA data. Detailed implementation can be seen [here](https://github.com/Jasonzzzz28/Transformer-QA/blob/main/data/data_pipeline/extract_data.py)
```bash
docker compose -f data/data_pipeline/docker-compose-etl.yaml run extract-data
```
Transform raw data into QA data with context
- Source Code: Question is generated using the paraphrase of "What does the {type} {name} do?". Answer is generated Use Ministral-8B-Instruct-2410 to summarize the usage of the function/class given the docstring.
- Git Commit History: Question is generated using the paraphrase of "What changed in commit {hash}?", "Who is the author of the commit {hash}?" and "When was the commit {hash} made?". Answer is generated using rule base extraction.
- Real World QA Data: Use real world question and answer to construct the QA pairs.

Then source Code and Git Commit History is combined as the **training data** and the **evaluation data** for model training and offline evaluation. Real World QA Data is splitted into **online data** and **product data** for online evaluation and product evaluation(close the feedback loop)

Detailed implementation can be seen [here](https://github.com/Jasonzzzz28/Transformer-QA/blob/main/data/data_pipeline/transform_data.py)
```bash
# Start Ministral-8B-Instruct-2410 vllm server for data transformation
bash data/data_pipeline/vllm_server.sh
docker compose -f data/data_pipeline/docker-compose-etl.yaml run transform-data
```
Load the entire offline data into object store.
```bash
export RCLONE_CONTAINER=object-persist-jc13140
docker compose -f data/data_pipeline/docker-compose-etl.yaml run load-data
```

**Online data**

To simulate real-world usage and support real-time inference, we use real world question from [huggingface channel of stackoverflow](https://stackoverflow.com/questions/tagged/huggingface) as the online data. These data are divided into [online data](https://github.com/Jasonzzzz28/Transformer-QA/blob/main/data/online_data/online_data.json) and [product data](https://github.com/Jasonzzzz28/Transformer-QA/blob/main/data/online_data/production_data.json). Online data will be used for online evaluation as stated in **Canary Online Evaluation** section using the [script](https://github.com/Jasonzzzz28/Transformer-QA/blob/main/evaluation_monitor/online_eval/simulate_traffic.py). Product data will be used to **Close the Feedback Loop**.

**Difficulty points: Interactive data dashboard**

To provide transparency and insight into the internal QA dataset and system performance, we develop an interactive data dashboard:
- Transform QA data(json) into SQL database
- Use Grafana with sqlite plugin to build the interactive data dashboard. User can use SQL query to interact with the dashboard.
A live demo can be seen at [http://129.114.26.254:3000](http://129.114.26.254:3000)
<img width="1440" alt="截屏2025-05-11 19 54 05" src="https://github.com/user-attachments/assets/eb7ddab6-85d1-4e6a-a0f2-f1c1a0bebe4d" />

Detailed implementation can be seen [here](https://github.com/Jasonzzzz28/Transformer-QA/tree/main/data/data_dashboard).

```bash
# Run Grafana dashboard on http://129.114.26.254:3000
docker compose -f data/data_dashboard/docker-compose-dashboard.yml up -d --build
```

#### Continuous X




