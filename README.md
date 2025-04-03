
## Machine Learning System Engineering

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

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. Name of data/model, conditions under which it was created (ideally with links/references), conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| Transformers QA dataset   |     Created by ourselves (see detail in data pipeline section)               |    Training and evaluation               |
| Stanford Question Answering Dataset (SQuAD)  |      Downloaded from Hugging Face              |       Training            |
| Qwen2.5-14B-Instruct-1M |       Downloaded from Hugging Face             |      Base QA model             |
| BGE-M3          |       Downloaded from Hugging Face             |      Embedding model for RAG             |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), how much/when, justification. Include compute, floating IPs, persistent storage. The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     |            |
| `gpu_a100`     | 2 A100 for 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the diagram, (3) justification for your strategy, (4) relate back to lecture material, (5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, and which optional "difficulty" points you are attempting. -->



I. Training Pipeline Architecture

II. Core Implementation
  2.1 Model Selection & Training Strategy

    base model:

      python"
        model = AutoModelForCausalLM.from_pretrained(
            "codeQwen/CodeQwen2.5-14B-hf",  # Optimized for code understanding
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
  3.1 MLFlow Experiment Tracking

        with mlflow.start_run():
    mlflow.log_params({
        "max_seq_length": 4096,
        "grad_accum_steps": 4
    })
    mlflow.pytorch.log_model(
        pipeline, 
        "model",
        registered_model_name="code_qwen_qa"
    )
    
  3.2 Ray Cluster Configuration


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







#### Model serving and monitoring platforms

***Model Serving***

**Serving from an API Endpoint** 

- Backend: We will deploy our fine-tuned Qwen2.5-14B-Instruct model using NVIDIA Triton Inference Server, exposing it through a REST API. The API will accept JSON-formatted requests containing user questions and return predictions in JSON format.

- Frontend: A Flask-based web interface will handle user interactions and send requests to the NVIDIA Triton backend for real-time predictions.

- Endpoint Accessibility: The API will be hosted on Chameleon Cloud and accessible via a configurable URL.

**Requirements** 

- Model Size: The expected size of the trained and optimized Qwen2.5-14B-Instruct model is approximately 28GB (14 billion parameters * 2 bytes per parameter in FP16 precision). This model will be stored in persistent storage on Chameleon (as defined in Unit 8).

- Throughput (Batch Inference): We anticipate a relatively low batch inference requirement. We aim for 6 - 12 QPS.

- Latency (Online Inference): For real-time question answering, we aim to achieve a latency of 500ms - 1.5s per request. While not strictly real-time, this latency is crucial for a responsive and interactive user experience.

- Concurrency (Cloud Deployment): Our cloud deployment on Chameleon must support a concurrency of 8 simultaneous requests to handle concurrent users asking questions. This represents a minimal concurrency requirement for our prototype.

**Model Optimizations**

- Model Format: Convert to ONNX for broader optimization support.

- Graph Optimization: Utilize ONNX Runtime's graph optimizations.

- Quantization: Implement INT8/FP16 quantization, allowing a maximum accuracy loss of 0.01 (on validation set).

- Execution Provider: Benchmark and select the faster performer between CUDA and TensorRT execution providers within ONNX Runtime.

**System Optimizations**

- Model Server: Utilize Triton Inference Server with the ONNX backend for efficient execution of the optimized Qwen2.5-14B-Instruct model.

- Scaling: Deploy the model across 2 GPUs, with 2 Triton instances running on each GPU.

- Concurrency: Configure each Triton instance to handle a concurrency of 2, resulting in a total system concurrency of 8.

- Batching Strategy: Enable Triton's dynamic batching for efficient request aggregation.

<br>

***Evaluation and monitoring***

**Offline Evaluation**

Automated testing after training:

- Standard/Domain Use Cases: Evaluate on the Transformers documentation test set.

- Population/Slice Analysis: Analyze performance on different question categories and complexities.

- Known Failure Modes: Test for adversarial prompts, hallucinations, and reasoning limitations.

- Unit Tests: Verify basic factual questions.

- Metrics: Track F1, Exact Match, Precision, and Recall.

- Model Registration: Register new model only if it exceeds performance thresholds.

**Staging Load Test**

Using load testing frameworks, like Locust, simulate concurrent users and measure:

- QPS (Target: 6-12)

- Latency (Target: 500ms - 1.5s)

- Error Rate

- Resource Utilization (CPU/GPU)

- Optimize resources based on results

**Canary Online Evaluation** 

We will conduct online evaluations in a canary environment by role-playing distinct user personas (novice, experienced developer, researcher). We'll generate questions representative of each persona and assess the model's usefulness and accuracy on these simulated interactions, in addition to standard load test metrics (QPS, latency, error rate).

For example:

- Novice User: "What is a Transformer?" "How do I load a pre-trained model?"

- Experienced Developer: "How do I fine-tune a Trainer with FSDP?" "What are the expected input formats for pipeline?"

- Researcher: "What are the key differences between BERT and RoBERTa?" "How does the attention mechanism work in Longformer?"
  
- Transformer Enthusiast: "Which Transformer is stronger: Optimus Prime or Megatron?"

**Close the Loop** 

We will gather feedback about the quality of the model's predictions through:

- User feedback (thumbs up/down) within the front end.

- Periodic review and annotation of model outputs by human annotators to identify and correct errors, biases, or areas for improvement.

**Business Evaluation** 

- Tracking Active Users: Monitoring the number of daily/weekly active users interacting with the QA bot.

- Measuring Task Completion Rates: Analyzing whether users are able to more quickly resolve coding problems or understand the codebase with the help of the QA bot. This might be inferred from usage patterns, like time spent viewing documentation after a QA interaction.

- Conducting User Surveys: Collecting qualitative feedback from users regarding their satisfaction with the bot and how it impacts their workflow.

- Monitoring Support Ticket Volume: Observing if the QA bot leads to a reduction in the number of support tickets related to the Transformers library, indicating improved self-service and code understanding.

**Monitor for model degradation**

- User Feedback Tracking: Continuously monitor user ratings (thumbs up/down) to detect declines in satisfaction.

- Human Annotator Analysis: Track correction frequency to identify recurring model errors.

- Canary Deployments: Test new models with a small portion of live traffic while comparing performance against the current model to prevent regressions.

- Automatic Retraining: Set performance thresholds (e.g., rating drops or error increases) to trigger retraining with fresh labeled data.

<br>
<br>

#### Data pipeline

**Persistent storage**

Following the example of Lab 8, we will provision persistent storage on Chameleon, which may be attached to and detached from our infrastructure. This will store all materials that should persist for the duration of the project, but are not tracked in Git: model training artifacts, test artifacts, models, container images, data.

**Offline data**

In this project, we aim to construct a high-quality internal QA dataset using the [HuggingFace Transformers Project](https://github.com/huggingface/transformers) as the internal knowledge base. The goal is to extract meaningful question-answer pairs from various components of the project to support the training of a retrieval-augmented generation (RAG) QA system tailored to this domain.

The dataset will be built by parsing and processing multiple information sources within the repository, including:

- Source code: Function and class definitions with docstrings and inline comments will be parsed to generate technical "what" and "how" questions.

- Git commit history: Commit messages and associated diffs will be analyzed to produce "why" and "what changed" style QA pairs.

- GitHub issues and discussions: I will extract the issue title and comments to reconstruct problem-resolution dialogues, turning them into user-centered QA pairs.

- Pull requests and code reviews: Review comments and PR descriptions will be mined for code reasoning and decision-making insights.

- Documentation files: Official documentation and model cards including README of the github repository and the [documentation from huggingface website](https://huggingface.co/docs/transformers/en/index) will serve as another source of well-structured, factual QA content.

The QA dataset will be generated in the following ways: 
- Rule-based automatic generation (e.g., "What does this function do?", "Why was this line changed?").
- LLM generation by strict source-alignment prompts. All QA examples will include metadata linking back to their origin (e.g., file path, commit hash, issue number) to maintain traceability.
  - Using self-instruct to generate questions given each context.
  - Using proper prompt to guide the LLM to generate answer given the question and the context.
- Mannually design data for evaluation purpose
- We will also mix some high quality general QA data as our training data(won't be used in evaluation)

To ensure the data quality and reduce hallucinations:
- Most answer should be the original text from the knowledge base (easy to check correctness)
- We will also generate some QA data that will needs some summarization and reasoning ability. The correctness of these data should be validated first before being used in training.
- For LLM generated question, we will use [self-instruct](https://arxiv.org/abs/2212.10560) to make sure the high diversity of the questions.
- For LLM generated answer, we will use another answer validation model to ensure answer correctness.

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

The resulting dataset will be indexed using dense retrieval techniques and used to power a RAG-based LLM capable of answering technical questions about the Transformers codebase and its development history.

**Data pipelines**

Our data pipeline is designed to support continuous learning and adaptation of a RAG-based QA system built on the HuggingFace Transformers repository. It consists of two main components: upstream monitoring and data extraction, and downstream user feedback handling.

***Upstream: Continuous Monitoring and QA Pair Generation***

We implement an automated monitoring system that continuously tracks updates in the Transformers GitHub repository. The system detects:

- New commits

- Newly closed issues

- Merged pull requests

- Updated documentation

Upon detecting updates, the pipeline automatically extracts the relevant content and start the data generation pipeline. These QA pairs are stored temporarily in a staging buffer. Once a sufficient number of QA pairs are collected, they are added to the main QA dataset. This triggers an automatic retraining or fine-tuning process for the model, ensuring that the system stays up-to-date with the latest changes in the codebase and community discussions.

***Downstream: User Feedback and Iterative Refinement***

On the serving side, we continuously collect user feedback on the generated responses. When a user flags a response as incorrect or unhelpful, the associated QA query is logged along with the model's output and user-provided annotations (if any).

These flagged examples are periodically reviewed through two possible strategies:

Manual correction by domain experts or annotators

Automatic correction using a feedback-aware refinement model (e.g., instruction-following LLMs guided by the original context)

Corrected QA pairs are validated and added back into the dataset. Once enough feedback-corrected examples are accumulated, a new round of retraining is triggered. This enables the model to gradually improve over time by learning from its past mistakes and user interactions.

**Online data**

To simulate real-world usage and support real-time inference, I will implement a streaming data pipeline that handles online query data in near real-time.

The pipeline will consist of:

- An online query handler that receives incoming questions (e.g., via a REST API or simulated queue) and sends the queries to the RAG model.

- A monitoring layer to log incoming queries, model responses, and latency metrics for further analysis or retraining triggers.

- A simulation script that continuously sends pseudo-realistic user queries to feed into the system.

For the simulated online data, we will use the mannully design question as the simulated queries.

**Difficulty points: Interactive data dashboard**

To provide transparency and insight into the internal QA dataset and system performance, I will develop an interactive data dashboard for the team. This dashboard will help track the health, quality, and growth of the dataset over time, as well as monitor the behavior of the deployed QA system.

The dashboard will include:

Dataset Overview: Total number of QA pairs, categorized by source type (code, issues, commits, documentation, etc.) and time of generation.

Data Quality Metrics: Automatic scores such as LLM confidence, hallucination detection rate, and manual feedback stats (e.g., number of bad responses flagged by users).

Retraining Activity: Visual indicators of retraining cycles, including model versioning, training data volume, and before-after performance metrics.

Online Inference Monitoring: Real-time charts displaying incoming user queries (simulated or real), response latency, error rates, and trending questions.

Interactive Filtering: Team members will be able to filter the data by source, time range, model version, or quality score to perform targeted analysis.

The dashboard will serve as a central control panel for understanding and managing the evolving QA system.




#### Continuous X


**1. Continuous Integration (CI)**

* **Trigger:** Code (backend API logic, frontend Flask app, data processing scripts, evaluation scripts, etc.) committed to the Git repository.
* **Build & Test:**
    * **Code Testing:** Automatically run unit tests and integration tests covering the Flask frontend, API interaction logic, data pipeline scripts (e.g., QA generation, validation logic).
    * **Data Validation Testing:** Include steps in the pipeline to test data extraction, QA generation, and validation logic with a small sample dataset to ensure they work as expected.
    * **Model Basic Testing:** (If CI includes model building steps) Verify the model can be successfully loaded, accepts input in the expected format, and produces output in the expected format.
    * **Dependency Check:** Verify all dependencies are correctly installed and configured.
* **Artifacts:**
    * Tested code.
    * Built Docker images (containing application code and dependencies, but not necessarily the large model itself; the model might be pulled during the deployment phase).
    * Test reports.
* **Platforms/Tools:** Git (e.g., GitHub/GitLab), CI/CD service (e.g., Jenkins, GitLab CI, GitHub Actions), Pytest, Docker.

**2. Continuous Delivery/Deployment (CD)**

* **Trigger:** CI successfully completes, or a new model passes evaluation in the CT process and is registered.
* **Preparation Stage:**
    * **Model Conversion & Optimization:** Automated scripts execute model format conversion (ONNX), graph optimization (ONNX Runtime), and quantization (INT8), verifying accuracy loss (<0.01).
    * **Benchmarking:** Automatically run benchmarks to select the best execution provider (CUDA vs TensorRT).
    * **Packaging:** Package the optimized model, Triton configurations, and API service code into the final deployment unit (e.g., updated Docker image).
* **Deploy to Staging:**
    * **Infrastructure Provisioning:** (If needed) Use IaC tools (Terraform, Pulumi) to configure compute resources (GPU nodes), networking, and persistent storage on Chameleon Cloud.
    * **Service Deployment:** Deploy the packaged model and service to the Triton Inference Server in the Staging environment (across 2 GPUs, 2 instances per GPU, total concurrency 8).
    * **Apply Configuration:** Apply Triton configurations like dynamic batching.
* **Staging Validation:**
    * **Automated Load Testing:** Run load tests using Locust to verify QPS (6-12), latency (500ms - 3s), error rate, and resource utilization meet requirements.
* **Deploy to Production (Canary Release):**
    * **Traffic Splitting:** Route a small portion of production traffic (e.g., internal test traffic simulating specific user personas) to the newly deployed model version.
    * **Canary Online Evaluation:** Execute your designed Canary evaluation plan (questions from different user personas), monitoring model effectiveness (usefulness, accuracy) and performance metrics (QPS, latency, error rate).
* **Full Rollout/Rollback:**
    * If the Canary evaluation is successful, gradually shift all traffic to the new version.
    * If the Canary evaluation fails, automatically or manually roll back to the previous stable version.
* **Platforms/Tools:** CI/CD service, Docker, Kubernetes (or similar orchestrator), Triton Inference Server, ONNX Runtime, Locust, Chameleon Cloud API/CLI, Git (for version control and triggers).

**3. Continuous Training (CT)**

* **Trigger:**
    * **Upstream Data Updates:** Automated monitoring system detects new Commits/Issues/PRs/Docs in the Transformers repository, triggering data extraction and QA generation. Training starts when a sufficient number of QA pairs accumulate in the staging buffer.
    * **Downstream User Feedback:** Enough user feedback flagged as incorrect/low-quality (via frontend thumbs up/down or manual annotation) is collected, triggering feedback-based fine-tuning.
    * **Model Performance Degradation:** CM system detects key metrics (e.g., user satisfaction, manual correction frequency, Canary comparison performance) falling below thresholds.
    * **Scheduled Task:** Retraining occurs at fixed intervals (e.g., weekly, monthly).
* **Data Pipeline (Automated):**
    * **Extraction:** Automatically pull new content from Git/GitHub/website.
    * **Generation:** Run rule-based/LLM (including self-instruct) methods to generate QA pairs.
    * **Validation:** Use an LLM validation model or other methods to check answer correctness, especially for QA requiring reasoning/summarization.
    * **Storage & Indexing:** Add validated QA pairs (with metadata) to the main dataset in persistent storage and update the index used by RAG.
* **Model Training/Fine-tuning:**
    * Fine-tune the Llama 3.3 70B model using the updated dataset (or, depending on the RAG architecture, primarily update the index and potentially fine-tune the retriever/generator).
    * Use your defined training platform and process.
* **Model Evaluation (Automated):**
    * **Offline Evaluation:** After training completes, automatically run evaluations on the test set (including standard use cases, slice analysis, known failure modes, unit tests), calculating metrics like F1, EM, Precision, Recall.
* **Model Registration:**
    * If the new model's performance exceeds preset thresholds (compared to the previous version or a fixed standard), version it and register it in the model registry.
* **Platforms/Tools:** Git/GitHub API, Data processing frameworks (Pandas, Spark), LLM for generation/validation, Vector database/Index library (FAISS, Elasticsearch), ML Training frameworks (PyTorch, Transformers Trainer), MLFlow/Weights & Biases (for experiment tracking and model registry), Persistent Storage (Chameleon), Workflow orchestration tools (Airflow, Kubeflow Pipelines).

**4. Continuous Monitoring (CM)**

* **Monitoring Targets:**
    * **Model Service Performance:** QPS, latency, error rate, GPU/CPU/memory usage of Triton instances.
    * **Model Quality (Online):**
        * User Feedback: Real-time collection of thumbs up/down from the frontend.
        * Human Annotation: Track frequency and patterns of manual review and correction.
        * Canary Comparison: Performance comparison between new and old models on a live traffic subset.
    * **Data/Business Impact:**
        * (Via Dashboard) Track active users, task completion rate (proxy metrics), user satisfaction (surveys), changes in support ticket volume.
        * (Via Dashboard) Monitor query patterns, popular questions, uncovered or poorly answered topic areas.
* **Monitoring Methods:**
    * **Infrastructure Monitoring:** Use Chameleon Cloud or standard cloud monitoring tools.
    * **Application Performance Monitoring (APM):** Integrate APM tools or use logging and metrics endpoints from Triton/Flask.
    * **Log Aggregation:** Centralize and analyze logs from Triton, Flask application, and data pipelines.
    * **Interactive Dashboard:** Implement your designed dashboard to visualize dataset overview, quality metrics, training activity, and online inference monitoring.
    * **Alerting System:** Configure alerting rules to automatically notify the team when key metrics (e.g., high latency, surge in error rate, drop in user satisfaction) cross thresholds.
* **Feedback Loop:**
    * Monitored performance degradation or quality issues can automatically trigger alerts, rollbacks (if in Canary phase), or initiate the CT pipeline.
    * Data insights from the dashboard guide manual investigations, data annotation priorities, and future model improvement directions.
* **Platforms/Tools:** Prometheus/Grafana, ELK Stack (Elasticsearch, Logstash, Kibana), APM tools (Datadog, Dynatrace), User feedback database, Interactive dashboard frameworks (Streamlit, Dash, Grafana), Alerting systems (Alertmanager, PagerDuty).


