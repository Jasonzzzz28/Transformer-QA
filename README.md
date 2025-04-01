
## Machine Learning System Engineering

<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be used in an existing business or service. (You should not propose a system in which a new business or service would be developed around the machine learning system.) Describe the value proposition for the machine learning system. What’s the (non-ML) status quo used in the business or service? What business metric are you going to be judged on? (Note that the “service” does not have to be for general users; you can propose a system for a science problem, for example.)
-->

### Contributors

<!-- Table of contributors and their roles. First row: define responsibilities that are shared by the team. Then each row after that is: name of contributor, their role, and in the third column you will link to their contributions. If your project involves multiple repos, you will link to their contributions in all repos here. -->

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| Zeqiao Huang                 | model training |                                    |
| Siyuan Zhang                 |  model serving and monitoring |    https://github.com/Jasonzzzz28/ECE-GY-9183-Project/commits?author=Jasonzzzz28                                |
| Junzhi Chen                   |       data pipeline          |                                    |
| Chuangyu Xu                  |      continuous X pipeline           |                                    |



### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. Must include: all the hardware, all the containers/software platforms, all the models, all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. Name of data/model, conditions under which it was created (ideally with links/references), conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| Data set 1   |                    |                   |
| Data set 2   |                    |                   |
| Base model 1 |                    |                   |
| etc          |                    |                   |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), how much/when, justification. Include compute, floating IPs, persistent storage. The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the diagram, (3) justification for your strategy, (4) relate back to lecture material, (5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, and which optional "difficulty" points you are attempting. -->

#### Model serving and monitoring platforms

***Model Serving***

**Serving from an API Endpoint** 

- Backend: We will deploy our fine-tuned LLaMA-3.3-70B model using NVIDIA Triton Inference Server, exposing it through a REST API. The API will accept JSON-formatted requests containing user questions and return predictions in JSON format.

- Frontend: A Flask-based web interface will handle user interactions and send requests to the NVIDIA Triton backend for real-time predictions.

- Endpoint Accessibility: The API will be hosted on Chameleon Cloud and accessible via a configurable URL.

**Requirements** 

- Model Size: The expected size of the trained and optimized Llama 3.3 70b model is approximately 140GB (70 billion parameters * 2 bytes per parameter in FP16 precision). This model will be stored in persistent storage on Chameleon (as defined in Unit 8).

- Throughput (Batch Inference): We anticipate a relatively low batch inference requirement. We aim for 6 - 12 QPS.

- Latency (Online Inference): For real-time question answering, we aim to achieve a latency of 500ms - 3s per request. While not strictly real-time, this latency is crucial for a responsive and interactive user experience.

- Concurrency (Cloud Deployment): Our cloud deployment on Chameleon must support a concurrency of 8 simultaneous requests to handle concurrent users asking questions. This represents a minimal concurrency requirement for our prototype.

**Model Optimizations**

- Model Format: Convert to ONNX for broader optimization support.

- Graph Optimization: Utilize ONNX Runtime's graph optimizations.

- Quantization: Implement INT8 quantization, allowing a maximum accuracy loss of 0.01 (on validation set).

- Execution Provider: Benchmark and select the faster performer between CUDA and TensorRT execution providers within ONNX Runtime.

**System Optimizations**

- Model Server: Utilize Triton Inference Server with the ONNX backend for efficient execution of the optimized Llama 3.3 70b model.

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

- Latency (Target: 500ms - 3s)

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

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which optional "difficulty" points you are attempting. -->


