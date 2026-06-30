# INBOX — JournalClub.io Paper Scan
Last updated: 2026-06-30 · 1 new paper(s) scored this run

## New Papers (this run)

### ⚪ [Bayesian deep learning for uncertainty aware medical image segmentation correction and feature engineering: Applications to OCT imaging] — Score 3/10
> Domain-specific medical imaging paper (Ball et al., University of Liverpool, Informatics in Medicine Unlocked, 2026) modifying a U-Net segmentation pipeline for OCT retinal imaging to output per-pixel epistemic and aleatoric uncertainty maps via Bayesian weight sampling (64 MC-dropout forward passes) — no LLM, RAG, retrieval, or agent pipeline component; uncertainty quantification concepts (epistemic vs aleatoric) have abstract applicability to RAG confidence calibration but the implementation is entirely domain-specific ophthalmology with no drop-in applicability to AI engineering production systems.

## Previously Scanned (2026-06-29)

### ⚪ [Towards an MLOps infrastructure for Elastic Pattern experimentation] — Score 3/10
> MLOps platform paper (Rodriguez-Cardos et al., University of Castilla-La Mancha, SoftwareX, 2026) presenting an Airflow + MLflow + Docker pipeline that auto-generates and runs the 84 parameter-grid configurations of Elastic Pattern classifiers (7 feature representations × 3 aggregation strategies × 4 deformation methods) in parallel, logging results for cross-configuration comparison — no LLM, RAG, retrieval, or agent pipeline component; the MLOps orchestration pattern (Airflow DAGs + MLflow tracking) is generic and well-established, domain is a niche cognitive-psychology-inspired classification framework with no drop-in applicability to AI engineering production systems.

## Previously Scanned (2026-06-28)

### ⚪ [Detecting Uncoded Self-Harm in Veterans' Electronic Health Records Using Positive and Unlabeled Learning: Retrospective Cohort Study] — Score 2/10
> Domain-specific medical informatics paper (Kumar et al., University of New Mexico Health Sciences Center, Journal of Medical Internet Research, 2026) applying Positive Unlabeled (PU) Learning via the PULSNAR algorithm (XGBoost-based classifier with biased-sample correction and subgroup clustering) to detect undercoded self-harm events in Veterans Health Administration electronic health records — no LLM, RAG, retrieval, or agent pipeline component; PU learning framing has abstract relevance to training-data annotation problems but the core domain is clinical informatics not drop-in applicable to AI engineering production systems.

## Previously Scanned (2026-06-27)

### ⚪ [Mitigating data incompleteness in additive manufacturing using imputation techniques] — Score 2/10
> Domain-specific manufacturing engineering paper (Mojumder et al., Washington State University, Machine Learning: Engineering, 2026) benchmarking ten data imputation techniques (mean, median, mode, kNN, MICE, SVD, random forest, GMMs, XGBoost, VAEs) on laser powder bed fusion additive manufacturing datasets with controlled missing-data scenarios — no LLM, RAG, retrieval, or agent pipeline component; pure domain-specific materials-science imputation study with no drop-in applicability to AI engineering production systems.

## Previously Scanned (2026-06-26)

### ⚪ [A hierarchical Bayesian approach for predictive analytics of depression severity in medical students] — Score 2/10
> Domain-specific healthcare analytics paper (Chen et al., Dartmouth College, Healthcare Analytics, 2026) applying hierarchical Bayesian regression with partial pooling to predict PHQ-9 depression-severity scores in medical students using Healthy Minds survey covariates — no LLM, RAG, retrieval, or agent pipeline component; pure domain-specific medical/statistical modeling not drop-in applicable to AI engineering production systems.

### ⚪ [Practical guidelines for multiple instance learning in computational pathology] — Score 2/10
> Medical imaging paper (Miccolis et al., University of Modena, Frontiers in Bioinformatics, 2026) benchmarking patch-level encoders (ResNet50, ProvGigaPath, UNI, CONCH) and slide-level encoders (TITAN) for whole-slide image survival prediction across six cancer cohorts — no LLM, RAG, or retrieval pipeline component; domain-specific computational pathology with no drop-in applicability to AI engineering production systems.

## Previously Tracked (2026-06-25)

### 🔴 [GrepSeek: Direct Corpus Interaction — Hybrid Search for AI Agents] — Score 9/10
> Drop-in applicable: DCI bypasses vector embeddings for exact-match retrieval (error codes, version constraints, file paths) while semantic retrieval handles broad recall — GrepSeek fuses both via RRF achieving nDCG@10=0.847 (+19% vs pure semantic RAG). Direct POC: implement a hybrid search demo with BM25 semantic layer + regex DCI layer + RRF fusion over a corpus containing both natural-language docs and exact identifiers.
> **Source:** AlphaSignal newsletter deep-dive · 2026-05-31 · Scored via Gmail connector

**POC idea:** Gradio demo with three search modes (semantic / DCI / hybrid) over a corpus of AI engineering documents — showing exactly where semantic search fails on exact strings and DCI recovers them.
**Implemented:** Yes — `hf-space/grepseek_dci/` · Tab 12 in `hf-space/app.py` (2026-06-25)

### ⚪ [Transfer learning for T-cell response prediction] — Score 2/10
> Domain-specific bioinformatics/immunology paper (Stadelmaier et al., Eberhard Karls University of Tübingen, BMC Bioinformatics, 2026) building transformer-based models (BASE-T, ADA-T with adversarial domain adaptation, FINE-T with per-source fine-tuning) to predict T-cell immune responses to peptide sequences. Key contribution is a domain-aware evaluation scheme that detects shortcut learning by computing group-level AUC within peptide-source × MHC-allele strata — removing the ability to cheat by learning dataset composition instead of biology. The shortcut-learning detection framework has abstract relevance to LLM benchmark inflation, but the core domain (peptide immunogenicity, MHC binding) is purely medical and not drop-in applicable to AI engineering production systems.

## Previously Tracked (this run)

### ⚪ [ELY-SLAM: An efficient SLAM algorithm with lightweight YOLO-based collaborative perception] — Score 1/10
> Domain-specific robotics/SLAM paper (Zhao et al., UCSI University, Array, 2026) proposing ELY-SLAM — a dynamic-environment SLAM framework layering LSR-YOLO bounding-box detection, Lucas-Kanade pyramidal optical flow, and PROSAC outlier rejection on top of ORB-SLAM3 to filter moving-object features before pose estimation — no LLM, embeddings, RAG, or retrieval pipeline component; pure robotics/computer-vision engineering with no drop-in applicability to AI engineering production systems.

## Previously Tracked (this run)

### ⚪ [No cancer left behind: a testbed and demonstration of concept for photoacoustic tumor bed inspection] — Score 1/10
> Domain-specific surgical robotics / medical imaging paper (Connolly et al., Queen's University, Computer Assisted Surgery, 2026) presenting an experimental testbed combining a da Vinci robot, photoacoustic laser excitation, external ultrasound detection, RGB-D depth sensing, electromagnetic tracking, and 3D Slicer navigation to prototype and compare intraoperative tumor-bed inspection workflows — no LLM, RAG, retrieval, or agent pipeline component; pure domain-specific medical imaging engineering with no drop-in applicability to AI engineering production systems.

## Previously Tracked (this run)

### ⚪ [Visual navigation method for intelligent flight of highway inspection unmanned aerial vehicle] — Score 1/10
> Domain-specific robotics/computer vision paper (Song et al., Inner Mongolia University of Technology, Machine Learning: Engineering, 2026) presenting UAV-YOLO — a YOLOv3-tiny-based perception-to-control pipeline combining anchor box reclustering, SPPCSP-M spatial pyramid pooling, spatial attention, CIOU-NMS, and DeepSORT tracking to autonomously fly a drone along highway centerlines for road inspection — no LLM, RAG, retrieval, or agent pipeline component; domain-specific autonomous UAV navigation with no drop-in applicability to AI engineering production systems.

## Previously Tracked (this run)

### ⚪ [Reference-free k-mer based dissimilarity measures for metagenomes comparison] — Score 1/10
> Domain-specific bioinformatics paper (Gallina et al., University of Padova, Frontiers in Bioinformatics, 2026) testing whether Bray-Curtis and Jaccard dissimilarity metrics remain reliable when applied to reference-free k-mer profiles of metagenomic samples instead of species-count profiles — no LLM, RAG, retrieval, or agent pipeline component; pure metagenomics/ecology application with no drop-in applicability to AI engineering production systems.

## Previously Tracked (this run)

### ⚪ [DFL-RUL: Decentralized Federated Learning for Battery Remaining Useful Life Estimation on Heterogeneous Edge-to-cloud] — Score 2/10
> Domain-specific EV battery degradation paper (Pournazari et al., University of Warwick, Energy and AI, 2026) building a decentralized federated LSTM system to estimate remaining battery life across heterogeneous edge-to-cloud clients using FedAvg aggregation — no LLM, RAG, retrieval, or agent pipeline component; domain-specific battery engineering with no drop-in applicability to AI production systems.

## Previously Tracked (this run)

### ⚪ [Predicting fishing vs. not-fishing in small-scale fisheries: A sample vessel tracking dataset and a reproducible machine learning approach] — Score 1/10
> Domain-specific fisheries application (Lattanzi et al., Institute of Marine Biological Resources and Biotechnologies, SoftwareX, 2026) building a binary GPS-trajectory classifier (Logistic Regression, Decision Tree, Random Forest, XGBoost) to distinguish fishing from non-fishing vessel movement in small-scale fleets — no LLM, RAG, retrieval, or agent pipeline component; domain-specific marine biology / fisheries monitoring application not drop-in applicable to AI engineering production systems.

## Previously Tracked (this run)

### ⚪ [A Scene-Adaptive Super-Resolution Framework for Video Compression] — Score 2/10
> Domain-specific video compression paper (Zha et al., Jinan University, Journal of Imaging, 2026) proposing SASVC — a pipeline that downsamples video, encodes it with a standard codec, and reconstructs it at the receiver using a frozen SR backbone augmented with scene-adaptive LoRA-style adapter modules whose compressed deltas are transmitted as a side channel — no LLM, RAG, retrieval, or agent pipeline component; video-engineering technique not drop-in applicable to AI production systems.

## Previously Tracked (this run)

### ⚪ [Scalable Platform Enabling Reservoir Computing With Nanoporous Oxide Memristors for Image Recognition and Time Series Prediction] — Score 2/10
> Hardware neuromorphic computing paper (Donald et al., Loughborough University, Advanced Intelligent Systems, 2026) presenting a physical nanoporous niobium oxide memristor as a reservoir computing substrate for time-series prediction and image recognition — no LLM, RAG, retrieval, or agent pipeline component; domain-specific neuromorphic hardware engineering with no drop-in applicability to AI engineering production systems.

## Previously Tracked (this run)

### 🟡 [Concept drift detection in district heating: A characterization framework and detector-centric benchmark] — Score 5/10
> Domain-specific concept drift benchmarking paper (Derzsi et al., RWTH Aachen University, Journal of Energy and AI, 2026) proposing a 4-stage characterization framework (data processing → representation → dissimilarity measurement → hypothesis testing) for systematically comparing drift detectors on district heating load forecasting data — no LLM, embeddings, or retrieval component; the framework is transferable to RAG pipeline drift monitoring but requires significant adaptation, and the evaluated detectors performed poorly even in the target domain.

**POC idea:** Apply the 4-stage characterization framework to systematically evaluate and compare concept drift detectors for monitoring RAG pipeline performance degradation (embedding score drift, BM25 distribution shift, alpha drift over time).

## Previously Tracked (this run)

### ⚪ [An automatic detection model for spread through air spaces in postoperative pathological sections based on deep learning in NSCLC] — Score 2/10
> Domain-specific computational pathology paper (Zhang et al., Harbin Medical University) presenting STAS-Net, a two-stage pipeline combining DeepLabv3+ (5× tumor segmentation) and modified YOLOv7 (40× STAS detection with spatial-channel cross-attention and adaptive spatial feature fusion) for lung cancer air-space spread detection — no LLM, RAG, retrieval, or agent pipeline component; medical computer vision with no drop-in applicability to AI engineering production systems.

### ⚪ [Optical Flow Enables Hand Tracking With EyeGlove Low-Cost Cameras in Confined Environments] — Score 1/10
> Domain-specific industrial inspection hardware paper (Sun et al., University of Nottingham) introducing EyeGlove, a wearable finger-mounted dual-camera system using Lucas-Kanade sparse optical flow for hand tracking inside turbine engines — no LLM, embeddings, RAG, or agent pipeline component; robotics/hardware engineering with no drop-in applicability to AI engineering production systems.

## Previously Tracked (this run)

### ⚪ [A Flexible and Energy-Efficient Compute-in-Memory Accelerator for Kolmogorov–Arnold Networks] — Score 2/10
> Hardware chip design paper (Sudarshan et al., Peter Grunberg Institute) introducing KA-CIM, a compute-in-memory accelerator purpose-built for Kolmogorov–Arnold Networks (KANs) using piecewise-linear function approximation and memristive 1T1R cells with RC-discharge sensing; reports ~2000× energy-delay improvement over CPU for KAN inference — domain-specific hardware/chip-design paper with no LLM, RAG, retrieval, or agent pipeline component; not drop-in applicable to AI engineering production systems.

### ⚪ [Terrestrial Cyborg Insects for Real-Life Applications] — Score 1/10
> Domain-specific biorobotics review paper (Le et al., University of Queensland) cataloging locomotion control, onboard sensing, autonomous navigation, power systems, and real-world applications of cyborg insects (cockroaches, beetles, grasshoppers) equipped with electronic backpacks for search-and-rescue and hazard detection — no LLM, embeddings, RAG, or agent pipeline component; bioengineering/robotics domain with no drop-in applicability to AI engineering production systems.

## Previously Tracked (this run)

### ⚪ [Characterizing wind-dependent low-frequency ambient sound with ocean observatories initiative hydrophones] — Score 1/10
> Domain-specific ocean acoustics paper (Ragland et al., Woods Hole Oceanographic Institution) using 8-year OOI hydrophone recordings to characterize wind-speed vs. low-frequency ambient sound relationships via piecewise regression across two seafloor sites — purely geophysical signal processing study with no LLM, embeddings, RAG, or agent pipeline component; oceanography domain with no drop-in applicability to AI engineering production systems.

### ⚪ [Carbon-aware scheduling and distributionally robust optimization for cloud systems: forecasting, battery storage, and adaptive ambiguity refinement] — Score 2/10
> Operations research paper applying DRO + CVaR + Benders decomposition to carbon-aware cloud job scheduling (Ruparel et al.); LSTM used only for electricity/carbon intensity time-series forecasting — no LLM, embeddings, RAG, or agent pipeline component; domain-specific cloud infrastructure green computing optimization with no drop-in applicability to AI engineering production systems.

## Previously Tracked (this run)

### ⚪ [Alert System for Non-Responsive State of an Elderly Person] — Score 1/10
> Classical computer vision pipeline (ORB + K-means + Random Forest + Farneback optical flow) for elderly healthcare monitoring (Mane et al., Vishwakarma Institute of Technology) — no LLM, embeddings, retrieval, or RAG component; pure domain-specific healthcare application not drop-in applicable to AI production systems.

### ⚪ [A Formation Inversion Algorithm Based on Collaborative Fuzzy Gradient Neural Dynamics for Natural Gamma Logging While Drilling] — Score 1/10
> Domain-specific oil & gas drilling paper (Liu et al., Lanzhou University) framing geological formation inversion as a constrained nonconvex optimization problem solved by CFGND — a hybrid fuzzy-logic + gradient-neural-dynamics algorithm combining population-based global search with local refinement. No LLM, retrieval, RAG, or agent component; domain is geophysics/drilling engineering with no drop-in applicability to AI production systems.

### 🟡 [Analogical reasoning with large language models: a co-creative framework and benchmarking of LLMs in design ideation] — Score 6/10
> LLM-driven co-creative agent (ALIA, Kokate et al., IIT) using structured synectics stages (problem identification → keyword exploration → cross-domain analogy excursion → dialectical conflict resolution) with real-time STT+summarization; LLM is the core engine but the domain is design ideation, not RAG/retrieval/evaluation — applicable with adaptation to multi-stage agent query expansion workflows.

**POC idea:** Multi-stage analogical query expansion agent using structured 4-phase prompting (problem framing → keyword expansion → cross-domain analogy → dialectical synthesis) to enrich RAG queries with diverse retrieval perspectives.

### ⚪ [A machine learning approach using autoencoders to perform quality control on meteorological data] — Score 2/10
> Domain-specific meteorological QC application (Spohn et al., Met Éireann) comparing standard autoencoders, VAEs, and LSTM autoencoders for anomaly detection in weather station time-series — LSTM-AE wins but the contribution is entirely within the meteorology/QC domain; no LLM, retrieval, RAG, or agent component, and not drop-in applicable to AI engineering production systems.

## Previously Tracked (older runs)

### ⚪ [Deep learning framework for enhanced neutrino reconstruction of single-line events in the ANTARES telescope] — Score 1/10
> Domain-specific particle physics paper (Albert et al., University of Strasbourg) applying a modular deep CNN (N-fit) to reconstruct direction, position, energy, and event type from single-line hits in the ANTARES underwater neutrino telescope — no LLM, retrieval, RAG, or agent component; domain-specific astrophysics engineering not drop-in applicable to AI production systems.

### ⚪ [Principal component analysis and K-means clustering of fuel–air mixing in gas turbine combustors] — Score 2/10
> Domain-specific computational fluid dynamics paper (Salvador-Jasin et al., Loughborough University) applying PCA, K-means, and Large-Eddy Simulation to analyze fuel-air mixing patterns in gas turbine combustors — no LLM, retrieval, RAG, or agent component; domain-specific aerospace engineering not drop-in applicable to AI production systems.

### ⚪ [Comparison of Three Optimization Algorithms for Designing a Robust Power System With an Advanced Heffron-Phillips Model] — Score 2/10
> Pure power-engineering paper comparing Genetic Algorithm, Moth-Flame Optimization, and Snake Optimization for tuning a Power System Stabilizer (PSS) against Low-Frequency Oscillations in electrical grids; no LLM, retrieval, RAG, or agent component — domain-specific electrical engineering with no drop-in applicability to AI production systems.

### ⚪ [A Temporally Disentangled Contrastive Diffusion Model for Spatiotemporal Imputation] — Score 3/10
> Domain-specific application of diffusion models to spatiotemporal sensor imputation (traffic and air-quality datasets); uses trend/seasonal disentanglement and GNN spatial conditioning with contrastive loss — no LLM, retrieval, RAG, or agent component; technique not drop-in applicable to AI engineering production systems.

### ⚪ [TricP: A Novel Approach for Human Activity Recognition Using Tricky Predator Optimization Based on Inception and LSTM] — Score 2/10
> Domain-specific HAR application using a bio-inspired metaheuristic (Tricky Predator Optimization, coyote-hunting-inspired population search) to tune Inception-LSTM hyperparameters for surveillance video activity recognition; no LLM, retrieval, RAG, or agent component.

### ⚪ [AI-driven dynamic electronic road pricing to reduce air pollution and traffic congestion] — Score 2/10
> Domain-specific transportation policy paper using evolutionary multi-objective optimization (NSGA-II/III) to generate Pareto-optimal dynamic road-toll schedules balancing pollution, congestion, and driver cost. Interesting optimization framing but purely an urban-mobility/policy application — no LLM, retrieval, RAG, or agent component.

### ⚪ [A Fuzzy-IoT Irrigation Framework with LoRa-Based Connectivity for Climate Resilient Agriculture in Semi-Arid Regions] — Score 1/10
> Domain-specific IoT/agriculture application combining fuzzy logic decision-making with LoRa wireless communication for smart irrigation management in water-scarce regions; no LLM, retrieval, or RAG component — pure embedded-systems engineering for agricultural sensor networks.

### ⚪ [Multi-Scale Spatio-Temporal Transformer Network for Intelligent Healthcare and Transportation Systems: A Generative AI Approach] — Score 3/10
> Domain-specific spatio-temporal traffic prediction system for emergency vehicle routing; integrates hospital-rhythm signals (shift changes, clinic hours) into GNN+attention forecasting — not an LLM/RAG/retrieval technique despite the "Generative AI" title.

### ⚪ [sFWI: physics-informed score-based generative modeling for robust full waveform inversion] — Score 2/10
> Domain-specific geophysics application (seismic full waveform inversion) using score-based diffusion models; the technique is innovative but the domain is earth sciences, not LLM evaluation, RAG pipelines, or retrieval engineering.

### ⚪ [Self-adaptive weighting and sampling for physics-informed neural networks] — Score 1/10
> Physics-Informed Neural Networks (PINNs) for solving PDEs in physics simulations; this is a domain-specific scientific ML technique with no connection to LLM evaluation, RAG pipelines, or retrieval engineering.

### ⚪ [Dynamic Facial Expression Recognition of Learners via Adaptive Global Attention and Differential Temporal Transformer] — Score 3/10
> Domain-specific computer vision application for educational facial expression tracking using a hybrid spatial-temporal transformer; no LLM, retrieval, or RAG component makes it non-applicable to AI engineering production systems.

### 🔴 [TopoMAS: Large Language Model Driven Topological Materials Multi-Agent System] — Score 8/10
> LLM-driven multi-agent system (CoreAgent → KnowledgeAgent → PropertyAgent → SynthesisAgent → ValidationAgent) for topological materials discovery. Directly implements a hierarchical agent-routing architecture with knowledge-graph retrieval, HITL feedback loops, and automated DFT failure recovery — all patterns applicable to production RAG orchestration and LLM agent pipelines. **IMPLEMENTED (Tab 9)**

### ⚪ [Reading between the green lines: A framework for scalable greenwashing detection] — Score 5/10
> NLP pipeline (VADER sentiment + embedding cosine similarity) for detecting greenwashing in corporate sustainability reports. Uses embeddings and cosine similarity relevant to vector search, but the core contribution (GTS formula) is domain-specific and not directly applicable to RAG or LLM evaluation.

## Previously Tracked

_Papers from earlier scans are appended here automatically._

- ✅ **Query-Adaptive Hybrid Search** (score 10/10) — **IMPLEMENTED (Tab 1)** — The paper describes a technique to dynamically predict the right alpha for hybrid retrieval (IDF-based alpha computation), directly applicable to RAG and other retrieval-based systems.

- ✅ **CAD score: context-aware semantic diversity for scalable creativity evaluation of LLMs** (score 9/10) — **IMPLEMENTED (Tab 2)** — The paper presents a direct evaluation metric for LLM outputs, specifically tailored for assessing semantic diversity in creative tasks.

- ✅ **Dynamic Sub-Sequence Warping: A Representation-Based Similarity Measure for Long Time Series** (score 9/10) — **IMPLEMENTED (Tab 3)** — This paper proposes a dynamic sequence warping technique that is applicable to comparing and matching long time series data, relevant to implementing retrieval pipelines and AI agents.
- ✅ **The attribution crisis in LLM search results: Estimating ecosystem exploitation** (score 8/10) — **IMPLEMENTED (Tab 5)** — The paper discusses ecosystem exploitation in LLM search results, which is relevant to AI engineers implementing and evaluating retrieval pipelines and LLM applications.
- ✅ **Overcoming labelled data scarcity for defect classification in scanning tunnelling microscopy** (score 8/10) — **IMPLEMENTED (Tab 10)** — The paper addresses labelled data scarcity in defect classification, which could be an issue in implementing computer vision applications including those that utilize LLMs and retrieval pipelines.
- ✅ **Group-Based Recommendation System Using Bi-Stage Adaptive Deep Learning Model** (score 8/10) — **IMPLEMENTED (Tab 7)** — The proposed bi-stage adaptive deep learning model for group recommendations is clearly applicable to implementation in LLM applications and retrieval pipelines.
- ✅ **A synergistic engine paradigm for real-time, context-aware decision-making: integrating declarative processes and event streams** (score 8/10) — **IMPLEMENTED (Tab 4)** — This paper's technique for integrating declarative processes with event streams could enhance the context-aware decision-making capabilities of retrieval pipelines and RAG systems.
- ✅ **Opinion Maximization Based on Fairness in Social Networks** (score 8/10) — **IMPLEMENTED (Tab 8)** — The paper presents a framework to maximize overall positive opinion in social networks while ensuring fairness for minority groups, applicable to graph-RAG seed node selection and knowledge graph traversal with fairness guarantees.
- ✅ **Predictability of precipitation extremes in the Western United States and the U.S. Gulf Coast using AI weather models** (score 8/10) — **IMPLEMENTED (Tab 11)** — The paper explores the application of ML-based alternatives to NWP, focusing on forecasting extreme events, which is relevant to the development of LLM applications and retrieval pipelines. POC: Extreme Event AI reliability calibration for RAG edge-case queries.
- ✅ **OLR-WAA: Adaptive and Drift-Resilient Online Regression with Dynamic Weighted Averaging** (score 8/10) — **IMPLEMENTED (Tab 6)** — This paper applies to online regression, which is relevant to implementing dynamic estimation models for user properties values, a real estate application.
- 🟡 **Industrial chain risk monitoring with explainable dynamic knowledge graph learning** (score 7/10) — The paper applies explainable dynamic knowledge graph learning to industrial chain risk monitoring, which is similar to retrieval pipeline applications in AI engineering.
- 🟡 **KALEFormer a transformer based model integrating KMeans clustering and LDA for market basket analysis** (score 7/10) — The paper proposes a transformer-based model integrating KMeans clustering and LDA for market basket analysis, applicable to AI engineering work in retail recommendation systems.
- 🟡 **Detecting Phishing Emails Using GloVe Word Embeddings and Ensemble Machine Learning Model** (score 7/10) — The paper combines GloVe word embeddings with an ensemble machine learning model for phishing email detection, which involves techniques relevant to NLP and ML engineering work, such as embedding usage and model stacking.
- 🟡 **Reinforcement Learning-Driven Enterprise Financial Management and Spatiotemporal Convolution Optimization Model** (score 7/10) — The paper applies reinforcement learning to financial management, which may involve optimization and retrieval pipelines relevant to AI engineering work.
- 🟡 **Drop-in density-aware partitioning for tree-based anomaly detection** (score 7/10) — It improves the Isolation Forest algorithm, which can be relevant for anomaly detection in AI agents or agent decision-making processes.
- 🟡 **Efficient multi-robot exploration of unknown environments using inverted ant colony optimization and reinforcement learning** (score 7/10) — The paper applies reinforcement learning and a unique optimization algorithm to coordinate multiple robots, showing potential applications in complex search and discovery tasks, but doesn't directly apply to production systems I implement.
- 🟡 **VogFashion: an AI-driven intelligent system for fashion design** (score 6/10) — The paper touches on generative AI pipeline for fashion design, but it doesn't directly apply to daily tasks like RAG, LLM evaluation, or vector search.
- 🟡 **Generative AI for seismic data enhancement: a novel workflow using stable diffusion models** (score 6/10) — The paper explores the application of Stable Diffusion models in enhancing seismic data, which can be related to AI engineering work in retrieval pipelines and LLM applications but requires some abstraction.
- 🟡 **Grey Wolf Optimization and Deep Belief Networks for Data-Efficient Forecasting in Smart Renewable Energy Systems** (score 6/10) — The paper combines Deep Belief Network and Grey Wolf Optimization to create a data-efficient forecasting pipeline, which seems tangentially related to AI engineering work such as LLM evaluation and retrieval pipelines.
- 🟡 **A lightweight approach to software fault localization using static features of statements in cloud computing environments** (score 6/10) — The paper presents an approach using static features of statements in cloud computing environments for software fault localization, which is relevant to AI engineering work related to software testing and debugging.
