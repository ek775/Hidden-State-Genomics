---
title: "Hidden State Genomics: *Graph-Based Analysis of Sparse Auto-Encoder Feature Activity in Genomic Language Models*"
author: Eliot Kmiec, Samuel O'Brien, Matthew McCoy
date: April 10th, 2026 
subtitle: Georgetown University
geometry: margin=3cm
output: pdf_document
header-includes:
	- \usepackage{graphicx}
	- \usepackage{array}
---

## Abstract

Pre-trained genomic language model (gLM) representations have been anticipated to enable enhanced deep learning predictions on several genomics tasks, but current benchmarking has led to questions over what they actually encode. We studied this with mechanistic interpretability on InstaDeep’s Nucleotide Transformer v2 (500M), training sparse autoencoders across all 24 encoder layers to probe latent features. Correlation-based annotation against reference regulatory tracks was inconsistent across layers and insufficient for causal interpretation. We therefore built typed sequence-to-feature knowledge graphs to explore the SAE feature space and compared cisplatin-binding versus non-binding RNA sequence communities by PageRank centrality, validating candidate features with decoder-based interventions and a CNN binding classifier. Interventions showed asymmetric effects: suppressive features could collapse predictive signal, while binding-associated features shifted predictions cumulatively with the presence of other binding-associated signals. Dependency maps further indicated strong local feature sensitivity within sequences. Together, these results provide evidence that gLM representations encode highly granular sequence syntax and conservation patterns, aligning more strongly with tightly coupled molecular interactions and local biophysical constraints than with complex, distributed regulatory logic. Within the scope of our intervention setting, this pattern is consistent with stronger performance on selected molecular tasks and weaker performance on broader regulatory inference, motivating scalable methods for causal feature annotation.

## Background

The success of pre-trained foundation models in natural language processing tasks such as text-generation and language translation has led to the training of a wide array of Genomic Language Models (gLMs) based on Artificial Neural Networks (ANNs) [[1](#ref-1), [2](#ref-2)]. These models are trained on DNA sequences using a self-supervised training objective to learn generalized representations of the information contained within genomic sequence data. Typically, these training objectives mirror their natural language counterparts, which utilize autoregressive or causal language-modelling techniques to find maximum likelihood estimates of a series of masked positions in the sequence. This process has been shown in other domains to reliably produce an information-rich general representation of sequence data that can be utilized for multiple downstream predictive tasks. Thus, gLMs have been expected to yield similar benefits for predictive modelling on tasks where genomic sequence data is utilized [[3](#ref-3)].

Despite the success of foundation models in other contexts, such as the GPT models that form the basis of ChatGPT, gLMs have struggled to deliver similar performance benefits to some downstream applications in genomics, and often require significant computing resources allocated for fine-tuning in order to achieve desired performance [[4](#ref-4), [5](#ref-5)]. Due to the complexity of these models, published discourse on the limitations of gLMs currently focuses on performance benchmarking, and mechanisms are largely derived from intuition rather than empirical study. While the ANN architecture that these foundation models have been built on has long been considered a “black box”, significant advances have recently been made toward developing methods to understand their activation patterns and post-training behavior.

In 2024, researchers at Anthropic demonstrated the use of Sparse Auto-Encoders (SAEs) to extract interpretable latent vectors from the activation space of their flagship large language model, Claude [[6](#ref-6)]. This approach is based on the theory of superposition, which states that the perceptrons in an ANN represent features in the data using combinations of perceptron outputs rather than the activity of a single perceptron. The SAE allows the isolation of features encoded by the model by expanding the latent space and applying a sparsity constraint. SAEs typically achieve this by utilizing a single hidden layer with an L1 sparsity penalty applied, and scaling the hidden layer to a chosen multiple of the original embedding matrix dimensionality. An annotation process is then used to assign meaning to these new feature vectors, also referred to as latents, typically by providing sample activity from the latents to a Large Language Model (LLM) for automated analysis. To validate the labels, steering experiments are performed using the decoder from the SAE to construct embeddings with artificial latents and then fed through the original model to see what effect it has on the output. Two independent groups have so far successfully applied these methods to ESM-2, revealing interpretable features of proteins learned by the foundation model [[7](#ref-7), [8](#ref-8)].

Applying these methods to gLMs has the potential to address some major questions about their performance. In 2024, Boshar and colleagues compared performance of gLMs and Protein Language Models (pLMs), and found evidence that gLMs such as InstaDeep’s Nucleotide Transformer family of models could be used to predict properties of transcribed proteins with comparable performance to pLMs. Yet, in 2025, Tang and colleagues found that gLMs failed to recognize cell-specific regulatory motifs and some models had worse performance than convolutional neural networks (CNNs) prior to task-specific fine-tuning [[9](#ref-9), [4](#ref-4)]. This performance gap between the domain in which the models are trained and the domains in which they perform well seems to suggest a misalignment of pre-training incentives and desired feature encoding; however, specific knowledge of what gLMs encode is currently an open research question. Having knowledge of what is encoded by gLMs after pre-training would provide needed insight into the mode of failure that is producing the misalignment, as well as potentially revealing novel biological mechanisms.

## Model Selection and SAE Training

InstaDeep’s Nucleotide Transformer (v2) series of models were selected for this study, as the models are open source, readily accessible via the HuggingFace ecosystem, and have been utilized in multiple previous studies [[2](#ref-2)]. With the resources we had access to, we trained 72 SAEs on embeddings from all 24 layers of InstaDeep’s NTv2-500m-human-ref encoder (NTv2) using a set of approximately 20,000 sequences from the human reference genome (GRCh38/hg38, GENCODE v49). NTv2 is a traditional encoder-only transformer architecture, and the version we used was pre-trained on 500 million sequences from the human reference genome. All downstream models were trained with the NTv2 weights frozen for consistency. Each SAE is composed of an input layer whose dimensions match the NTv2 embedding dimension, a single hidden layer where the L1 sparsity penalty is applied to an expanded latent space of 8, 16, or 32 times the original embedding dimension; and a single decoding layer which reconstructs the original embeddings. These SAEs were trained using an L1 sparsity penalty of 0.001, and a linear annealing schedule of 100 steps as shown by Simon and Zou to gradually increase the L1 sparsity penalty for training stability [[7](#ref-7)].

## Limitations of Correlation Scores for Annotation

Use of LLMs to automate the annotation process for mechanistic interpretability studies on LLMs and pLMs is enabled by the natural language domain and readily accessible protein-specific metadata from the Protein Data Bank, respectively [[6](#ref-6), [7](#ref-7)]. DNA is not a human-readable language, and the functions of individual elements are often dependent on interactions with other elements in the genome, so the annotation process in gLMs is necessarily more complex. Earlier mechanistic interpretability experiments on gLMs have attempted to utilize correlation scores between NCBI reference tracks and SAE latents to label individual features; however, this approach has limitations [[1](#ref-1)].

Although the correlation approach is an intuitive adaptation, there is no inherent causality in correlations, thus, it is extremely difficult to design sound steering experiments to validate annotations generated from correlations. This poses a major epistemic limitation compared to information theoretic approaches such as attribution maps [[10](#ref-10)]. Additionally, it is also impossible to label anything other than previously understood biology by this method, thereby eliminating one of the anticipated benefits of interpretability study on gLMs.

In our experiments, we also find that correlation between latents and known references varies empirically across layers. We computed Pearson and normalized cross-correlation coefficients for each feature against GRCh38/hg38 RefSeq regulatory element tracks obtained from the UCSC genome browser. Feature signal per token was used to calculate these correlations and identify the most highly correlated feature for each regulatory element track as displayed in *Figure 1*. By plotting the best correlated feature scores for these regulatory elements layer by layer, we demonstrate how likely the latents are to extract the desired information and which layers extract this information better than others. While we do expect natural variations in per layer representations, we generally expect deeper layers of the encoder to capture more information and for the trend to be positively correlated with the layer depth.

\begin{center}
\begin{tabular}{cc}
\includegraphics[height=0.40\textheight]{figures/image1.png} &
\includegraphics[height=0.40\textheight]{figures/image2.png}
\end{tabular}
\end{center}

*Figure 1 \- Ridgeline density plot showing distribution of highest correlated features for the NCBI RefSeq regulatory element track set from the UCSC genome browser. Comparison between Pearson metric (left) and Normalized Cross-Correlation (right) to account for phase / amplitude differences. Black dotted lines indicate global average, red dotted lines indicate local layer-wise average. Colored bars represent quartiles.*

\begin{center}
\begin{tabular}{cc}
\includegraphics[height=0.40\textheight,keepaspectratio]{figures/image3.png} &
\includegraphics[height=0.40\textheight,keepaspectratio]{figures/image4.png}
\end{tabular}
\end{center}

*Figure 2 \- Ridgeline density plot showing distribution of highest correlated features by Normalized Cross-Correlation for subsets of the NCBI RefSeq regulatory element track set from the UCSC genome browser. Comparison between LINE subset (left) and PBS (right). Black dotted lines indicate global average, red dotted lines indicate local layer-wise average. Colored bars represent quartiles.*

Contrary to the expected pattern, after the initial spike in the second layer, the SAEs struggle to find latents that capture the labeled regulatory elements. This suggests that either the hypothesis by Tang and colleagues [[4](#ref-4)] about misaligned pre-training incentives is correct, or that the latents are splitting in ways that favor minute syntax over broad categories of regulatory logic [[6](#ref-6)]. Some initial evidence for the latter hypothesis can be found in Figure 2, which compares subcategories of regulatory elements belonging to the family of Long Interspersed Nuclear Elements (LINE) and the family of protein binding sequences (PBS). LINEs are a group of highly specific transposons found throughout the human genome, and we see that deeper representations are necessary for the SAEs to extract information related to them, whereas PBSs are a large category of general functionality which retains the negative trend across layers. However, we observed similar trends and average correlation differences of less than 0.02 across different SAE expansion sizes, where we would expect splitting of latents to be more significant. If splitting of latents is responsible for this phenomenon, it cannot be determined from correlation alone.

## Constructing Knowledge Graphs of SAE Activity

Using LLMs to construct knowledge graphs is an increasingly common approach for retrieval augmented generation (RAG), and presents a potential method for addressing the gap in causal determination for SAE feature meaning during the annotation process [[11](#ref-11)]. For this task, LLMs are often paired with fine-tuned named entity recognition (NER) or relationship extraction (RE) heads, which are then used to add nodes to a graph and connect them based on their relationships. Applying similar methods to SAE feature activations on a set of sample sequences allows us to explore the activations using graph methods, thereby enabling further analysis of activation patterns and the use of relationship information when comparing sequences to known references.

To construct a SAE knowledge graph as described above, we can apply NTv2 to a sample dataset to generate latents for a given layer of NTv2, parse the feature activations by connecting the sequence IDs to the strongest activating features, and type each edge according to the tokens which caused the activation. Complete information can be found in the supplementary materials, but the basic graph structure is a *typed heterogenous multigraph* as shown below:

$G = (V, E, \tau, \lambda)$

Where:

- $V$ is the vertex set
- $E$ is the edge set (multiset, allowing parallel edges)
- $\tau: E \to \Sigma$ is the edge type function
- $\lambda: V \cup E \to A$ assigns attributes to vertices and edges

Although the full knowledge graph we constructed is too large to practically visualize, the subgraph in *Figure 3* demonstrates the basic structure of this graph. Using the full collection of RefSeq annotations as found in the UCSC genome browser, we can also record functional metadata for each sequence to perform additional analysis on different sub-communities of SAE activation.

\begin{center}
\begin{tabular}{c}
\includegraphics[width=0.75\textwidth]{figures/image6.png}
\end{tabular}
\end{center}

*Figure 3 \- Neighborhood subgraph of feature 7244 containing all inbound edges from first-degree neighbors. Full knowledge graph was constructed from layer 23 SAE activations (10240 feature neurons) from NTv2-500m-human-ref on a set of putative cisplatin binding sequences. Purple nodes represent sequences, edge labels describe the nucleotide sequence of the token, and red nodes represent the central feature of the subgraph.*

Our sample dataset comes from work by Krishnaraj and colleagues [[12](#ref-12)], who are investigating the binding of RNA sequences to the chemotherapy drug, Cisplatin. By developing a novel click-chemistry transcriptome assay called PlatRNA-seq, they identified a subset of transcripts and RNA fragments that have the capacity to bind to Cisplatin, potentially revealing a new mechanism for chemotherapy resistance. These RNAs are believed to form stable rG4 G-Quadruplex structures in the presence of positive cations such as K\+, with cisplatin competing in this role prior to initial formation [[13](#ref-13)]. However, structural characterization in RNA is still somewhat limited, and large RNA structure databases that contain reliably characterized structures of these transcripts of interest are not yet established. Thus, the precise mechanism is still unknown. Using the mapped loci from Krishnaraj and colleagues to obtain DNA reference sequences (GRCh38), we combined this set of sequences with an equivalent set of randomly selected sequences from the remainder of the human reference genome to create training and testing datasets for CNN classification heads as well as knowledge graphs and intervention studies.

The ongoing line of work by Krishnaraj and colleagues provides an interesting opportunity to address a point of discussion from Simon and Zou [[7](#ref-7)], who suggested that mechanistic interpretability may be able to reveal novel biological patterns learned during pretraining that have not yet been considered by the broader scientific community. This hypothesis suggests that the process of annotating gLM feature activations may reveal novel mechanisms, syntax, or binding motifs which would explain the observations by Krishnaraj and colleagues [[12](#ref-12)]. At present, methods for addressing this question are extremely limited as most approaches to mechanistic interpretability in genomics require a known reference or ground truth. Graph structures are relational, thereby enabling unsupervised analysis of the graph itself to derive patterns of activation which may indicate functional patterns in the underlying biology.

To demonstrate this, we began by constructing SAE knowledge graphs as described above for the cisplatin-binding sequences and non-cisplatin-binding sequences using the final layer of the NTv2 encoder. Based on earlier observations, features at this layer are likely the most fragmented and granular, thus the likelihood of finding novel patterns is highest in this layer. It is also critical to understand why SAE latents fail to capture currently understood regulatory elements consistently in this layer when the concept was more readily extracted in previous layers.

## Graph Topology

By nature, the sequence-to-feature graph structure is relatively sparse, meaning that most nodes are not connected to each other. This can be represented using a density coefficient that reflects the ratio of existing edges to possible edges, where, a perfectly connected graph has a density of one, and a graph with no connections between nodes has a density of zero. The cisplatin-binding graph had a density of *4.01e-3*, and the non-binding graph had a density of *2.16e-2*, indicating that most sequences tended to have a few key active features across their length, and most of these were common enough to be useful descriptors of the cisplatin-binding and non-binding phenotypes. Although they are both sparse, the non-binding graph is an order of magnitude more dense than the cisplatin-binding graph, indicating more diverse SAE feature activity the non-binding phenotype.

Ideally, the graph topology should be relatively sparse with key features being major hubs of feature activation. This enables us to perform focused intervention studies and demonstrates the degree of monosemanticity that is being achieved for the target concept. In *Figure 4*, we demonstrate this by progressively removing the highest PageRank centrality feature nodes from the graph and plotting the rate of fragmentation by counting the number of disconnected subgraphs and measuring the density coefficient. PageRank centrality was originally used for search-engine optimization and measures the importance of nodes by estimating the probability of landing on that node via a combination of random walks and jumps like someone surfing the internet and clicking hyperlinks; these metrics and subsequent graph topological analyses were implemented using the NetworkX library [[14](#ref-14)]. The cisplatin-binding graph fragments much more readily than the non-binding graph, suggesting that NTv2 is able to represent the cisplatin-binding phenotype with fewer SAE latents than the randomly selected non-binding sequences. The precipitous drop in graph density after removing the first 10 nodes also indicates the highly centralized structure of the graph. The cisplatin-binding graph contains 2071 feature nodes, while the non-binding graph contains 3039. This means that graph density reaches a minimum at approximately 4% and 6.5% of feature nodes removed, respectively.

\begin{center}
\begin{tabular}{cc}
\textbf{Cisplatin-Binding} & \textbf{Non-Binding} \\
\includegraphics[width=0.48\textwidth]{figures/posPRfragcurve.png} &
\includegraphics[width=0.48\textwidth]{figures/negPRfragcurve.png} \\
\includegraphics[width=0.48\textwidth]{figures/posPRdenscurve.png} &
\includegraphics[width=0.48\textwidth]{figures/negPRdenscurve.png} \\
\end{tabular}
\end{center}

*Figure 4 \- Fragmentation (top) and density shift (bottom) curves for the cisplatin-binding (left) and non-binding (right) graphs when progressively removing the highest PageRank centrality feature nodes. "Number of Connected Components" refers to the number of disconnected subgraphs. These are segregated subgraphs which are connected to themselves, but not connected to other graphs or communities by edges from any of its members.*

If we continue to remove nodes from the graphs in this manner until visualization becomes practical, we also find interesting layering patterns as shown in *Figure 5*. With approximately 50% of their feature nodes removed, the cisplatin-binding graph exhibits a layered fragmentation pattern where some sequences are still connected to multiple remaining features, but many of them are fully disconnected and many sequences have been dropped from visualization as complete orphan nodes. Approximately 10.3% of nodes remain in the cisplatin-binding graph, some of which are loosely associated with the core component, while the minor components are spread in a ring-like structure around the expanding core. By contrast, the non-binding graph retains 27.6% of nodes, mostly in the core, and exhibits minimal fragmentation. The retention of a major graph core component after removing the central nodes in this case does suggest some sequences still have very diverse feature sets, and the degree of interdependence observed may be proportional to the capacity of NTv2 to capture the targeted phenomenon.

\begin{center}
\begin{tabular}{cc}
\textbf{Cisplatin-Binding} & \textbf{Non-Binding} \\
\includegraphics[width=0.48\textwidth]{figures/poskcorefrag1k.png} &
\includegraphics[width=0.48\textwidth]{figures/negkcorfrag2k.png} \\
\end{tabular}
\end{center}

*Figure 5 \- K-core fragmentation visualization of the cisplatin-binding (left) and non-binding (right) graphs, highlighting structural differences between the two phenotypes. Feature nodes are colored red, and sequence nodes are colored purple.*

The above observations suggest that the cisplatin-binding sequences have a shared set of key features describing their location within the latent space of NTv2 that we can identify, whereas the non-binding class has diverse feature activation sets that make it suitable for a baseline comparison. Since exposure to cisplatin is usually the result of medical intervention, we do not expect a single latent to represent this non-physiological phenomenon, however, the latents appear to still be able to represent this information using a small set that we can target and search for.

## Selecting Features for Intervention

To identify differentially important SAE features between the two sets of sequences, we compared their centrality in their respective graphs as shown in *Figure 6*. As we did above, we selected PageRank for this purpose since it is an efficient algorithm for large-scale directed graph analysis, and we found it to be the most effective at selecting nodes which would fragment the graph. As with most centrality metrics, a higher centrality score means the feature is more relevant and important to the selected graph and thus its experimental condition.

\begin{center}
\begin{tabular}{c}
\includegraphics[width=0.78\textwidth]{figures/image5.png}
\end{tabular}
\end{center}

*Figure 6 \- Log-Fold change of PageRank centrality for shared feature nodes between the cisplatin-binding and non-binding graphs. Negative log-fold change indicates the feature had a higher score in the non-binding graph, while positive log-fold change indicates the feature had a higher score in the cisplatin-binding graph. Feature ids are sorted from highest to lowest log-fold change with negatively associated ids on the left y-axis and positively associated ids on the right y-axis.*

The differential by log-fold change is important for grounding the feature selection in real effects. While several unique features were identified, their centrality scores were extremely low, suggesting they were more likely to be noise than reliable signal. Using log-fold change also normalizes the magnitude of centrality shifts to avoid biasing toward features that had generally high scores.

## Intervention Studies

The current standard for validating feature annotations is to use the SAE decoder to reconstruct embeddings with a particular feature activated or suppressed. The artificially reconstructed embeddings for the original model can then be used instead of a full forward pass to reliably “steer” gLM behavior. In early work by researchers at Anthropic, subjecting human language models such as Claude to this causes them to generate text reflecting the selected feature, such as sycophancy [[6](#ref-6)].

Because the generative head of a gLM produces artificial DNA sequences that are themselves uninterpretable by humans, we trained a convolutional neural network (CNN) classification head on NTv2 embeddings to predict whether a sequence would bind or not bind Cisplatin based on the data provided by Krishnaraj and colleagues [[12](#ref-12)]. The CNN was composed of 2 convolution layers and 2 max pooling layers with a single linear layer out. The convolution kernel was 1 dimensional, sliding across the sequence, while the pooling layers operated on the embedding dimension, gradually reducing the representation down to a logit vector used to compute class probabilities. Using cross-entropy loss, we trained the CNN for up to 100 epochs on shards from an 80:20 train-test data split combining ~44 thousand binding sequences with ~40 thousand non-binding sequences. Early stopping was conditioned on decreasing loss over 10 epochs and triggered after 33 epochs, resulting in a model with a test accuracy of 96.2% and an F1 score of 0.9648 after only seeing 33% of the training set. Using the 20% holdout set of test data, we used the unseen sequences to perform steering experiments using the 8x expansion SAE for layer 23 of the NTv2 encoder to create modified embeddings as input to the CNN classification head.

In prior studies, simple clamping of feature values to a given threshold produced desirable shifts in model behavior [[6](#ref-6), [7](#ref-7), [8](#ref-8)]. However, the therapeutic context of cisplatin exposure is inherently artificial, leading to unexpected probability shifts in the CNN classification head as shown in *Figure 7*. As a result, we found it necessary to experiment with various intervention patterns to elicit the underlying mechanisms. We adopted a grid search intervention protocol involving initial clamping of the selected feature to a minimum value across each token position, followed by element-wise multiplication of an intervention matrix to scale targeted features by a factor $\alpha$ and non-targeted features by $1/\alpha$, as described below. For latent matrix $Z \in \mathbb{R}^{n \times D}$, target feature $f$, minimum activation $\delta$, and scale $\alpha$:

$$Z_{:,f} \leftarrow \max(Z_{:,f},\, \delta), \qquad Z'_{i,j} = Z_{i,j} \cdot \begin{cases} \alpha, & j = f \\ 1/\alpha, & j \neq f \end{cases}$$

A complete formal definition is provided in the supplementary materials. Because of the computational requirements of grid search algorithms, we limited our sequence sample size to the first 1000 sequences from the test holdout set.

### Predicted Probability Shifts for Selected Features Pre/Post SAE Intervention

\begin{center}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ >{\centering\arraybackslash\footnotesize}m{0.18\textwidth} >{\centering\arraybackslash}m{0.38\textwidth} >{\centering\arraybackslash}m{0.38\textwidth} }
\hline
\normalsize\textbf{Condition} & \textbf{Non-Binding Class} \newline \textbf{(Feature 8161)} & \textbf{Cisplatin-Binding Class} \newline \textbf{(Feature 1371)} \\
\hline
Min Act = 10 \newline Amp/Suppress = Ablation &
\includegraphics[width=\linewidth]{figures/image7.png} &
\includegraphics[width=\linewidth]{figures/image8.png} \\
\hline
Min Act = 0.1 \newline Amp/Suppress = 10.0 &
\includegraphics[width=\linewidth]{figures/image9.png} &
\includegraphics[width=\linewidth]{figures/image10.png} \\
\hline
\end{tabular}
\end{center}

*Figure 7 \- Predicted probability of cisplatin binding for 1000 DNA sequences from the test holdout set, with and without SAE intervention on a selected feature. “Min Act” refers to the minimum activation threshold, and raw feature activation values below this threshold are set to this value for the selected feature. “Amp/Suppress” refers to the target feature amplification factor and the inverse value used to suppress all other feature activity. A value of “ablation” indicates that all non-target features were set to zero. Feature 1371 was associated with cisplatin binding and feature 8161 was associated with the non-binding condition.*

As shown in *Figure 7*, the simple clamping of feature activations resulted in consistent shifts in predicted probabilities for all sampled sequences. However, these probabilities often fell between 0.4 and 0.6, well within typical decision-boundary thresholds for binary classification, and not always clearly demonstrating the feature-to-class association identified prior to intervention. One possible explanation is that cisplatin exposure represents a non-physiological perturbation that may be not be directly represented in NTv2 embeddings. Based on this, we altered the intervention to retain some of the original feature information to build a hybrid SAE intervention matrix as described above. Under these new intervention settings, some features produced stronger probability shifts while for other features this same experiment shifted probabilities back toward the baseline.

Confirmation of our hypothesis can be found by examining the Area Under the Receiver Operating Characteristic (AUROC) curves for the experiments above as shown in *Figure 8*. The AUROC metric is rank-invariant, meaning that it indicates whether a binding sample has a higher value than a non-binding sample without respect to how large the difference is. As a result, the AUROC curve shifts indicate whether a feature contributes a monotonic shift to all samples, or completely disrupts all predictive signals. When we use f/1371 to intervene, the discriminative power of the CNN remains, whereas discrimination is significantly impeded by intervention on f/8161 in this experimental setup. This pattern is consistent with latents being highly granular, making the final CNN prediction modulated by co-occurring latent activations. Biologically, it suggests that multiple binding-associated features may be necessary to fully promote cisplatin binding, meaning that the features themselves point to incredibly specific conservation patterns that may be linked to tightly coupled biophysical relationships rather than pathways and functions that may be affected by cell state or environment.

### AUC-ROC Curves for Selected Features Pre/Post SAE Intervention

\begin{center}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ >{\centering\arraybackslash\footnotesize}m{0.18\textwidth} >{\centering\arraybackslash}m{0.38\textwidth} >{\centering\arraybackslash}m{0.38\textwidth} }
\hline
\normalsize\textbf{Condition} & \textbf{Non-Binding (Feature 8161)} & \textbf{Cisplatin-Binding (Feature 1371)} \\
\hline
Min Act = 10.0 \newline Amp/Suppress = Ablation &
\includegraphics[width=\linewidth]{figures/image11.png} &
\includegraphics[width=\linewidth]{figures/image12.png} \\
\hline
Min Act = 0.1 \newline Amp/Suppress = 10.0 &
\includegraphics[width=\linewidth]{figures/image13.png} &
\includegraphics[width=\linewidth]{figures/image14.png} \\
\hline
\end{tabular}
\end{center}

*Figure 8 \- Area Under the Receiver Operating Characteristic (AUROC) curve for 1000 DNA sequences from the test holdout set with and without SAE intervention on the selected feature. Selections mirror Figure 7\. “Min Act” refers to the minimum activation threshold, and raw feature activation values below this threshold are set to this value for the selected feature. “Amp/Suppress” refers to the target feature amplification factor and the inverse value used to suppress all other feature activity. A value of “ablation” indicates that all non-target features were set to zero. Feature 1371 was associated with cisplatin binding and feature 8161 was associated with the non-binding condition.*

## Granularity of SAE Features & Focus on Local Interactions

To gain insight into what these features attend to and how they may be interrelated with the task of predicting cisplatin-binding of RNA transcripts, we adapted dependency mapping techniques from Silva and colleagues [[15](#ref-15)]. They demonstrated that *in-silico* site-directed mutagenesis experiments could identify some functional elements and their dependencies by examining the change in predicted probability for a given base at a given position. We designed a variant of this approach operating on SAE latents, and computed average positionwise deltas of feature activity for all possible substitution variants in selected sequences. The resulting heatmaps in *Figure 9* show how sensitive a feature is at a given base (x-axis) to a substitution at a chosen position (y-axis).

\begin{center}
\begin{tabular}{c}
\includegraphics[width=0.85\textwidth]{figures/image15.png} \\[6pt]
\includegraphics[width=0.85\textwidth]{figures/image16.png}
\end{tabular}
\end{center}

*Figure 9 \- Dependency Maps showing selected feature sensitivity at a given position (x-axis) for substitution at a selected position (y-axis) in a selected cisplatin-binding sequence. Heat color represents the average change in feature activity across all three possible substitutions at the mutated position. Features were selected based on the strongest steering results.*

These dependency maps show that even single features can exhibit sensitivity at a surprisingly local level. An interesting point of comparison in *Figure 9* is the relatively local pattern of sensitivity in f/1371 compared to f/8161, which appears to identify key positions that are sensitive to substitutions throughout the sequence. By contrast, changes in f/1371 activity are mostly limited to the token where the mutation occurred. Most of the features we performed intervention on had patterns similar to f/1371, and f/8161 was the only feature to achieve complete leftward probability shift during the hybrid intervention experiment. Combined with recent literature, this suggests that despite the transformer architecture’s advantages in tracking long-range relationships between tokens, the pre-training objective incentivizes gLMs to focus on local biophysical relationships and context over broader regulatory logic that depends on interactions between distant elements.

If these latents accurately approximate the basis of the gLM latent space, it may be nearly impossible for broad regulatory syntax and regulatory pathways to be directly captured by gLM representations that appear to focus on local biophysics. Representing such information would require a composition of several of these latents, at a minimum, or may not be possible to represent at all if the regulatory mechanics are dependent on extrinsic factors such as metabolic feedback loops. If the pre-training objective is the source of the bias, it also remains unclear whether expansion of parameter count or context window size would overcome a bias toward local information.

## Visualizing Feature Sensitivity to Putative Physical Interactions

Although Krishnaraj and colleagues [[12](#ref-12)] were able to determine that cisplatin binds to putative rG4 quadruplexes and can inhibit cation stabilization by potassium, the exact mechanism is still undetermined. To discuss what that mechanism might be and what biophysical properties these features were sensitive to, we generated predicted structures with Boltz-2 [[16](#ref-16)] for a series of experimentally validated rG4 quadruplexes in the Quadratlas database [[17](#ref-17)] that were found wholly within the mapped regions identified by Krishnaraj and colleagues [[12](#ref-12)]. These structures were then linked to dependency maps of selected features by projecting the dependency maps into a 1-dimensional representation using a spectral graph laplacian to produce a single per-base sensitivity score that could be visualized on the predicted structures generated by Boltz-2.

As we examine these structures, it is important to note that RNA structures tend to be highly dynamic, and some conformations may require interaction with various proteins to become energetically favorable [[18](#ref-18)]. These predicted structures are valid for hypothesis generation, but are not themselves conclusive. While Boltz-2 has predicted mostly hairpin structures for the selected sequences, many conformations are possible, including varying intermediates, in the absence of experimentally characterized structures.

\begin{center}
\begin{tabular}{c}
\includegraphics[width=0.78\textwidth]{figures/image17.png}
\end{tabular}
\end{center}

*Figure 10 \- Electron space filling visualization of predicted RNA secondary structure for chr1:31938031-31938149(-) by Boltz-2 in the presence of 2 cisplatin molecules. The red circle highlights the presence of cisplatin (white) within the major groove of the RNA hairpin helix (purple). Dotted blue lines indicate hydrogen bonding.*

In the predicted Boltz-2 structures, it is common to observe cisplatin within the major groove of the RNA hairpin double helix as shown in *Figure 10*. This makes intuitive physical sense given it is the most obvious binding site in the predicted structure, and coincidently aligns with some feature sensitivity scores and known biochemistry [[19](#ref-19), [18](#ref-18), [20](#ref-20)]. For example, the canonical therapeutic mechanism for cisplatin involves forming dimers between adjacent purine bases in DNA. Cisplatin is typically aquated when it passes into the cytosol due to the relative concentration of chloride, which allows it to form dimers by nucleophilic attack, usually by the N7 position on guanine [[19](#ref-19)]. In *Figure 11,* we see that f/3378 seems to be sensitive to a series of guanine bases which are in exposed positions that would be conducive to this mechanism of cisplatin dimerization. Alternatively, zinc finger motifs are commonly found in proteins which bind DNA and RNA for various regulatory functions [[18](#ref-18), [20](#ref-20)]. These motifs commonly include GC boxes and purine-rich sequences involved in rG4 quadruplex formation, where arginine, histidine, and threonine residues may interact to stabilize and fold the RNA into the quadruplex structure.

\begin{center}
\begin{tabular}{c}
\includegraphics[width=0.78\textwidth]{figures/image18.png}
\end{tabular}
\end{center}

*Figure 11 \- Stick and ribbon visualization of predicted RNA secondary structure for chr1:31938031-31938149(-) by Boltz-2 in the presence of 2 cisplatin molecules. The red circle shows the cisplatin molecules (purple) within the major groove of the RNA hairpin helix. The helix is colored according to feature sensitivity score (f/3378) derived from dependency mapping via NTv2-500m-human-ref and SAE output from the final encoder layer. Red indicates high sensitivity and blue represents low sensitivity. The red arrow points to exposed consecutive guanine bases in the major groove which have notably high f/3378 sensitivity score.*

While this evidence should not be construed as empirical evidence for a biochemical mechanism of cisplatin-RNA binding, the existence of plausible biochemical mechanisms which explain f/3378 sensitivity is consistent with the hypothesis that some out-of-domain gLM behavior may originate from pretraining-induced representations, although targeted experimental validation is still required [[21](#ref-21)].

## Discussion

Although NTv2 was unable to track many of the current annotated regulatory elements from the reference genome, it appears that it was able to track minute biophysical syntax that enabled downstream predictions of non-physiological phenomena. This granular capture of syntax can be understood functionally through the relationship between the pre-training objective, information entropy, and evolutionary constraint. By minimizing cross-entropy loss during pre-training, gLMs are inherently incentivized to compress and predict low-entropy sequence motifs. In the genome, regions of low information entropy correspond to high evolutionary conservation, which often strictly preserves local biophysical interactions and stable secondary structures (e.g., RNA G-quadruplexes).

Consequently, the "minute syntax" encoded by individual SAE features represents these low-entropy, biophysically constrained structural building blocks. To predict a complex macroscopic event like cisplatin binding, downstream readout mechanisms must aggregate multiple of these syntactical features rather than relying on a single abstract "binding" latent. This explains why gLMs track highly granular conservation patterns tightly aligned with local biophysical properties rather than broad, abstract regulatory logic, resulting in apparent out-of-domain performance while struggling to natively categorize large-scale regulatory sequences. This is consistent with findings from Boshar et al. [[21](#ref-21)] and Tang et al. [[4](#ref-4)], who independently discovered these particular strengths and weaknesses of gLMs through performance evaluation.

Our findings contribute a mechanistic understanding of why gLMs perform as they do in each domain, however, they also demonstrate significant limitations for both gLMs and mechanistic interpretation of these models. In terms of the hidden biological patterns among gLM representations, our results are consistent with other studies that have indicated the presence of this information in pre-trained representations [[9](#ref-9), [15](#ref-15)]. However, current methods are fundamentally limited in determining what the function of these novel elements are, which limits our ability to design studies to validate or discover the purpose of these patterns in the genome. There is also a limitation of scale, as current interpretability methods tend to either provide low-resolution insights at scale or high-resolution insights on the scale of individual genes due to the use of algorithms with exponential complexity. Unpacking these hidden biological patterns therefore requires algorithmic innovations that improve on the limitations of current methods.

For mechanistic interpretability in particular, the extraction of SAE latents has already been demonstrated at scale [[1](#ref-1)], but the annotation and exploration of what those latents mean still has major obstacles to overcome. We demonstrated several innovations that may contribute to addressing those challenges, but they primarily address epistemic limitations and still scale with exponential complexity, meaning they will quickly become impractical for large scale annotation efforts. Further development of interpretability efforts must address the computational complexity challenge to enable widespread adoption.

## Data and Code Availability

All source code is publicly available at: https://github.com/ek775/Hidden-State-Genomics.

## Acknowledgements and Conflicts of Interest

**Funding.**

<!-- markdownlint-disable MD033 -->
## References

1. <a id="ref-1"></a> Brixi G, Durrant MG, Ku J, et al. Genome modeling and design across all domains of life with Evo 2. bioRxiv. 2025.02.18.638918. doi: [10.1101/2025.02.18.638918](https://doi.org/10.1101/2025.02.18.638918).
2. <a id="ref-2"></a> Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. Nat Methods. 2025;22:287-297. doi: [10.1038/s41592-024-02523-z](https://doi.org/10.1038/s41592-024-02523-z).
3. <a id="ref-3"></a> Zheng Y, Koh HY, Ju J, et al. Large language models for scientific discovery in molecular property prediction. Nat Mach Intell. 2025;7:437-447. doi: [10.1038/s42256-025-00994-z](https://doi.org/10.1038/s42256-025-00994-z).
4. <a id="ref-4"></a> Tang Z, Somia N, Yu Y, et al. Evaluating the representational power of pre-trained DNA language models for regulatory genomics. Genome Biol. 2025;26:203. doi: [10.1186/s13059-025-03674-8](https://doi.org/10.1186/s13059-025-03674-8).
5. <a id="ref-5"></a> Favor A, Quijano R, Chernova E, et al. De novo design of RNA and nucleoprotein complexes. bioRxiv. 2025.10.01.679929. doi: [10.1101/2025.10.01.679929](https://doi.org/10.1101/2025.10.01.679929).
6. <a id="ref-6"></a> Templeton A, Conerly T, Marcus J, et al. Scaling monosemanticity. Transformer Circuits; 2024. Available at: [Transformer Circuits article](https://transformer-circuits.pub/2024/scaling-monosemanticity/).
7. <a id="ref-7"></a> Simon E, Zou J. InterPLM: discovering interpretable features in protein language models via sparse autoencoders. Nat Methods. 2025;22:2107-2117. doi: [10.1038/s41592-025-02836-7](https://doi.org/10.1038/s41592-025-02836-7).
8. <a id="ref-8"></a> Adams E, Bai L, Lee M, Yu Y, AlQuraishi M. From mechanistic interpretability to mechanistic biology: training, evaluating, and interpreting sparse autoencoders on protein language models. bioRxiv. 2025.02.06.636901. doi: [10.1101/2025.02.06.636901](https://doi.org/10.1101/2025.02.06.636901).
9. <a id="ref-9"></a> Seitz EE, McCandlish DM, Kinney JB, Koo PK. Uncovering the mechanistic landscape of regulatory DNA with deep learning. bioRxiv. 2025.10.07.681052. doi: [10.1101/2025.10.07.681052](https://doi.org/10.1101/2025.10.07.681052).
10. <a id="ref-10"></a> Zhou J, Rizzo K, Christensen T, Tang Z, Koo PK. Uncertainty-aware genomic deep learning with knowledge distillation. bioRxiv. 2024.11.13.623485. doi: [10.1101/2024.11.13.623485](https://doi.org/10.1101/2024.11.13.623485).
11. <a id="ref-11"></a> Raieli S, Iuculano G. Building AI Agents with LLMs, RAG, and Knowledge Graphs. Packt Publishing; 2025.
12. <a id="ref-12"></a> Krishnaraj A, Wei X, Thakral R, Guo W, Subramaniyan SB, Zhang X, Alley KR, Feng A, Hoy A, Kung D, Tiwari PB, Uren A, Maillard R, DeRose VJ, Nair SJ. Transcriptome-wide mapping reveals an RNA-dependent mechanism of platinum cancer drugs. bioRxiv [Preprint]. 2025 Dec 23:2025.12.20.694502. doi: [10.64898/2025.12.20.694502](https://doi.org/10.64898/2025.12.20.694502).
13. <a id="ref-13"></a> Lyu K, Chow EYC, Mou X, Chan TF, Kwok CK. RNA G-quadruplexes (rG4s): genomics and biological functions. Nucleic Acids Res. 2021;49(10):5426-5450. doi: [10.1093/nar/gkab187](https://doi.org/10.1093/nar/gkab187).
14. <a id="ref-14"></a> Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008
15. <a id="ref-15"></a> da Silva TP, Karollus A, Hingerl J, et al. Nucleotide dependency analysis of genomic language models detects functional elements. Nat Genet. 2025;57:2589-2602. doi: [10.1038/s41588-025-02347-3](https://doi.org/10.1038/s41588-025-02347-3).
16. <a id="ref-16"></a> Passaro S, Corso G, Wohlwend J, et al. Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction. bioRxiv [Preprint]. 2025 Jun 18:2025.06.14.659707. doi: [10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707).
17. <a id="ref-17"></a> Bourdon S, Herviou P, Dumas L, Destefanis E, Zen A, Cammas A, Millevoi S, Dassi E. QUADRatlas: the RNA G-quadruplex and RG4-binding proteins database. Nucleic Acids Res. 2023 Jan 6;51(D1):D240-D247. doi: [10.1093/nar/gkac782](https://doi.org/10.1093/nar/gkac782).
18. <a id="ref-18"></a> Meier-Stephenson V. G4-quadruplex-binding proteins: review and insights into selectivity. Biophys Rev. 2022;14(3):635-654. doi: [10.1007/s12551-022-00952-8](https://doi.org/10.1007/s12551-022-00952-8).
19. <a id="ref-19"></a> Rocha CRR, Silva MM, Quinet A, Cabral-Neto JB, Menck CFM. DNA repair pathways and cisplatin resistance: an intimate relationship. Clinics (Sao Paulo). 2018;73(suppl 1):e478s. doi: [10.6061/clinics/2018/e478s](https://doi.org/10.6061/clinics/2018/e478s).
20. <a id="ref-20"></a> Shu H, Zhang R, Xiao K, Yang J, Sun X. G-Quadruplex-Binding Proteins: Promising Targets for Drug Design. Biomolecules. 2022;12(5):648. doi: [10.3390/biom12050648](https://doi.org/10.3390/biom12050648).
21. <a id="ref-21"></a> Boshar S, Trop E, de Almeida BP, Copoiu L, Pierrot T. Are genomic language models all you need? Exploring genomic language models on protein downstream tasks. Bioinformatics. 2024;40(9):btae529. doi: [10.1093/bioinformatics/btae529](https://doi.org/10.1093/bioinformatics/btae529).

## Supplementary Materials

### Supplementary Methods S1: Notation

Let a sequence produce token-level hidden states $h_i \in \mathbb{R}^d$ for positions $i=1,\dots,n$. Let $m$ denote the SAE expansion factor and $D = m d$ denote the latent dictionary size. Let $z_i \in \mathbb{R}^D$ be the sparse latent activation at position $i$.

### Supplementary Methods S2: Sparse Autoencoder Training Objective

We train a one-hidden-layer sparse autoencoder over transformer hidden states with reconstruction and sparsity regularization. For each token embedding $h_i$, the model produces $z_i$ and reconstruction $\hat{h}_i$. The optimization target is

$$
\mathcal{L}_t = \frac{1}{n}\sum_{i=1}^{n} \lVert h_i - \hat{h}_i \rVert_2^2 + \lambda_t \frac{1}{n}\sum_{i=1}^{n} \lVert z_i \rVert_1.
$$

The sparsity weight is linearly annealed during early optimization:

$$
\lambda_t = \lambda_{\max} \min\left(1, \frac{t}{T_{\text{anneal}}}\right),
$$

where $t$ is the training step and $T_{\text{anneal}}$ is the annealing horizon. Hidden-state sequences are shuffled, partitioned with a 20% validation holdout, and processed in shards to control memory footprint. Model selection uses validation tracking with early stopping.

### Supplementary Methods S3: Knowledge Graph Construction

We construct a typed heterogeneous directed multigraph

$$G = (V, E, \tau, \lambda),$$

with disjoint vertex sets

$$V = V_{\text{tok}} \cup V_{\text{feat}},$$

where $V_{\text{tok}}$ are nucleotide token strings and $V_{\text{feat}} = \{0,1,\dots,D-1\}$ are feature indices. For each sequence position $i$, define the dominant feature

$$f_i^* = \arg\max_j z_{i,j}.$$

A directed edge $(t_i, f_i^*)$ is added from token $t_i$ to feature $f_i^*$. The edge type mapping $\tau$ assigns token-mediated edge identity, and $\lambda$ stores sequence and genomic metadata (sequence ID, coordinate, strand, and annotation attributes). This yields a multigraph with parallel edges across repeated token-feature events in distinct genomic contexts.

### Supplementary Methods S4: Feature Intervention Operator

For latent matrix $Z \in \mathbb{R}^{n \times D}$, target feature index $f$, minimum activation $\delta \ge 0$, and intervention scale $\alpha \ge 0$:

1. Threshold clamp on target feature:

$$Z_{:,f} \leftarrow \max(Z_{:,f}, \delta).$$

2. Construct multiplicative intervention mask $V$:

$$V_{i,j} =
\begin{cases}
\alpha, & j=f,\\
1/\alpha, & j \ne f,\ \alpha \ne 0,\\
0, & j \ne f,\ \alpha = 0.
\end{cases}$$

3. Apply intervention:

$$Z' = Z \odot V.$$

When $\alpha=0$, non-target features are ablated. Intervened representations are then passed to downstream prediction for comparative evaluation against baseline predictions.

### Supplementary Methods S5: Data Splits and Evaluation Protocol

Cisplatin-binding and non-binding sequence sets are deduplicated prior to partitioning. Data are split into train-test with ratio 80:20, and the training partition is further split into train-validation with ratio 80:20, yielding an effective 64:16:20 train-validation-test composition. Intervention analyses are performed on a reduced holdout subset under compute constraints, as noted in the main text.

### Supplementary Methods S6: Reproducibility Inputs

Core runtime inputs include model reference path, sequence repository path, reference cache files, and cloud storage location for large artifacts. Source data include BED or FASTA sequence inputs, genomic annotations, trained SAE checkpoints, and downstream classifier weights.

### Supplementary Methods S7: CNN Classification Head Architecture and Training

To probe whether SAE representations are predictive in a downstream supervised task, a convolutional neural network (CNN) classification head is trained to discriminate cisplatin-binding RNA sequences from non-binding controls. The classifier accepts as input either the SAE latent matrix $Z \in \mathbb{R}^{L \times D}$ (features mode) or the transformer hidden-state matrix $H \in \mathbb{R}^{L \times d}$ (embeddings mode), where each sequence is zero-padded to a fixed length of $L = 1000$ tokens.

**Architecture.** An adaptive kernel width $k = \max\!\left(\lfloor D^{1/5} \rfloor,\, 2\right)$ is derived from the input feature dimension $D$. The network applies the following operations sequentially:

1. Convolutional block 1: $D \to \lfloor D/8 \rfloor$ channels, kernel width $k$, half-padding $\lfloor k/2 \rfloor$, ReLU activation.
2. Convolutional block 2: $\lfloor D/8 \rfloor \to 64$ channels, same kernel and padding, ReLU activation.
3. Max-pooling: kernel width $k$, stride 6, ReLU activation.
4. Dropout with rate 0.5.
5. Adaptive max-pooling to temporal dimension 1, ReLU activation.
6. Fully connected layer: $64 \to 2$ logits for binary classification.

**Training objective.** Parameters are estimated by minimizing categorical cross-entropy:

$$\mathcal{L}_{\text{cls}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c \in \{0,1\}} y_{i,c} \log \hat{p}_{i,c},$$

where $y_{i,c} \in \{0,1\}$ is the one-hot class indicator and $\hat{p}_{i,c}$ is the softmax-normalized predicted class probability for sample $i$ and class $c$.

**Optimization.** The Adam optimizer is used with learning rate $\eta = 10^{-3}$. Training data are uniformly fragmented across epochs, allowing the full dataset to be traversed over the training run without loading all upstream representations into memory simultaneously. Early stopping against validation loss terminates training when no improvement is observed within a fixed patience window.
