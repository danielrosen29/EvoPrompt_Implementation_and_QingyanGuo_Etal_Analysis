# CONNECTING LARGE LANGUAGE MODELS WITH EVOLUTIONARY ALGORITHMS YIELDS POWERFUL PROMPT OPTIMIZERS
**Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang**
****************************************************************************************************
**Tsinghua University, Microsoft Research, Northeastern University**
****************************************************************************************************
## Introduction: 
In recent times it has become clear that Large Language Models (LLMs) exhibit remarkable capabilities for performing various Natural Language Processing (NLP) tasks. However, achieving peak performance is often contingent on the quality of the prompt fed to the model. Traditionally, crafting these prompts has required substantial human effort, often relying on engineers with specialized knowledge and deep understanding of the models in question. In this paper, Qingyan Guo et al. propose a data-driven, automated technique for prompt development called EvoPrompt. EvoPrompt leverages Evolutionary Algorithms (EA) to iteratively refine and optimize prompts for specific NLP tasks without the need for human intervention. Their findings indicate that this automated approach can yield improvements in performance by up to 25% in some cases, marking a significant advancement in the field of prompt engineering.
 
## Discrete Prompts:
A discrete prompt is the use of explicit instructions to the input text fed to LLMs. These discrete prompts guide LLMs perform specific tasks with negligible increases to computational cost meanwhile eliminating
the need to access the parameters and gradients of the models (Liu et al., 2023). These are the types of prompts which EvoPrompt aims to improve. 

## Other Discrete Prompt Generation Methods:
- **Reinforcement learning (RL):** Methods like RLPrompt (Deng et al. 2022) and TEMPERA (Zhang et al. 2023) train an RL agent to generate prompts by interacting with the LLM. They require access to model internals like output probabilities.
- **Enumeration:** Methods like PromptSource (Bach et al. 2022) and APE (Zhou et al. 2022) generate a large set of prompt candidates via techniques like sampling and then select the best. Higher variance algorithm. 
- **Revision:** Approaches like GRIPS (Prasad et al. 2022) and AutoPrompt (Shin et al. 2020) iterate on an initial prompt by fixing incorrect predictions. Higher bias algorithm. 
- **Editing:** Methods like Instruction Tuning (Prasad et al. 2022) edit the words in a prompt to improve it. These also focus on local improvements.

## Evolutionary Algorithms:
Evolutionary Algorithms (EAs) are a family of optimization algorithms inspired by the process of natural evolution. These algorithms are heuristics used to find approximate solutions to optimization and search problems. There are many different types and subtypes of EAs including but not limited to:

- **Genetic Algorithms (GAs):** Perhaps the most well-known type, focused on string-based chromosomes and typically employs crossover, mutation, and selection.
- **Differential Evolution (DE):** A population-based optimization algorithm that optimizes a problem by iteratively improving candidate solutions with regard to a given measure of quality.
- **Estimation of Distribution Algorithms (EDAs):** Rather than using crossover and mutation, these algorithms build probabilistic models of promising solutions and sample from these models to generate new candidates.
- **Neuroevolution:** A method for iteratively building a neural network through genetic propogation. 

## EvoPrompt:
![image](https://github.com/danielrosen29/QingyanGuo_Etal_Analysis/assets/75226826/162d89db-a9ee-4772-81cb-c2ffa9916c87)

This algorithm is the general outline for implementing the evolutionary process of discrete prompts, but can be implemented different ways depending on which evolutionary operators (EO) are used. In the paper, Qingyan Guo et al. propose two implementations. The first of these is using a Genetic Algorithm as the EO, the second uses Differential Evolution. The notation for these in the paper is EvoPrompt(EA) and EvoPrompt(DE), respectively. 

### EvoPrompt(GA):
<img src="https://github.com/danielrosen29/EvoPrompt_Implementation_and_QingyanGuo_Etal_Analysis/assets/75226826/236d8997-5e35-42a6-9c6f-a5b2dd16ea8f" alt="EvoPrompt(GA) Pseudocode" align='right' width=50%>
Genetic algorithms attempt to find the best solution by mimicking the process of natural evolution—inheritance, mutation, selection, and crossover are the primary operators.

**Basic Steps for Genetic Algorithms:**

- **Initialization:** Create an initial population of candidate solutions (chromosomes).
- **Evaluation:** Evaluate the fitness of each chromosome in the population.
- **Selection:** Select parents based on their fitness.
- **Crossover:** Create new chromosomes (offspring) by combining the genetic information of the parents.
- **Mutation:** Apply random changes to the offspring.
- **Replacement:** Replace the old population with the new population of offspring.
- **Termination:** Repeat steps 2-6 until a termination condition is met (e.g., max number of generations, a solution with acceptable fitness is found).

**Crossover:**

This step combines genetic material from two parent chromosomes to produce one or more offspring. The idea is to inherit good traits from both parents, increasing the likelihood that the offspring will be more "fit." There are several types of crossover techniques:

**Mutation:**

This step serves to maintain genetic diversity and helps in exploring the search space more broadly. After crossover, the offspring undergo mutation with a small probability. Mutation changes one or more gene values in a chromosome.
  <img/>
<img src="https://github.com/danielrosen29/QingyanGuo_Etal_Analysis/assets/75226826/4f021b7f-649e-474f-a57b-2315a72ede77"/> 
<div align="center">
  <em>Demonstration of Genetic Algorithm Implemented for Evolving Discrete LLM Prompt. (Qingyan Guo et al. 2023)</em>
</div>



### EvoPrompt(DE):
Differential Evolution (DE) is a population-based optimization algorithm commonly used for solving optimization problems, including those that are nonlinear and non-differentiable. It is particularly well-suited for optimization in a continuous domain and has applications in various fields like engineering, data science, and finance.

<img src= "https://github.com/danielrosen29/QingyanGuo_Etal_Analysis/assets/75226826/658c1844-ed70-4cb7-86bf-4a3cd9e0c63a" align='right' width=50%>

**Basic Steps for Differential Evolution:**

- **Initialization:** A population of potential solutions is randomly initialized within the problem's search space. Each individual in the population is usually represented as a vector of real numbers.
- **Mutation:** For each target vector in the current population, a mutant vector is generated by combining the vectors of three other individuals selected randomly from the population. The combination is generally a weighted difference between two of these vectors, which is then added to the third vector.
  
		Mutant Vector = Target + F × (Vector1 − Vector2)
		F = mutation factor: usually between 0 and 2.
	
- **Crossover:** The mutant vector then undergoes crossover with the target vector to produce a trial vector. Elements of the mutant vector have a chance to be mixed with elements of the target vector, depending on a crossover probability parameter CR.
- **Selection:** The trial vector is then evaluated using the objective function. If it offers a better solution than the target vector, it replaces the target vector in the next generation. Otherwise, the target vector remains in the population.
- **Termination:** Steps 2-4 are repeated until a termination criterion is met, such as a maximum number of generations or an acceptable level of convergence.
<img/>
<img src="https://github.com/danielrosen29/QingyanGuo_Etal_Analysis/assets/75226826/030189ee-b047-4b4b-929d-e9da034a98d4"/>
<div align="center">
  <em>Demonstration of Differential Evolution Implemented for Evolving Discrete LLM Prompt. (Qingyan Guo et al. 2023)</em>
</div>

## Experimental Results:
The study uses GPT-3.5 for performing evolutionary operations to optimize prompts with EvoPrompt for both open-source Alpaca-7b and closed-source GPT-3.5. Their approach was compared against three methods:

- Manual Instructions (MI) which are predefined in Zhang et al. (2023b) and  Sanh et al. (2021)
- PromptSource and Natural Instructions (NI) that find related human-written prompts from multiple datasets. 
- APE which uses iterative Monte Carlo Search on initial prompts.

**Language understanding:**

Seven datasets were used, focusing on sentiment classification, topic classification, and subjectivity classification. EvoPrompt showed improved results compared to previous methods. Notably, EvoPrompt (DE) showed a significant advantage of 9.7% accuracy over EvoPrompt (GA) for subjectivity classification (Subj).

![image](https://github.com/danielrosen29/QingyanGuo_Etal_Analysis/assets/75226826/352588a6-3c42-4ad0-a556-392bd7113674)
<div align="center">
  <em>(Qingyan Guo et al. 2023)</em>
</div>

**Language Generation (Summarization):**

EvoPrompt was evaluated for text summarization on the SAMSum dataset and text simplification on the ASSET dataset. The results indicate that EvoPrompt outperforms both manual and APE-generated prompts on Alpaca-7b and GPT-3.5. Particularly, EvoPrompt (DE) performed better on the summarization task, while both GA and DE versions performed similarly on the simplification task.

![image](https://github.com/danielrosen29/QingyanGuo_Etal_Analysis/assets/75226826/6164f350-a555-4e48-a0c6-c6c0a7af31c8)
<div align="center">
  <em>(Qingyan Guo et al. 2023)</em>
</div>

**Results by Iteration:**

Because this is an evolutionary algorithm based method, one would expect the quality of responses to improve as the number of iteration increases. The following graph provided by the authors shows that this was in fact the case. 

![image](https://github.com/danielrosen29/QingyanGuo_Etal_Analysis/assets/75226826/af10d540-6ce8-4c07-89f7-dc3075320e93)
<div align="center">
  <em>(Qingyan Guo et al. 2023)</em>
</div>

## Summary of Results:

**Dataset Performance:**

- EvoPrompt(GA) performs better for sentiment analysis. (SST-5 dataset)
- EvoPrompt(DE) is superior at question answering (SUBJ dataset)

**Scenario-Based Recommendations:**
- Use GA when the initial manual prompts are of high quality.
- Use DE when the initial prompts are poor, as it's better at escaping local optima.

## Discussion Questions:
*As we can see from the demonstration, there was no need to interact with any model parameters or gradients. Can anyone think of any benefits this may provide?*

![Alt Text](https://media.giphy.com/media/26FfieBFKHaHCivte/giphy.gif)

- Model Improvement without additional training: EvoPrompt is a more data-driven approach to using the tool which was already  in production that improves results.
- Black-box Utilization: This feature enables EvoPrompt to work with LLMs as black-box entities, meaning it can be applied to a variety of pre-trained models without needing specific adaptations.
- Speed and Efficiency: Not having to backpropagate or update the neural network parameters might make the algorithm faster and more computationally efficient in certain scenarios.

*What are the implications for the need to have a scoring metric mean for this the usage of this algorithm?*

- Because you need a metric to decide which prompts to select for the next generation, this algorithm is only applicable for discrete prompts whose successfulness is measurable. 

## Critical Analysis:
### Implications:
- **Model Improvement Without Additional Training:**
EvoPrompt provides a data-driven framework to improving model performance without additional training. This is crucial for scenarios where re-training a model is either computationally expensive or practically infeasible. 

- **Black-Box Utilization:**
Because EvoPrompt does not need access to model parameters, EvoPrompt is capable of improving the usage of model-as-service products or black-box LLMs. Further, it can be applied to a variety of pre-trained models without needing specific adaptations.

- **Computational Efficiency:**
The lack of a need for gradient calculations and parameter updates significantly speeds up the optimization process. This is especially important when the optimization has to be performed multiple times or in real-time scenarios.

- **AI Implications** There is a framework implemented for the model to improve itself, is that true AI?

### Limitations:
- **Limited Cases where it is applicable**
	- Because you need a ground truth, even if the model comes up with a better answer you may mark it worse than the original despite it being better.
- **Can anyone think of any others?**

## References:
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1–35, 2023.

Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric P Xing, and Zhiting Hu. Rlprompt: Optimizing discrete text prompts with reinforcement learning. arXiv preprint arXiv:2205.12548, 2022.

Tianjun Zhang, Xuezhi Wang, Denny Zhou, Dale Schuurmans, and Joseph E Gonzalez. Tempera: Test-time prompt editing via reinforcement learning. In The Eleventh International Conference on Learning Representations, 2023.

Stephen Bach, Victor Sanh, Zheng Xin Yong, Albert Webson, Colin Raffel, Nihal V Nayak, Abheesht Sharma, Taewoon Kim, M Saiful Bari, Thibault Févry, et al. Promptsource: An integrated development environment and repository for natural language prompts. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 93–104, 2022.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910, 2022.

Archiki Prasad, Peter Hase, Xiang Zhou, and Mohit Bansal. Grips: Gradient-free, edit-based instruction search for prompting large language models. arXiv preprint arXiv:2203.07281, 2022.

Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4222–4235, 2020.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. In ACL, 2022b.

John H. Holland. Adaptation in Natural and Artificial Systems. University of Michigan Press, Ann Arbor, 1975. ISBN 0262581116.

Rainer Storn and Kenneth Price. Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. Journal of global optimization, 11:341–359, 1997.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. In ACL, 2022c.

Chengwei Qin, Aston Zhang, Zhuosheng Zhang, Jiaao Chen, Michihiro Yasunaga, and Diyi Yang. Is chatgpt a general-purpose natural language processing task solver? arXiv preprint arXiv:2302.06476, 2023.

Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207, 2021.

Yue Zhang, Leyang Cui, Deng Cai, Xinting Huang, Tao Fang, and Wei Bi. Multi-task instruction tuning of llama for specific scenarios: A preliminary study on writing assistance. arXiv preprint arXiv:2305.13225, 2023c.

Bei Li, Rui Wang, Junliang Guo, Kaitao Song, Xu Tan, Hany Hassan, Arul Menezes, Tong Xiao, Jiang Bian, and JingBo Zhu. Deliberate then generate: Enhanced prompting framework for text generation. arXiv preprint arXiv:2305.19835, 2023.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine
Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables
zero-shot task generalization. arXiv preprint arXiv:2110.08207, 2021.
