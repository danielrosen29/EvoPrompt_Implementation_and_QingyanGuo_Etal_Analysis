# CONNECTING LARGE LANGUAGE MODELS WITH EVOLUTIONARY ALGORITHMS YIELDS POWERFUL PROMPT OPTIMIZERS
**Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang**
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

### EvoPrompt(EA):
<img src="https://github.com/danielrosen29/QingyanGuo_Etal_Analysis/assets/75226826/747a5372-a1b0-4865-bc69-48a186d6df0d" alt="EvoPrompt(GA) Pseudocode" align='right' width=50%>
Genetic algorithms attempt to find the best solution by mimicking the process of natural evolution—inheritance, mutation, selection, and crossover are the primary operators.

**Basic Steps:**

- **Initialization:** Create an initial population of candidate solutions (chromosomes).
- **Evaluation:** Evaluate the fitness of each chromosome in the population.
- **Selection:** Select parents based on their fitness.
- **Crossover:** Create new chromosomes (offspring) by combining the genetic information of the parents.
- **Mutation:** Apply random changes to the offspring.
- **Replacement:** Replace the old population with the new population of offspring.
- **Termination:** Repeat steps 2-6 until a termination condition is met (e.g., max number of generations, a solution with acceptable fitness is found).
- **Crossover** (Recombination)

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

**Basic Steps:**

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
The study uses GPT-3.5 for performing evolutionary operations to optimize prompts with EVOPROMPT for both open-source Alpaca-7b and closed-source GPT-3.5. Their approach was compared against three methods:

- Manual Instructions (MI)
- PromptSource and Natural Instructions (NI) that use human-written prompts
- APE which uses iterative Monte Carlo Search on initial prompts.

**Language understanding:**

Seven datasets were used, focusing on sentiment classification, topic classification, and subjectivity classification. EVOPROMPT showed improved results compared to previous methods. Notably, EVOPROMPT (DE) showed a significant advantage of 9.7% accuracy over EVOPROMPT (GA) for subjectivity classification (Subj).

![image](https://github.com/danielrosen29/QingyanGuo_Etal_Analysis/assets/75226826/352588a6-3c42-4ad0-a556-392bd7113674)
<div align="center">
  <em>(Qingyan Guo et al. 2023)</em>
</div>

**Language Generation (Summarization):**

EVOPROMPT was evaluated for text summarization on the SAMSum dataset and text simplification on the ASSET dataset. The results indicate that EVOPROMPT outperforms both manual and APE-generated prompts on Alpaca-7b and GPT-3.5. Particularly, EVOPROMPT (DE) performed better on the summarization task, while both GA and DE versions performed similarly on the simplification task.

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

**Summary of Results:**

**Performance on Datasets:**

- On the SST-5 dataset, EVOPROMPT using GA outperforms its DE variant.
 On the Subj dataset, the DE variant of EVOPROMPT performs better.

**Selection Strategies:**

- GA's selection strategy prioritizes prompts with higher scores for generating new prompts, making it more likely to explore around the current best solutions.
- DE, selects each prompt in the population as a basic prompt and chooses two additional prompts at random.

**Scenario-Based Recommendations:**

When the initial manual prompts are of high quality, as in the SST-5 dataset, GA tends to perform better. The GA variant benefits from high-quality starting points and optimizes further from there.
DE is recommended when the existing prompts are of poor quality, as in the Subj dataset. DE has a higher likelihood of escaping local optima, which led to a remarkable 25% improvement in performance over manual prompts in the case of the Subj dataset.

**Local Optima:**

- GA is prone to getting trapped in local optima when starting from poor-quality prompts.
- DE is better at escaping local optima, thanks to its selection strategy and well-designed evolutionary operators.

**In summary, we suggest choosing EVOPROMPT (GA) when several high-quality prompts already exist, and choosing EVOPROMPT (DE) otherwise.**
  
# Demonstration of EvoPrompt(GA)!

## Discussion Questions:
*As we can see from the demonstration, there was no need to interact with any model parameters or gradients. Can anyone think of any benefits this may provide?*

![Alt Text](https://media.giphy.com/media/26FfieBFKHaHCivte/giphy.gif)

- Model Improvement without additional training: EvoPrompt is a more data-driven approach to using the tool which was developed which improves results.
- Black-box Utilization: This feature enables EVOPROMPT to work with LLMs as black-box entities, meaning it can be applied to a variety of pre-trained models without needing specific adaptations.
- Speed and Efficiency: Not having to backpropagate or update the neural network parameters might make the algorithm faster and more computationally efficient in certain scenarios.

*What are the implications for the need to have a scoring metric mean for this the usage of this algorithm?*

- Because you need a metric to decide which prompts to select for the next generation, this algorithm is only applicable for discrete prompts whose successfulness is measurable. 

## Critical Analysis:
EvoPrompt offers:
- **Model Improvement Without Additional Training:**
As you pointed out, EvoPrompt provides a data-driven way to improve model performance without additional training. This is crucial for scenarios where re-training a model is either computationally expensive or practically infeasible.

- **Black-Box Utilization:**
The ability to treat LLMs as black-boxes opens up the possibility of applying EvoPrompt across different domains and for different tasks, making it a versatile tool for NLP applications.

- **Computational Efficiency:**
The lack of a need for gradient calculations and parameter updates significantly speeds up the optimization process. This is especially important when the optimization has to be performed multiple times or in real-time scenarios.
