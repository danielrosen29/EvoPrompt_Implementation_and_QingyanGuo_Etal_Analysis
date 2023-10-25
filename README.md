# CONNECTING LARGE LANGUAGE MODELS WITH EVOLUTIONARY ALGORITHMS YIELDS POWERFUL PROMPT OPTIMIZERS
**Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang**
****************************************************************************************************
## Introduction: 
In recent times it has become clear that Large Language Models (LLMs) exhibit remarkable capabilities for performing various Natural Language Processing (NLP) tasks. However, achieving peak performance is often contingent on the quality of the prompt fed to the model. Traditionally, crafting these prompts has required substantial human effort, often relying on engineers with specialized knowledge and deep understanding of the models in question. In this paper, Qingyan Guo et al. propose a data-driven, automated technique for prompt development called EvoPrompt. EvoPrompt leverages Evolutionary Algorithms (EA) to iteratively refine and optimize prompts for specific NLP tasks without the need for human intervention. Their findings indicate that this automated approach can yield improvements in performance by up to 25% in some cases, marking a significant advancement in the field of prompt engineering.
 
## Discrete Prompts:
A discrete prompt is the use of explicit instructions to the input text fed to LLMs. These discrete prompts guide LLMs perform specific tasks with negligible increases to computational cost meanwhile eliminating
the need to access the parameters and gradients of the models (Liu et al., 2023). These are the types of prompts which EvoPrompt aims to improve. 

## Other Discrete Prompt Generation Methods:
- Reinforcement learning (RL): Methods like RLPrompt (Deng et al. 2022) and TEMPERA (Zhang et al. 2023) train an RL agent to generate prompts by interacting with the LLM. They require access to model internals like output probabilities.
- Enumeration: Methods like PromptSource (Bach et al. 2022) and APE (Zhou et al. 2022) generate a large set of prompt candidates via techniques like sampling and then select the best. They focus on exploration but can be inefficient.
- Revision: Approaches like GRIPS (Prasad et al. 2022) and AutoPrompt (Shin et al. 2020) iterate on an initial prompt by fixing incorrect predictions. They emphasize local search so can get stuck in local optima.
- Editing: Methods like Instruction Tuning (Prasad et al. 2022) edit the words in a prompt to improve it. They also focus on local improvements.

## Evolutionary Algorithms:
Evolutionary Algorithms (EAs) are a family of optimization algorithms inspired by the process of natural evolution. These algorithms are heuristics used to find approximate solutions to optimization and search problems. There are many different types and subtypes of EAs including but not limited to:

- Genetic Algorithms (GAs): Perhaps the most well-known type, focused on string-based chromosomes and typically employs crossover, mutation, and selection.
- Differential Evolution (DE): A population-based optimization algorithm that optimizes a problem by iteratively improving candidate solutions with regard to a given measure of quality.
- Estimation of Distribution Algorithms (EDAs): Rather than using crossover and mutation, these algorithms build probabilistic models of promising solutions and sample from these models to generate new candidates.
- Neuroevolution – A method for iteratively building a neural network through genetic propogation. 

## EvoPrompt:
![image](https://github.com/danielrosen29/QingyanGuo_Etal_Analysis/assets/75226826/162d89db-a9ee-4772-81cb-c2ffa9916c87)

This algorithm is the general outline for implementing the evolutionary process of discrete prompts, but can be implemented differnt ways depending on which evolutionary operators (EO) are used. In the paper, Qingyan Guo et al. propose two implementations. The first of these is using a Genetic Algorithm as the EO, the second uses Differential Evolution. The notation for these in the paper is EvoPrompt(EA) and EvoPrompt(DE), respectively. 

### EvoPrompt(EA):
EvoPrompt(GA):

// EvoPrompt with Genetic Algorithm
//Algorithm EvoPrompt(N, T)
    Input:
        N: Size of the population of prompts
        T: Number of iterations
    
    // Step 1: Initialize Population
    P <- Empty list of size N
    
    // Include human-engineered prompts
    AddHumanEngineeredPrompts(P)
    
    // Generate random prompts using LLM
    GenerateRandomPromptsLLM(P)
    
    // Step 2: Evolutionary Loop
    for t = 1 to T do
    
        // Step 2.1: Parent Selection via Roulette Wheel
        p1, p2 <- RouletteWheelSelection(P)
        
        // Calculate selection probability based on performance score
        pi <- CalculateSelectionProbability(p1, p2, P)
        
        // Step 2.2: Crossover
        // Instruct LLM to combine parts of p1 and p2 to create a new prompt
        p' <- LLMCrossover(p1, p2)
        
        // Step 2.3: Mutation
        // Instruct LLM to randomly change words in the crossover result
        p'' <- LLMMutate(p')
        
        // Step 2.4: Evaluation
        // Evaluate the mutated prompt on a development set to get its score
        s'' <- EvaluateOnDevSet(p'')
        
        // Step 2.5: Update Population
        // Add new prompt and its score to the population
        AddToPopulation(P, p'', s'')
        
        // Remove the prompt with the lowest score to maintain population size N
        RemoveLowestScorePrompt(P)
        
    end for
    
    // Step 3: Return the Best Prompt
    p* <- GetHighestScorePrompt(P)
    
    return p*


### EvoPrompt(DE):

Initialize population P with N prompts
For t = 1 to T iterations:
For each prompt pi in P:
Randomly select prompts a, b, c from P
Instruct LLM to identify different parts between b and c
Instruct LLM to mutate different parts
Instruct LLM to incorporate differences into a to get p'
Instruct LLM to crossover p' with pi to get p''
Evaluate p'' on dev set D to get score s''
If s'' > si, replace pi with p'' in P
Return prompt p* in P with highest score


Notes:
- It's an interesting cycle because you use LLMs to make the results from LLMs better.
- Algorithm is called EvoPrompt
- There's two different ways to implement EvoPrompt:
  - Genetic algorithm
  - Differential Evolution
  - In summary, we suggest choosing EVOPROMPT (GA) when several high-quality prompts already exist, and choosing EVOPROMPT (DE) otherwise.
  - We optimize prompts for both closed- and open-source LLMs including GPT-3.5 and Alpaca, on 9 datasets spanning language understanding and generation tasks. EVOPROMPT significantly outperforms human-engineered prompts and existing methods for automatic prompt generation by up to 25% and 14% respectively
  - Given the wide variation in prompts across language models and tasks, the prompt design typically requires substantial human effort and expertise with subjective and relatively limited guidelines.
    
    
- Benefits include:
  - You can perform this on a black box model-as-a-service.
  - Improvements up to 14 percent.
  - Non gradient based algo
  - Data Driven

- Negatives:	
	- It doesn't seem like you can just do it for any prompt because you need to have some sort of scoring metric, this means it likes only works for instructions with some predefined way to score. 
