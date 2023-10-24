# CONNECTING LARGE LANGUAGE MODELS WITH EVOLUTIONARY ALGORITHMS YIELDS POWERFUL PROMPT OPTIMIZERS
**Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang**
****************************************************************************************************

Notes:
- It's an interesting cycle because you use LLMs to make the results from LLMs better.
- Algorithm is called EvoPrompt
- There's two different ways to implement EvoPrompt:
  - Genetic algorithm
  - Differential Evolution
  - In summary, we suggest choosing EVOPROMPT (GA) when several high-quality prompts already exist, and choosing EVOPROMPT (DE) otherwise.
    
- Benefits include:
  - You can perform this on a black box model-as-a-service.
  - Improvements up to 14 percent.
