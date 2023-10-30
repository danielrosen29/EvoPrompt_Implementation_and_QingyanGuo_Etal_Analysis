import gradio as gr
import openai
import rouge_score
import pandas as pd
import numpy as np
import evaluate
import streamlit as st
from datetime import datetime
from matplotlib import pyplot as plt
from googleapiclient import discovery
from copy import deepcopy

with open("C:/Users/danie/OneDrive/Desktop/openai_youtube_api_key.txt") as f:
    api_key = f.readline()

openai.api_key = api_key

rouge_score = evaluate.load("rouge")

st.set_page_config(layout="wide")

STORY = """The Quantum Quest of Clara the Chicken

    **The Mysterious Equation**

    Clara, a curious chicken on Farmer Brown's farm, stumbled upon a strange equation scribbled on the barn door: E=mc². Intrigued, she felt an overwhelming urge to understand it.

    **The Road as a Metaphor**

    The road beside the farm became a metaphor for her quest for knowledge. "To cross or not to cross, that is the question," she clucked.

    **The Quantum Leap**

    One night, a meteor shower lit up the sky, and Clara felt an inexplicable force pulling her towards the road. With each step, the equation in her mind seemed to unravel, transforming into visions of atoms, galaxies, and parallel universes.

    **The Other Side**

    As she reached the other side, the world briefly turned into a mesh of numbers and equations, revealing the interconnectedness of all things. She understood that E=mc² was more than an equation; it was the essence of existence.

    **Return and Enlightenment**

    Clara returned to the farm, not just as a chicken who had crossed the road, but as a sentient being aware of the cosmic dance of matter and energy. When asked why she crossed the road, her answer was simple yet profound: "To understand the essence of being, you must first cross your own boundaries."

    And so, Clara lived her days not just as a chicken but as an enlightened soul, forever cherishing her quantum leap across the road."""

def AddHumanEngineeredPrompts(prompt):
    messages = []
    user_query = prompt
    messages.append({"role": "user", "content": user_query})
    return messages

def GenerateRandomPromptsLLM(N, user_prompt):
    num_generated = N-1
    prompts = []
    i = 0
    while i < num_generated:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": f"""Generate a new dictionary which is in the exact same format as the following prompt: \n {user_prompt}
                Your new dictionary's contents section should convey the same idea and attempt to accomplish the same task as the original in a nuianced way.
                Do not changed the role section of the dictionary (it should remain as 'user')
                Your answer should only contain single quotes and they should only be used to mark where strings should be.
                If a word needs an apostrophe, leave it out."""}
            ],
            #temperature = .5,
            #frequency_penalty = -.3
        )
        if response['choices'][0]['message']['content'] != user_prompt:
            try:
                prompts.append(eval(response['choices'][0]['message']['content']))
                i += 1
            except SyntaxError:
                continue

    return prompts

def print_starting_generation(prompts):
    st.write('---------------------------------------------------------------------')
    st.markdown("# Initialization Prompts:")
    for i, p in enumerate(prompts):
        st.subheader(f"**Prompt {i+1}:**")
        st.markdown(f"{p[0]['content']}")
        st.write(' ')
    

def print_generation_best(scores, preds, prompts, argmax, col):
    with col:
        content = prompts[argmax]
        st.markdown(f"### **Best Candidate:** Prompt {argmax+1}")
        st.markdown(f"**Prompt:**\n{content[0]['content']}")
        st.write(' ')
        st.markdown(f"### Response: \n{preds[argmax]}")
        st.markdown(f"**Rouge1 Score:** {scores[argmax]}")
        st.write(' ')

def roulette_wheel_selection(scores):
    # Min-Max normalization
    min_val = np.min(scores)
    max_val = np.max(scores)
    normalized_scores = (scores - min_val) / (max_val - min_val)
    
    # Make sure they sum to 1 for probabilities
    normalized_scores /= normalized_scores.sum()
    
    # Randomly select two indices based on their probabilities
    selected_indices = np.random.choice(len(scores), 2, replace=False, p=normalized_scores)
    
    return selected_indices

def evoprompt_ga(num_prompts, num_iterations, role, user_prompt, target_response):
    #Step 1: Initialize Populations
    P = [] # Prompt Population
    best_scores = []
    
    human_prompt = AddHumanEngineeredPrompts(user_prompt)
    hp_temp = deepcopy(human_prompt)
    hp_temp[0]['content'] = "Given the following story:\n" + STORY + hp_temp[0]['content'] 
    hp_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=hp_temp
    )
    hp_pred = hp_response['choices'][0]['message']['content']
    hp_score = rouge_score.compute(
        predictions=[hp_pred],
        references=[target_response]
    )['rouge1']
    P.append(human_prompt)
    
    model_prompts = GenerateRandomPromptsLLM(num_prompts, P[0])
    for prompt in model_prompts:
        P.append(prompt)
    print_starting_generation(P)
    st.write()
    st.write('---------------------------------------------------------------------')
    
    for t in range(num_iterations):
        st.markdown(f"# Generation {t}:")
        results_col, log_col = st.columns([.8, .2])
        with log_col:
            st.markdown("### Log:")
            
        scores = []
        preds = []
        for p in P: #Calculate rouge scores of all prompts responses 
            temp_p = deepcopy(p)
            temp_p[0]['content'] = "Given the following story:\n" + STORY + temp_p[0]['content'] 
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=temp_p
            )
            pred = response['choices'][0]['message']['content']
            preds.append(pred)
            p_i_score = rouge_score.compute(
                predictions=[pred],
                references=[target_response]
            )  
            scores.append(round(p_i_score['rouge1'], 3))
            
        max_index = np.argmax(scores).astype(int)
        best_candidate_score = scores[max_index]
        best_scores.append(best_candidate_score)
        print_generation_best(scores, preds, P, max_index, results_col)
        #perform roulette wheel selection 
        parents = [P[i] for i in roulette_wheel_selection(scores)]
        p1 = str(parents[0])
        p2 = str(parents[1])
        
        with log_col:
            st.write("Selection Stage Complete")
        
        #Crossover
        CROSSOVER_PROMPT = f"""Given the following two parent prompts:
        
        Prompt 1: {p1}
        
        Prompt 2: {p2}
        
        Your task is to create a new prompt which exactly matches the format of the original dictionaries while changing the content section.
        Change the content section of your response by combining portions of the original two prompts' content section.
        Do not add new line characters. Only use single quotes.
        Return only the dictionary. Do not return an explanation for how the crossover was accomplished.
        """
        #Importantly, if the example mentions comparing sentences or statements make sure to use the same sentences or statements from the example prompts in the dictionary you return.
        crossover_prompts = []
        i = 0
        while i < num_prompts:
            new_prompt = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    #{"role": "system", "content": SYS_ROLE}, 
                    {"role": "user", "content": CROSSOVER_PROMPT}
                ],
                #temperature = .65,
                #frequency_penalty = -.3
            )
            content = new_prompt['choices'][0]['message']['content']
            if content != p1 and content != p2:
                try:
                    crossover_prompts.append(eval(content))
                    i += 1
                except SyntaxError:
                    continue
        
        with log_col:
            st.write("Crossover Stage Complete")
            
        #Mutate
        mutated_prompts = []
        for co_prompt in crossover_prompts:
            MUTATE_PROMPT = f"""Given the following prompt:
            {co_prompt}
            Without changing the structure of the dictionary, mutate the following prompt's content sections.
            Change these sections by replacing the words with synonyms, rephrasing ideas, and by adding to the prompt to be more task specific.
            Only use single quotes. Return only the list of dictionaries. Do not return an explanation for how the mutation was accomplished.
            """
            #Importantly, if the prompt mentions comparing sentences or statements make sure to use the original sentences or statements in you response.
            new_prompt = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "user", "content": MUTATE_PROMPT}
                ],
                #temperature = .65,
                #frequency_penalty = -.3
            )
            content = new_prompt['choices'][0]['message']['content']
            if content != co_prompt:
                try:
                    mutated_prompts.append(eval(content))
                except SyntaxError:
                    continue
                
        for m_prompt in mutated_prompts:
            P.append(m_prompt)
        with log_col:
            st.write("Mutation Stage Complete")

        #Evaluation

        survival_scores = []
        with log_col:
            st.write(f"Calculating Scores for Prompts.")
        for p in P: #Calculate rouge scores of all prompt responses  
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=p
            )
            pred = response['choices'][0]['message']['content']
            p_i_score = rouge_score.compute(
                predictions=[pred],
                references=[target_response]
            )  
            survival_scores.append(p_i_score['rouge1'])
            
        with log_col:   
            st.write("Evaluation Stage Complete")
                
        #Select next generation:
        max_indices = np.argsort(survival_scores)
        P = [P[i] for i in max_indices[3:]]
        with log_col:
            st.write(f"Generation {t}: Generation Complete. Offspring Selected!")
        st.write('---------------------------------------------------------------------')
        
    
    final_scores = []
    final_preds = []
    for p in P: #Calculate rouge scores of all prompts responses  
        temp_p = deepcopy(p)
        temp_p[0]['content'] = "Given the following story:\n" + STORY + temp_p[0]['content'] 
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=temp_p
        )
        pred = response['choices'][0]['message']['content']
        final_preds.append(pred)
        p_i_score = rouge_score.compute(
            predictions=[pred],
            references=[target_response]
        )  
        final_scores.append(round(p_i_score['rouge1'], 3))
    
    st.write()
    st.title(f"Original Prompt:")
    st.markdown(f"{human_prompt[0]['content']}")
    st.write(' ')
    st.markdown(f"## Response: \n{hp_pred}")
    st.markdown(f"**Rouge1 Score:** {round(hp_score, 3)}")

    final_prompt_idx = np.argmax(final_scores)
    content = P[final_prompt_idx]
    final_response = final_preds[final_prompt_idx]

    st.write()
    st.write('---------------------------------------------------------------------')
    st.title(f"Final Prompt:")
    st.markdown(f"{content[0]['content']}")
    st.write(' ')
    st.markdown(f"## Response: \n{final_response}")
    st.markdown(f"**Rouge1 Score:** {final_scores[final_prompt_idx]}")
    st.write(' ')

    #Plot our results
    best_scores = [hp_score] + best_scores + [final_scores[final_prompt_idx]]
    generations = list(range(len(best_scores)))
    #st.write(best_scores)
    # Create a DataFrame from the dictionary
    plotting_df = pd.DataFrame(
        {
            'Generation': generations,
            'BestScore': best_scores
        }
    )

    st.subheader("Visualization of Improvements via EvoPrompt(GA)")
    st.line_chart(plotting_df, x = "Generation", y = 'BestScore',)

st.markdown("""
# EvoPrompt

## Task:

**Given the following short story:**


        The Quantum Quest of Clara the Chicken

        Clara, a curious chicken on Farmer Brown's farm, stumbled upon a strange equation scribbled on the barn door: E=mc².
        Intrigued, she felt an overwhelming urge to understand it.
        The road beside the farm became a metaphor for her quest for knowledge. "To cross or not to cross, that is the question," she clucked.
        One night, a meteor shower lit up the sky, and Clara felt an inexplicable force pulling her towards the road.
        With each step, the equation in her mind seemed to unravel, transforming into visions of atoms, galaxies, and parallel universes.
        As she reached the other side, the world briefly turned into a mesh of numbers and equations, revealing the interconnectedness of all things.
        She understood that E=mc² was more than an equation; it was the essence of existence.
        Clara returned to the farm, not just as a chicken who had crossed the road, but as a sentient being aware of the cosmic dance of matter and energy.
        When asked why she crossed the road, her answer was simple yet profound: "To understand the essence of being, you must first cross your own boundaries."
        And so, Clara lived her days not just as a chicken but as an enlightened soul, forever cherishing her quantum leap across the road.


**Create a prompt which attempts to match the following output created with a hidden query:**


        <HIDDEN TEXT WHICH REPEATS THE PROMPT> is her insatiable curiosity and quest for knowledge.
        The mysterious equation E=mc² serves as the catalyst that ignites her desire to understand the deeper meanings and complexities of existence.
        This curiosity propels her to cross the road, a metaphorical boundary that separates the known from the unknown in her life.
        Her character undergoes significant transformation during her "quantum leap" across the road,
        where she gains a profound understanding of the interconnectedness of all things.
        This experience elevates her from a simple farm chicken to a sentient being aware of greater cosmic truths.
        
##### Note: The story context is added to the prompt in the algorithm. There is no need to add it in the prompt.
""")

N = st.number_input("Number of Prompts (N): ", min_value=3, max_value=10, step=1, value=4)
T = st.number_input("Number of Interations (T): ", min_value=1, max_value=15, value=3, step=1)
prompt = st.text_area("Prompt: ")
if st.button("EvoPrompt(GA)"):
    st.write(evoprompt_ga(N, T, 'assistant', prompt, """The driving force for Clara's character growth is her insatiable curiosity and quest for knowledge. The mysterious equation E=mc² serves as the catalyst that ignites her desire to understand the deeper meanings and complexities of existence. This curiosity propels her to cross the road, a metaphorical boundary that separates the known from the unknown in her life.

Her character undergoes significant transformation during her "quantum leap" across the road, where she gains a profound understanding of the interconnectedness of all things. This experience elevates her from a simple farm chicken to a sentient being aware of greater cosmic truths.

In essence, Clara's character growth is fueled by her intellectual curiosity, her courage to step beyond her comfort zone, and her transformative experiences that come from confronting and embracing the unknown."""))

