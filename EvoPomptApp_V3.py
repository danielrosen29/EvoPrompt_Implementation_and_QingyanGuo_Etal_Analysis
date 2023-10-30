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

STORY = """In an era characterized by burgeoning technological innovations and the rapid dissemination of information, society finds itself at an epochal crossroads. The labyrinthine interplay of ethical quandaries, socio-political vicissitudes, and economic volatilities has engendered a milieu replete with both opportunities and pitfalls. Paradoxically, the selfsame conduits that facilitate the untrammeled flow of knowledge also serve as a breeding ground for misinformation and malevolent agendas.

As the zeitgeist of our contemporary existence continually shifts, it's imperative that we exercise sagacity in sifting through the avalanche of data that inundates our daily lives. The pernicious influence of echo chambers and ideological silos cannot be overstated; these insidious constructs obfuscate objective truths and engender a myopic worldview.

Moreover, the sociopolitical landscape is fraught with internecine conflicts that have both local and global repercussions. Factionalism and sectarianism, often exacerbated by demagogues exploiting parochial interests, threaten to rend the very fabric of civil society. In this context, it behooves us to be circumspect in our judgments and assiduous in our efforts to cultivate an ethos of inclusivity.

Concomitantly, the inexorable march of technological progress presents a double-edged sword. While advancements in artificial intelligence, biotechnology, and renewable energy sources offer the tantalizing prospect of a utopian future, they also raise disquieting questions about the potential obsolescence of the human workforce and the ethical implications of unfettered scientific inquiry.

In summation, the multifaceted challenges and opportunities that confront us necessitate a holistic and nuanced approach. The onus is on each individual to be an informed, responsible, and proactive participant in this complex tapestry of modern life.."""

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
                {"role": "user", "content": f"""Generate a new dictionary which is in the exact same format as the following dictionary which is in between the <prompt> tags: 
                <prompt>
                {user_prompt}
                <prompt>
                 
                Adhere the following rules when responding:
                - Do not include <prompt> tags in your response.
                - This dictionary's contents section is a prompt for a large language model and your task is to create a new dictionary
                whose content section contains a new prompt which conveys the same idea and attempts to accomplish the same task as the original.
                - Consider adding more text to the original prompt to make the the new prompt more effective.
                - Do not changed the role section of the dictionary (it should remain as 'user')
                - Your answer should only contain single quotes and they should only be used to mark where strings should be.
                - If a word needs an apostrophe, leave it out.
                - Finally, only give one dictionary in your response."""}
            ],
            #temperature = .5,
            #frequency_penalty = -.3
        )
        if response['choices'][0]['message']['content'] != user_prompt:
            try:
                prompts.append(eval(response['choices'][0]['message']['content']))
                i += 1
            except SyntaxError as e:
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
    st.write("here")
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
        for p in mutated_prompts: #Calculate rouge scores of all prompt responses  
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

### Given the following complex body of text:


In an era characterized by burgeoning technological innovations and the rapid dissemination of information, society finds itself at an epochal crossroads. The labyrinthine interplay of ethical quandaries, socio-political vicissitudes, and economic volatilities has engendered a milieu replete with both opportunities and pitfalls. Paradoxically, the selfsame conduits that facilitate the untrammeled flow of knowledge also serve as a breeding ground for misinformation and malevolent agendas.

As the zeitgeist of our contemporary existence continually shifts, it's imperative that we exercise sagacity in sifting through the avalanche of data that inundates our daily lives. The pernicious influence of echo chambers and ideological silos cannot be overstated; these insidious constructs obfuscate objective truths and engender a myopic worldview.

Moreover, the sociopolitical landscape is fraught with internecine conflicts that have both local and global repercussions. Factionalism and sectarianism, often exacerbated by demagogues exploiting parochial interests, threaten to rend the very fabric of civil society. In this context, it behooves us to be circumspect in our judgments and assiduous in our efforts to cultivate an ethos of inclusivity.

Concomitantly, the inexorable march of technological progress presents a double-edged sword. While advancements in artificial intelligence, biotechnology, and renewable energy sources offer the tantalizing prospect of a utopian future, they also raise disquieting questions about the potential obsolescence of the human workforce and the ethical implications of unfettered scientific inquiry.

In summation, the multifaceted challenges and opportunities that confront us necessitate a holistic and nuanced approach. The onus is on each individual to be an informed, responsible, and proactive participant in this complex tapestry of modern life.

**TLDR: The original text discusses the complexities of modern life, emphasizing the dual nature of technology, information, and politics. It urges people to think critically about the information they consume and the leaders they follow, while also considering the ethical implications of rapid technological advancements.**

##### Create a prompt which attempts to match the simplify the following output such that non-native English speakers might be able to understand it better:
        
**Note: The context is added to the prompt in the algorithm. There is no need to add it in the prompt.**
""")

N = st.number_input("Number of Prompts (N): ", min_value=3, max_value=10, step=1, value=4)
T = st.number_input("Number of Interations (T): ", min_value=1, max_value=15, value=3, step=1)
prompt = st.text_area("Prompt: ")
if st.button("EvoPrompt(GA)"):
    st.write(evoprompt_ga(N, T, 'assistant', prompt, """We're living in a time where technology is growing fast and we have more information available than ever before. But this can be both good and bad. On one hand, it's easier to learn and connect with people. On the other hand, there's also a lot of false information out there.

Because things are always changing, it's important to think carefully about what we read and hear. We should also be aware that sometimes people only listen to opinions that agree with their own, which can make it hard to see the whole picture.

Politics is another area where there are many disagreements and conflicts. Sometimes, leaders take advantage of these conflicts for their own benefit, which can cause more problems in society. So, we need to be careful about who we listen to and make sure we're including everyone's viewpoint.

Technology is moving forward quickly, and that brings a lot of good things like better medicine and cleaner energy. But it also means that some jobs could be replaced by machines, and there are questions about what is right and wrong in science.

In short, life today is complicated with many chances to succeed but also many challenges. It's up to each of us to be informed and make good choices."""))

