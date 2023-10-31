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

def format_prompt(prompt):
    return {"role": "user", "content": prompt}

def GenerateRandomPromptsLLM(N, user_prompt):
    num_generated = N-1
    prompts = [user_prompt]
    i = 0
    while i < num_generated:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": f"""Please read the prompt following the <prompt> tag and rewrite it in a way that changes its structure and wording significantly, yet still conveys the same meaning or accomplishes the same task as the original sentence.
                 Your goal is to create a prompt that is functionally equivalent but stylistically distinct and more robust than the original.
                 
            <prompt>
            {user_prompt}
            """}
            ],
            temperature = .9,
            #frequency_penalty = -.3
        )
        temp = response['choices'][0]['message']['content']
        if temp != user_prompt:
            try:
                prompts.append(temp)
                i += 1
            except:
                continue

    return prompts

def print_starting_generation(prompts):
    st.write('---------------------------------------------------------------------')
    st.markdown("# Initialization Prompts:")
    for i, p in enumerate(prompts.keys()):
        st.subheader(f"**Prompt {i+1}:**")
        st.markdown(f"{p}")
        st.write(' ')
    

def print_generation_best(P, prompt, col):
    with col:
        st.markdown(f"### **Best Candidate:**")
        st.markdown(f"**Prompt:**\n{prompt}")
        st.write(' ')
        st.markdown(f"### Response: \n{P[prompt]['response']}")
        st.markdown(f"**Rouge1 Score:** {P[prompt]['score']}")
        st.write(' ')

def roulette_wheel_selection(dic):
    score_dic = {features['score']:prompt for (prompt,features) in dic.items()}
    scores = list(score_dic.keys())
    # Min-Max normalization
    min_val = np.min(scores)
    max_val = np.max(scores)
    normalized_scores = (scores - min_val) / (max_val - min_val)
    
    # Make sure they sum to 1 for probabilities
    normalized_scores /= normalized_scores.sum()
    
    # Randomly select two indices based on their probabilities
    selected_scores = np.random.choice(scores, 2, replace=False, p=normalized_scores)
    prompt_1 = score_dic[selected_scores[0]]
    prompt_2 = score_dic[selected_scores[1]]
    return (prompt_1, prompt_2)

def get_best_prompt(dic):
    best_prompt = ""
    max_score = 0
    for p, v in dic.items():
        if v['score'] > max_score:
            max_score = v['score']
            best_prompt = p
    
    return best_prompt

def evoprompt_ga(num_prompts, num_iterations, role, user_prompt, target_response):
    #Step 1: Initialize Populations
    P = {} # Prompt Population
    best_scores = []
    model_prompts = GenerateRandomPromptsLLM(num_prompts, user_prompt)
    for prompt in model_prompts:
        P[prompt] = {}
    print_starting_generation(P)
    
    for p in P: #Calculate rouge scores of all prompts responses 
        temp_p = deepcopy(p)
        temp_p = "Given the following story:\n" + STORY + temp_p 
        temp_p = format_prompt(temp_p)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[temp_p]
        )
        pred = response['choices'][0]['message']['content']
        P[p]['response'] = pred
        p_i_score = rouge_score.compute(
            predictions=[pred],
            references=[target_response]
        )  
        P[p]['score'] = round(p_i_score['rouge1'], 3)
    
    user_response = P[user_prompt]['response']
    user_score = P[user_prompt]['score']
    
    results_col, log_col = st.columns([.8, .2])    
    best_prompt = get_best_prompt(P)
    best_scores.append(P[best_prompt]['score'])
    print_generation_best(P, best_prompt, results_col)
    
    st.write()
    st.write('---------------------------------------------------------------------')
      
    
    #Evolutionary Loop
    for t in range(num_iterations):
        st.markdown(f"# Generation {t+1}:")
        results_col, log_col = st.columns([.8, .2])
        with log_col:
            st.markdown("### Log:")
            
        #perform roulette wheel selection 
        p1,p2 = roulette_wheel_selection(P)
        with results_col:
            st.markdown("### Parents Selected: ")
            st.markdown(f"**Parent Prompt 1:** {p1}")
            st.markdown(f"**Parent Prompt 2:** {p2}")
        
        with log_col:
            st.write("Parent Selection Stage Complete")
        
        #Crossover Section
        CROSSOVER_PROMPT = f"""Given the following two parent prompts which come after the <prompt> tag,
        create a new prompt by crossing over or combining portions of the parents. 
        The new prompt should convey the same idea and or accomplish the same task as the parents.
        Your new prompt should only contain single quotes and should not include the <prompt> tag.
        
        <prompt>
        Prompt 1: {p1}
        
        Prompt 2: {p2}
        """
        
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
            MUTATE_PROMPT = f"""Given the prompt which comes after the <prompt> tag,
            create a new prompt by mutating the original.
            These mutations should include but are not limited to:
            
            - Replacing the words with synonyms
            - Rephrasing ideas
            - Adding to the original to be more effective at completing its task.
            
            Your new prompt should only contain single quotes and should not include the <prompt> tag.
            
            <prompt>
            {co_prompt}
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
                
        with log_col:
            st.write("Mutation Stage Complete")
        #Evaluation        
        with log_col:
            st.write(f"Calculating Scores for Prompts.")
        for p in mutated_prompts: #Calculate rouge scores of all prompt responses  
            P[p] = {}
        for p in P.copy().keys():
            if not P[p]:
                temp_p = deepcopy(p)
                temp_p = "Given the following story:\n" + STORY + temp_p 
                temp_p = format_prompt(temp_p)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=[temp_p]
                )
                pred = response['choices'][0]['message']['content']
                P[p]['response'] = pred
                p_i_score = rouge_score.compute(
                    predictions=[pred],
                    references=[target_response]
                )  
                P[p]['score'] = round(p_i_score['rouge1'], 3)
        
        best_prompt = get_best_prompt(P)
        best_scores.append(P[best_prompt]['score'])
        print_generation_best(P, best_prompt, results_col)    
        with log_col:   
            st.write("Evaluation Stage Complete")
                
        #Select next generation:
        survival_scores = {features['score']:prompt for (prompt,features) in P.items()}
        P_temp = sorted(list(survival_scores.keys()), reverse=True)
        P_temp = P_temp[:num_prompts]
        P_reduced = [survival_scores[s] for s in P_temp]
        for p in P.copy().keys():
            if p not in P_reduced:
                del P[p]
                
        with log_col:
            st.write(f"Generation {t}: Generation Complete. Offspring Selected!")
        st.write('---------------------------------------------------------------------')
    
    #Find Best:
    final_score = best_scores[-1]
    final_prompt = ""
    final_response = ""
    for p, v in P.items():
        if v['score'] == final_score:
            final_prompt = p
            final_response = v['response']
         
    st.write()
    st.title(f"Original Prompt:")
    st.markdown(f"{user_prompt}")
    st.write(' ')
    st.markdown(f"## Response: \n{user_response}")
    st.markdown(f"**Rouge1 Score:** {round(user_score, 3)}")

    st.write()
    st.write('---------------------------------------------------------------------')
    st.title(f"Final Prompt:")
    st.markdown(f"{final_prompt}")
    st.write(' ')
    st.markdown(f"## Response: \n{final_response}")
    st.markdown(f"**Rouge1 Score:** {final_score}")
    st.write(' ')

    #Plot our results
    best_scores = [user_score]+best_scores
    generations = list(range(len(best_scores)))
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
# Model evaluation:
        
**The prompt we just develop was scored against the following piece of text using the ROUGE-1 score.**

---

We live in a time where technology and information are growing really fast. This brings both good and bad things. On one hand, we can easily access a lot of useful information. On the other hand, there's also a lot of false information out there.

As things keep changing, it's important for us to be smart about what information we believe. We have to be careful not to get trapped in "echo chambers" where we only hear opinions like our own, as this can make us narrow-minded.

There are also a lot of conflicts and divisions in society today, both locally and globally. Leaders sometimes make these worse by focusing only on what a specific group wants, and this can be harmful to everyone. So, we should all try to be open-minded and inclusive in how we think and act.

Technology is advancing quickly, offering great possibilities like cleaner energy and better medical treatments. But we also have to think about the downsides, like job loss due to automation, and ethical questions around things like AI and biotech.

In short, the modern world is complex and filled with both challenges and opportunities. It's up to each of us to be well-informed and responsible in how we navigate through it.

---

**This text was generated with the following prompt:**

Given the previous text,  please rewrite it in a simplified manner. Your goal is to capture the essence and key points of the original text, without losing important information. 
Remove jargon, elaborate vocabulary, and convoluted sentences to make it easily understandable for a general audience.

---

### ROUGE-1

The ROUGE-1 score is a metric used to evaluate the quality of summaries by comparing them to reference summaries. It is part of the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) set of metrics. Specifically, ROUGE-1 measures the overlap of single words (unigrams) between the generated summary and the reference summary. 

The ROUGE-1 score is calculated based on the following two metrics:

1. **Recall**: This measures what fraction of the words in the reference summary also appear in the generated summary. A higher recall means the generated summary covers more of the content in the reference summary.

2. **Precision**: This measures what fraction of the words in the generated summary also appear in the reference summary. A higher precision means the generated summary has fewer extraneous words not in the reference summary.
    
---
# Thanks for Watching!!!  
        
        """)
st.markdown("""
# EvoPrompt(GA)
---
# Task:

### Given the following complex body of text:
---

In an era characterized by burgeoning technological innovations and the rapid dissemination of information, society finds itself at an epochal crossroads. The labyrinthine interplay of ethical quandaries, socio-political vicissitudes, and economic volatilities has engendered a milieu replete with both opportunities and pitfalls. Paradoxically, the selfsame conduits that facilitate the untrammeled flow of knowledge also serve as a breeding ground for misinformation and malevolent agendas.

As the zeitgeist of our contemporary existence continually shifts, it's imperative that we exercise sagacity in sifting through the avalanche of data that inundates our daily lives. The pernicious influence of echo chambers and ideological silos cannot be overstated; these insidious constructs obfuscate objective truths and engender a myopic worldview.

Moreover, the sociopolitical landscape is fraught with internecine conflicts that have both local and global repercussions. Factionalism and sectarianism, often exacerbated by demagogues exploiting parochial interests, threaten to rend the very fabric of civil society. In this context, it behooves us to be circumspect in our judgments and assiduous in our efforts to cultivate an ethos of inclusivity.

Concomitantly, the inexorable march of technological progress presents a double-edged sword. While advancements in artificial intelligence, biotechnology, and renewable energy sources offer the tantalizing prospect of a utopian future, they also raise disquieting questions about the potential obsolescence of the human workforce and the ethical implications of unfettered scientific inquiry.

In summation, the multifaceted challenges and opportunities that confront us necessitate a holistic and nuanced approach. The onus is on each individual to be an informed, responsible, and proactive participant in this complex tapestry of modern life.


**TLDR: The original text discusses the complexities of modern life, emphasizing the dual nature of technology, information, and politics. It urges people to think critically about the information they consume and the leaders they follow, while also considering the ethical implications of rapid technological advancements.**

---

### Create a prompt which best simplifies the text while maintaining the important points.

""")

N = st.number_input("Number of Prompts (N): ", min_value=3, max_value=10, step=1, value=4)
T = st.number_input("Number of Interations (T): ", min_value=1, max_value=15, value=3, step=1)
prompt = st.text_area("Prompt: ")
st.markdown("**Note: The context is added to the prompt in the algorithm. There is no need to add it in the text box.**")
if st.button("EvoPrompt(GA)"):
    st.write(evoprompt_ga(N, T, 'assistant', prompt, """We live in a time where technology and information are growing really fast. This brings both good and bad things. On one hand, we can easily access a lot of useful information. On the other hand, there's also a lot of false information out there.

As things keep changing, it's important for us to be smart about what information we believe. We have to be careful not to get trapped in "echo chambers" where we only hear opinions like our own, as this can make us narrow-minded.

There are also a lot of conflicts and divisions in society today, both locally and globally. Leaders sometimes make these worse by focusing only on what a specific group wants, and this can be harmful to everyone. So, we should all try to be open-minded and inclusive in how we think and act.

Technology is advancing quickly, offering great possibilities like cleaner energy and better medical treatments. But we also have to think about the downsides, like job loss due to automation, and ethical questions around things like AI and biotech.

In short, the modern world is complex and filled with both challenges and opportunities. It's up to each of us to be well-informed and responsible in how we navigate through it."""))

