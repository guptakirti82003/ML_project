import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from PIL import Image
import requests
import json
import time

##### For handling CSV file #####
import pandas as pd
import numpy as np

#### For Text preprocessing modules ####
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#### For Converting the text into Numerical Data ####
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#### For Distribution of Data into Training and Testing
from sklearn.model_selection import train_test_split

#### Model use for Predicting the Personality ####
from sklearn.linear_model import LogisticRegression

###### Use for Text Preprocessing ######

##################################################################################
##################################################################################



st.set_page_config(page_title="Personality_Predictor",page_icon="🤔")


################## for Windows ##################
def load_lottiefile(filepath: str): #Applying animation from json file
    with open(filepath,"r",encoding='cp850') as f:
        return json.load(f)

##### LOAD ANIMATIONS #####
robot1_animate=load_lottiefile("lottiefiles/robot1.json")
predictive_animate=load_lottiefile("lottiefiles/predict.json")
analyze_animate=load_lottiefile("lottiefiles/analyze.json")
working_animate=load_lottiefile("lottiefiles/working.json")

robot2_animate=load_lottiefile("lottiefiles/robot2.json")
project1_animate=load_lottiefile("lottiefiles/project1.json")
project2_animate=load_lottiefile("lottiefiles/project2.json")
project3_animate=load_lottiefile("lottiefiles/project3.json")

process_animate=load_lottiefile("lottiefiles/processing.json")
working1_animate=load_lottiefile("lottiefiles/working1.json")
working11_animate=load_lottiefile("lottiefiles/working11.json")
working2_animate=load_lottiefile("lottiefiles/working2.json")
discussion_animate=load_lottiefile("lottiefiles/discussion.json")

dev_animate=load_lottiefile("lottiefiles/dev.json")
developers_animate=load_lottiefile("lottiefiles/developers.json")
team_animate=load_lottiefile("lottiefiles/team.json")



# nav=st.sidebar.radio("NAVIGATIONS",["INTRODUCTION","ABOUT","CONTRIBUTERS"],0)
with st.sidebar:
    nav=option_menu(
    menu_title="Main Menu",
    options=["INTRODUCTION","ABOUT PROJECT","PREDICT PERSONALITY","DEVELOPERS"],
    menu_icon="cast",
    icons=["easel2-fill","file-earmark-ppt-fill","info-circle","file-person"],
    orientation="vertical",
    default_index=0,
    styles={
            # "container": {"max-width": "100000px", "background-color": "#fafafa"},
            # "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"--hover-color": "#009999"},
            # "nav-link-selected": {"background-color": "#02ab21"},
        },
    )
    
    st.write("---")
    if nav=="INTRODUCTION":
        st_lottie(robot1_animate,speed=2,loop=True,quality="high")
    elif nav=="ABOUT PROJECT":
        st_lottie(robot2_animate,speed=1,loop=True,quality="high")
    elif nav=="PREDICT PERSONALITY":
        st_lottie(process_animate,speed=2,loop=True,quality="high")
    elif nav=="DEVELOPERS":
        st_lottie(dev_animate,speed=2,loop=True,quality="high")


if nav=="INTRODUCTION": # Condition 1
    st.markdown("<h1 style='text-align:center; color:white;'><u>MYERS-BRIGGS TYPE INDICATOR</u> <h1>",True)
    st_lottie(predictive_animate,speed=1.5,loop=True,quality="high")
    
    
    with st.container():
        st.header("Well, Hello :rainbow:")
        st.header("Bonjour, Salut :rainbow:")
        st.write("---")
        st.header(":point_right: LETS QUICK INTRO:grey_exclamation:")
        st.write(
            """ So Many of you heared about the "PERSONALITY" right:grey_question:
            In this world there are many people, and each and every people has their own personality.
            Some people are Introvert, some people are Extrovert and some are judging type of people and 
            some are very sensitive type of person, and these are some basic type of personality and there
            are many more types of personality present in the earth :grey_exclamation:"""
        )
    with st.container():
        st.write("---")
        st.header(":point_right: WHAT THIS WEB-APP DO?")
        st_lottie(working_animate,speed=1,loop=True,quality="high")
        st.write("""This website helps in predicting the personality of a person by Analyzing the text.
                Recently In Intro part we talk about the personaliy. So lets understand the working 
                of this website with refrence to the Intro Part. 
                To find out the personality, the personality is already classified into 16 Types.
                This classification is based upon MBTI (Myers Briggs Type Indicator).
                
                So This website will help to find out your personality by just typing your text.
                SOUNDS COOL RIGHT , :sunglasses:"""
        )
    with st.container():
        st.write("---")
        st.header("For More Information Click the link below..:point_down:")
        st.markdown("[<h5 style='text-align:right;'>https://www.myersbriggs.org/my-mbti-personality-type</h5>](https://www.myersbriggs.org/my-mbti-personality-type)",True)
        st_lottie(analyze_animate,speed=1,loop=True,quality="high")
        st.write("---")

elif nav=="ABOUT PROJECT": # Condition 2 
    st.markdown("<h1>Project Description💻</h1>",True)
    st.write("-----------------------")
    
    ###### For first portion ######
    left_col,right_col=st.columns((5,5))
    with left_col:
        st.markdown("<h3>PROJECT NAME</h3>",True)
        st.markdown("<h4>Human Personality Prediction</h4>",True)
        st.write(
                """This Project will helps to find out the personality by
                analyzing the text, which is given by the user.
                The Outcome of this project will be the form of Group of
                4 Letters like "INTP","ENFP",etc.
                """)
    with right_col:
        st_lottie(project1_animate,speed=1,loop=True,quality="high")
    
    st.write("---")

    ###### For Second portion ######
    left_col,right_col=st.columns((5,5))
    with left_col:
        st_lottie(project2_animate,speed=1,loop=True,quality="high")
    with right_col:
        st.markdown("<h3>How to Identify your Personality <b>?</b></h3>",True)
        st.write("""
                As According to Myers-Briggs Type Indicator the personality is define 
                with the help of 4 letters, each letter is come from 2 pairs.
                
                For more Clearity see below :point_down:
                - IE: Introversion (I) / Extroversion (E) 
                - NS: Intuition (N) / Sensing (S) 
                - FT: Feeling (F) / Thinking (T) 
                - JP: Judging (J) / Perceiving (P)
                As you can see above :point_up:, each letter come from 2 pair of group.
                The Outcome of the personality will come from the combination of these,
                2 pairs.
                """)   
    
    st.write("---")
    
    ###### For third Portion ######
    left_col,right_col=st.columns((5,5))
    with left_col:
        st.markdown("""
                        <h3><u>Project Specification</u></h3>
                        """,True)
        st.markdown("""
                    - This Project is based on Machine Learning
                    - Platform Used: Python
                    - Operation Sysetm: Mac Os, Windows, Linux(Recommended)
                    - Ram: 4GB
                    - Hard_disk: 80GB
                    """)
    with right_col:
        st_lottie(project3_animate,speed=1,loop=True,quality="high")
    
    st.write("---")


elif nav=="PREDICT PERSONALITY":# condition 3
    st.write("---")
    left_col,right_col=st.columns((2,8))

    with right_col:
        st.markdown(""" 
                <h1 style='text-align:center'; font-face:'arial';font-size:'80px'> Personality Predictifier </h1>
                """,True)
    with left_col:
        st_lottie(working11_animate,speed=1,loop=True,quality="high")
    st.write("---")
    
    
    st.markdown(""" 
                <h1 style='text-align:center'; font-face:'arial';font-size:'50px'> It's Time to Predict Personality ❕</h1>
                """,True)
    st_lottie(discussion_animate,speed=1,loop=True,quality="high")

    name=st.text_input("Write your name:")
    name=str(name)
    if len(name)!=0:
        left_col,right_col=st.columns(2)
        with left_col:
            st.write("## Hi",name.capitalize(),":wave:")
            st.write("# WELCOME ")
        with right_col:
            st_lottie(working2_animate,speed=1,loop=False,quality="high")
        b_pers = {'I':1,'E':0,'N':1,'S':0,'F':1,'T':0,'J':1,'P':0}
        b_pers_list = [{1:'I',0:'E'},{1:'N',0:'S'},{1:'F',0:'T'},{1:'J',0:'P'}]
        
        @st.cache
        def reading_data(csv_file):
            data=pd.read_csv(csv_file)
            return data
        data=reading_data('dataset/mbti_modified.csv')

        
        data=data.fillna('')
        posts=(data.content).to_numpy() #feature of the machine learning
        cntizer = CountVectorizer(analyzer="word",max_features=1000,max_df=0.7,min_df=0.1) 
        x_cnt = cntizer.fit_transform(posts)
        n_cnt=x_cnt.toarray()

        tfizer = TfidfTransformer()# condition 4
        x_tfidf =  tfizer.fit_transform(x_cnt).toarray()
        x_data=x_tfidf ## FEATURE OF MACHINE
        #Pre-Processing Stage
        lemmatiser = WordNetLemmatizer()
        
        # Remove these from the posts
        unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
            'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
        unique_type_list = [x.lower() for x in unique_type_list]
        
        b_pers = {'I':1,'E':0,'N':1,'S':0,'F':1,'T':0,'J':1,'P':0}
        b_pers_list = [{1:'I',0:'E'},{1:'N',0:'S'},{1:'F',0:'T'},{1:'J',0:'P'}]
        def translate_personality(personality):
            return[b_pers[i] for i in personality]
        #Transform the binary number to MBTI personality

        def translate_back(personality):
            mbti=""
            for i,j in enumerate(personality):
                mbti=mbti+b_pers_list[i][j]
            return mbti
        nltk.download('stopwords') ###install these modules before processing
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        useless_words = stopwords.words("english") # Remove the stop words for speed 
        def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
            list_posts = []
            total_rows=data.type.value_counts().sum()

            for row in data.iterrows():
                posts = row[1].posts
                #Remove url links 
                temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
                #Remove Non-words - keep only words
                temp = re.sub("[^a-zA-Z]", " ", temp)
                # Remove spaces > 1
                temp = re.sub(' +', ' ', temp).lower()
                #Remove multiple letter repeating words
                temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)
                #Remove stop words
                if remove_stop_words:
                    temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
                else:
                    temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])  
                #Remove MBTI personality words from posts
                if remove_mbti_profiles:
                    for t in unique_type_list:
                        temp = temp.replace(t,"")
                list_posts.append(temp)
            list_posts = np.array(list_posts)
            return list_posts
        
        list_personality = []
        for row in data.iterrows():
            type_labelized = translate_personality(row[1].type) 
            list_personality.append(type_labelized)
        list_personality = np.array(list_personality)
  
    my_posts=st.text_area("Write Something about you...")
    agree=st.button("Predict")
    if agree and len(name)!=0:    
        if len(my_posts)==0:
            pass
        elif len(my_posts)!=0:
            spaces=my_posts.count(" ")
            if spaces<=20:
                st.error(":confused:,Please write more than 20 Words :grey_exclamation:")
            elif spaces>20:
                st.write("Analyzing the Text....")
                mydata = pd.DataFrame(data={'type': ['Identifying..'], 'posts': [my_posts]})
                my_posts = pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)
                my_X_cnt = cntizer.transform(my_posts)
                tf=TfidfTransformer()
                x_feature=tf.fit_transform(my_X_cnt).toarray()
                st.success("Analyzing completed!:smile:")

                personality_type = ["IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)", 
                       "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"]
                result=[]
                st.write("### Please Wait :smile:, Predicting your personality")
                increment=0
                progress=st.progress(increment)
                for l in range(len(personality_type)):
                    y_data = list_personality[:,l]
                    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=7)
                    model = LogisticRegression()
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_feature)
                    result.append(y_pred[0])
                    increment=increment+25
                    progress.progress(int(increment))
                    # st.write(personality_type[l])
                    time.sleep(1)
                st.success("Personality Predicted :blush:")
                mbti=translate_back(result)
                st.write("## :point_right:",name.capitalize(),",your personality is:",mbti,":smile:")

                if mbti[0]=="E":
                    st.write("## EXTROVERT")
                    st.write(""":smiley: You are friendly type of person, you enjoy the social meetup. 
                                         you don't like or need a lot of alone time, you enjoy in between the groups.
                                         You also have many friends. You prefer to talk out problems or questions.
                                         You are also very happy and positive person. You prefer social groups rather than close circle.
                                         Also may be you are talkative person also.""")
                elif mbti[0]=="I":
                    st.write("## INTROVERT")
                    st.write(""":blush: You are shy person. You don't like much to envolve in social meetups or groups.
                                          In most of the time alone, you feel peace in mind and relief. 
                                          Sometimes you feel afraid in public space. To much socializing drains you.
                                          You find tought to share your thought to another. You personally don't like to take a risk.
                                          You work better on your own. You prefer close circle rather than groups.""")

                if mbti[1]=="N":
                    st.write("## INTUITION")
                    st.write(""":sunglasses: You are the person who believe in gut feeling rather than physical reality.
                                             You will always listen to your heart, not your mind in mostly cases. 
                                             You most Probably think about your future rather than your past.
                                             You have a ability for thinking in a different perspective or innovate the new ideas that can give a new direction.
                                             Sometimes your mind drift off during the conversation.""")
                elif mbti[1]=="S":
                    st.write("## SENSING")
                    st.write(""":hushed: You pay more attention to physical reality, like what you see,hear,touch,taste and smell.
                                         You are concerned with what is actual, present,current and real. 
                                         You notice the fact and remember the details that are important to you.
                                         You like to see the practical use of things and learn best when you see how,that you are learning.""")

                if mbti[2]=="F":
                    st.write("## FEELING")
                    st.write(""":wink: You are the person who listen of own heart and emotions when you make important decision.
                                       You are very caring, compassionate and warm.
                                       You are very protective for those whom you care about, it may be your family or other close person.
                                       You make decision, that tend to be based on the well-being of others.
                                       You will mostly more concerned about that the other people think.""")
                elif mbti[2]=="T":
                    st.write("## THINKING")
                    st.write("""🧐 You are the person who mostly depend on the logic and reason in every situation.
                                   You like to analyze pros and cons and make the decision from the mind not from the heart and emotions.
                                   You give value to the truth or facts rather than giving honest,soft or emotional content. 
                                   You are critical thinker and more oriented to your problem solving.""")

                if mbti[3]=="J":
                    st.write("## JUDGING")
                    st.write("""🤓 You are the person who like to be in control or in a structured way.
                                   You always have the backup plan,if first plan get failed.
                                   Once you start the plan or work, you will put all efforts to complete the work.
                                   You set specific goals that you hope to accomplish each day.
                                   """)
                elif mbti[3]=="P":
                    st.write("## PERCEIVING")
                    st.write("""😌 You are very Flexible and very adaptable type of person.
                                          You like to keep your plan minimum. 
                                          You enjoy to starting the task better than to finished it.
                                          You are the random thinker, who prefer to keep their option open.
                                          You are spontaneous and often mix-up several projects at once.""")
    elif((len(my_posts)!=0) and len(name)==0):
        st.error("🙂 Type your name first!")

    


elif nav=="DEVELOPERS":# condition 4
    st_lottie(team_animate,speed=1,loop=True,quality="high")
    st.write("---")
    st.markdown("""
                <h1 align="center">DEVELOPERS PROFILES</h1>
                """,True)
    st.write("---")
    left_col,right_col =st.columns((2,6))
    with left_col:
        image=Image.open('developers/dev_1.png')
        st.image(image,caption="ML_Programmer")
    with right_col:
        st.write("")
        st.write("### Hi, I am Kirti Goel :wave:")
        st.write("###### PROJECT: Human Personality Prediction")
        st.write("###### COURSE: Bachelor of Computer Application")
        st.write("###### ACADEMIC YEAR: 2020-2023")
        st.write("###### COLLEGE: Institute of Information Technology and Management")

    st.write("---")
    left_col,right_col=st.columns((6,2))
    with left_col:
        st.write("")
        st.write("### Hi, I am Mohit Bisht:wave:")
        st.write("###### PROJECT: Human Personality Prediction")
        st.write("###### COURSE: Bachelor of Computer Application")
        st.write("###### ACADEMIC YEAR: 2020-2023")
        st.write("###### COLLEGE: Institute of Information Technology and Management")
    with right_col:
        image=Image.open('developers/dev_2.png')
        st.image(image,caption='ML_Programmer')
    
    st.write("---")
 
    st_lottie(developers_animate,speed=1,loop=True,quality="high")   
    st.write("## Thank You for visiting this Webpage  :blush:")
    
    st.write("---")   
