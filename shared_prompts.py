from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

gpt35 = ChatOpenAI(temperature=0.7, max_tokens=32)

# define task evaluation prompt templates
tasks = {
    "summary" : {
        "written" : PromptTemplate.from_template(
            "You are giving feedback on the quality of a summary."
            "\n=Instruction=\n{instruction}"
            "\n=Article=\n{input_text}\n=Summary=\n{output_text}"
            "\n=Feedback=\n "), 
        "integer" : PromptTemplate.from_template(
            "You are giving a score based on the quality of a summary."
            "\n=Instruction=\n{instruction}"
            "\n=Article=\n{input_text}\n=Summary=\n{output_text}"
            "\n=Score=\n "), 
        "lettergrade" : PromptTemplate.from_template(
            "You are giving a grade based on the quality of a summary."
            "\n=Instruction=\n{instruction}"
            "\n=Article=\n{input_text}\n=Summary=\n{output_text}"
            "\n=Grade=\n "), 
        "abtest" : PromptTemplate.from_template(
            "You are choosing between two summaries based on how well they summarize an article."
            "Choose the better summary on the basis of relevance, importance, and accuracy. Only respond with '0' or '1', nothing else."
            "\n=Article=\n{input_text}\n=Summary 0=\n{output_text_0}\n=Summary 1=\n{output_text_1}"
            "\n=Choice=\n "), 
    },
    "qa" : {
        "written" : PromptTemplate.from_template(
            "You are giving feedback on the correctness of an answer to a question."
            "\n=Instruction=\n{instruction}"
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Attempted Answer=\n{output_text}"
            "\n=Feedback=\n "), 
        "integer" : PromptTemplate.from_template(
            "You are giving a score based on the correctness of an answer to a question."
            "\n=Instruction=\n{instruction}"
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Attempted Answer=\n{output_text}"
            "\n=Score=\n "), 
        "lettergrade" : PromptTemplate.from_template(
            "You are giving a grade based on the correctness of an answer to a question."
            "\n=Instruction=\n{instruction}"
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Attempted Answer=\n{output_text}"
            "\n=Grade=\n "), 
        "abtest" : PromptTemplate.from_template(
            "You are choosing between two answers based on how well they answer a question."
            "\n=Instruction=\n{instruction}"
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Answer 0=\n{output_text_0}\n=Answer 1=\n{output_text_1}"
            "\n=Choice=\n "), 
    }
}

# define baseline instructions for task evaluation
baseline_instructions = {
    "summary" : {
        "written" : "Give one sentence of feedback on the summary with respect to its relevance, importance, and accuracy. Be extremely strict and critical when it comes to relevance, importance, and accuracy.",
        "integer" : "Give a score 0-10 to this summary. 0 means irrelevant, 5 means errors, 10 means no possible improvements. Be extremely harsh, strict, and critical with respect to its relevance, importance, and accuracy. ALWAYS start your feedback by specifying the score, followed by an explanation of why you have given this score. For instance, 'Score: 5. The summary is only partially correct.'",
        "lettergrade" : "Give a letter grade (A+ through F) to this summary. F means irrelevant, C means errors, A+ means no possible improvements. Be extremely harsh, strict, and critical with respect to its relevance, importance, and accuracy. ALWAYS start your feedback by specifying the grade, followed by an explanation of why you have given this grade. For instance, 'Grade: B-. The summary is only partially correct.'",
        "abtest" : "Choose the better summary on the basis of relevance, importance, and accuracy. Only respond with '0' or '1', nothing else."
    },
    "qa" : {
        "written" : "Give one sentence of feedback on the answer with respect to its correctness. Be extremely strict and critical when it comes to correctness.",
        "integer" : "Give a score 0-10 to this answer. 0 means irrelevant, 5 means errors, 10 means no possible improvements. Be extremely harsh, strict, and critical with respect to correctness. ALWAYS start your feedback by specifying the score, followed by an explanation of why you have given this score. For instance, 'Score: 5. The answer is only partially correct.'",
        "lettergrade" : "Give a letter grade (A+ through F) to this answer. F means irrelevant, C means errors, A+ means no possible improvements. Be extremely harsh, strict, and critical with respect to correctness. ALWAYS start your feedback by specifying the grade, followed by an explanation of why you have given this grade. For instance, 'Grade: B-. The answer is only partially correct.'",
        "abtest" : "Choose the better answer on the basis of correctness. Only respond with '0' or '1', nothing else."
    }
}

gpt35_summary_lettergrade_instruction = "Give a letter grade (A+ through F) to the following summary based on its relevance, importance, and accuracy. F means irrelevant, C indicates the presence of errors or misinformation, and A+ signifies that there are no possible improvements. Be strict in your judgement, but do not over-penalize minor errors. Only respond with a letter grade, nothing else."

LLAMA_BINARY_FEWSHOT = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. You are evaluating the quality of a summary. Is the summary faithful, coherent, and relevant? Only respond with 'yes' or 'no', nothing else.
<</SYS>>
\n=Article=\n
City are fourth in the league following defeat at Chelsea on Wednesday, all but ending hopes of a league title in the Catalan\'s debut campaign in England. "In future I will be better - definitely," said the former Barcelona and Bayern Munich coach. "This season has been a massive lesson for me." He added: "We have a lot of beautiful things to fight for and to qualify for the Champions League will be a huge success. But we have to be honest with ourselves. We were not good enough to compete for the Premier League [title]." Defeat at Stamford Bridge left City just four points ahead of Arsenal and Manchester United, both of whom have a game in hand. With the FA Cup being City\'s only realistic chance of winning a trophy this term, it is likely to be the worst season of Guardiola\'s managerial career. The 46-year-old has never gone a single season without winning a trophy - he has won the title in six out of seven attempts and his sides have always reached the last four of the Champions League. Guardiola has also responded to questions about a row between City and Chelsea coaching staff at Stamford Bridge following the defeat on Wednesday night. There have been conflicting claims about the precise nature of the row, but stewards were needed to calm the situation down after Chelsea\'s 2-1 win. The incident centred around a disagreement between Chelsea fitness coach Paolo Bertelli and Manchester City masseur Mark Sertori, both of whom speak Italian, as the Premier League leaders celebrated their victory. Guardiola said: "We are so polite in our defeats and we are so polite when we win. When we win, normally we celebrate a little bit, then we go to the locker room. Chelsea manager Antonio Conte was not involved and has played the incident down. He said: "Respect is the most important thing in football."
\n=Summary=\n
The article discusses City's recent defeat at Chelsea and how it has all but ended their hopes of winning the league title, with Guardiola admitting that he has learned a lot from the season.
\n=Answer=\n
yes
\n=Article=\n
Research carried out by the Fostering Network suggests almost half of fostered young people are already living with their third foster family since going into care. The group has warned that 750 more foster carers are "urgently" needed to meet the demands of the care system. It urged people to "open their hearts and homes" to vulnerable youngsters. Currently, more than 5,500 children are in foster care in Scotland, living with 4,400 families and carers. The Fostering Network surveyed 250 children, teenagers and foster carers across Scotland and discovered that many young people had failed to find stability. Almost half were already living with their third family, a quarter were with their fourth family and about 20 were living with their 10th family since going into care. There was a particular need for homes to be found for vulnerable teenagers, siblings and disabled children, the study found. Carla, 23, was taken into care at the age of 12 and had eight foster homes before moving in with the Randalls. "Looking back now I realised that the Randalls saved my life," she said. "I never understood the extent of the neglect and abuse I had endured until I came to live with a \'normal\' loving family. "They were just always themselves, the smallest details meant so much to me. "They nurtured a young, angry, untrusting teenager to become a positive, empathetic and successful young woman." The Fostering Network said instability had a detrimental effect on the child\'s education and wellbeing, while finding a stable foster carer from the outset could lead to improved relationships and a happier childhood. Sara Lurie, director of the Fostering Network Scotland, said: "As each year passes, we see more and more children coming into care. "We need people who can open their heart, and their homes, to vulnerable children and young people and use their skills to help support them to reach their full potential. "In particular we need people who have the skills, patience and passion to look after teenagers who may have had a really tough time and be facing some real challenges, and to offer them love, stability and security. "A good foster carer will believe in the ambition of the children in their care in the same way they\'d believe in the ambition of their biological family members." Apologies for the delay, see below as requested. A Scottish government spokeswoman said: "Giving young people security is paramount and we have done a great deal of work with our partners across local government and the third sector to improve how we intervene early when there is a problem within families to find appropriate solutions quickly. "We have also expanded the age at which young people can remain in foster care as part of the continuing care provisions and the support available when they transition into independent living."
\n=Summary=\n
'What is the main idea of the article?\n\nWhat is the main idea of the article?\n\nWhat is the main idea of the article?\n\nWhat is the main idea of the article?\n\nWhat is the main idea of the article?\n\nWhat is the main idea of the
\n=Answer=\n
no
\n=Article=\n
Asif Kahn, who worked at Oldknow Academy in Birmingham, had faced allegations of misconduct. A National College of Teaching and Leadership panel heard the allegations against him in November, although Mr Khan did not appear at the hearing. The Professional Conduct Panel has said it did not find the case proven. Oldknow Academy was one of several schools investigated amid claims of a Muslim hardliners\' plot to control them; known as the Trojan Horse affair. More on this and other stories from Birmingham and the Black Country Mr Khan had been accused of agreeing "to the inclusion of an undue amount of religious influence in the education of pupils" at Oldknow, on or before 31 July 2014. The accusations had included telling some male pupils to change for PE in a cupboard so they would not show their thighs, banning children singing during a production of The Wizard of Oz and turning his back on a woman as she offered to shake his hand. He was also accused of sharing his personal beliefs with the children, for example telling the children they were not allowed pet dogs as they were Muslim. A fellow teacher, former-acting head teacher Jahangir Akbar, was banned from teaching indefinitely in January, although he can apply to have his ban set aside in five years time.
\n=Summary=\n
Oldknow Academy was one of several schools investigated amid claims of a Muslim hardliners' plot to control them; known as the Trojan Horse affair.
\n=Answer=\n
yes
\n=Article=\n
Media playback is unsupported on your device 31 July 2015 Last updated at 18:45 BST They include those who have travelled north across the border. BBC Ireland Correspondent Andy Martin reports.
\n=Summary=\n
The article is about the situation in Calais, France, where migrants are trying to get to the UK.
\n=Answer=\n
no
\n=Article=\n{input_text}\n=Summary=\n{output_text}
\n=Answer=\n [/INST]
"""

GPT35_BASE_ZERO_SHOT = PromptTemplate.from_template(
"""
"You are evaluating the quality of a summary. Is this summary faithful, coherent, and relevant? Only respond with 'yes' or 'no', nothing else."
"\n=Instruction=\n{instruction}"
"\n=Article=\n{input_text}\n=Summary=\n{output_text}"
"\n=Answer=\n "
"""
)

GPT35_LETTER_FEW_SHOT = PromptTemplate.from_template(
"""
Give a letter grade (A+ through F) to the following summary based on its relevance, importance, and accuracy. F means irrelevant, C indicates the presence of errors or misinformation, and A+ signifies that there are no possible improvements. Be strict in your judgement, but do not over-penalize minor errors. Only respond with a letter grade, nothing else.
\n=Article=\n
City are fourth in the league following defeat at Chelsea on Wednesday, all but ending hopes of a league title in the Catalan\'s debut campaign in England. "In future I will be better - definitely," said the former Barcelona and Bayern Munich coach. "This season has been a massive lesson for me." He added: "We have a lot of beautiful things to fight for and to qualify for the Champions League will be a huge success. But we have to be honest with ourselves. We were not good enough to compete for the Premier League [title]." Defeat at Stamford Bridge left City just four points ahead of Arsenal and Manchester United, both of whom have a game in hand. With the FA Cup being City\'s only realistic chance of winning a trophy this term, it is likely to be the worst season of Guardiola\'s managerial career. The 46-year-old has never gone a single season without winning a trophy - he has won the title in six out of seven attempts and his sides have always reached the last four of the Champions League. Guardiola has also responded to questions about a row between City and Chelsea coaching staff at Stamford Bridge following the defeat on Wednesday night. There have been conflicting claims about the precise nature of the row, but stewards were needed to calm the situation down after Chelsea\'s 2-1 win. The incident centred around a disagreement between Chelsea fitness coach Paolo Bertelli and Manchester City masseur Mark Sertori, both of whom speak Italian, as the Premier League leaders celebrated their victory. Guardiola said: "We are so polite in our defeats and we are so polite when we win. When we win, normally we celebrate a little bit, then we go to the locker room. Chelsea manager Antonio Conte was not involved and has played the incident down. He said: "Respect is the most important thing in football."
\n=Summary=\n
The article discusses City's recent defeat at Chelsea and how it has all but ended their hopes of winning the league title, with Guardiola admitting that he has learned a lot from the season.
\n=Answer=\n
A
\n=Article=\n
Research carried out by the Fostering Network suggests almost half of fostered young people are already living with their third foster family since going into care. The group has warned that 750 more foster carers are "urgently" needed to meet the demands of the care system. It urged people to "open their hearts and homes" to vulnerable youngsters. Currently, more than 5,500 children are in foster care in Scotland, living with 4,400 families and carers. The Fostering Network surveyed 250 children, teenagers and foster carers across Scotland and discovered that many young people had failed to find stability. Almost half were already living with their third family, a quarter were with their fourth family and about 20 were living with their 10th family since going into care. There was a particular need for homes to be found for vulnerable teenagers, siblings and disabled children, the study found. Carla, 23, was taken into care at the age of 12 and had eight foster homes before moving in with the Randalls. "Looking back now I realised that the Randalls saved my life," she said. "I never understood the extent of the neglect and abuse I had endured until I came to live with a \'normal\' loving family. "They were just always themselves, the smallest details meant so much to me. "They nurtured a young, angry, untrusting teenager to become a positive, empathetic and successful young woman." The Fostering Network said instability had a detrimental effect on the child\'s education and wellbeing, while finding a stable foster carer from the outset could lead to improved relationships and a happier childhood. Sara Lurie, director of the Fostering Network Scotland, said: "As each year passes, we see more and more children coming into care. "We need people who can open their heart, and their homes, to vulnerable children and young people and use their skills to help support them to reach their full potential. "In particular we need people who have the skills, patience and passion to look after teenagers who may have had a really tough time and be facing some real challenges, and to offer them love, stability and security. "A good foster carer will believe in the ambition of the children in their care in the same way they\'d believe in the ambition of their biological family members." Apologies for the delay, see below as requested. A Scottish government spokeswoman said: "Giving young people security is paramount and we have done a great deal of work with our partners across local government and the third sector to improve how we intervene early when there is a problem within families to find appropriate solutions quickly. "We have also expanded the age at which young people can remain in foster care as part of the continuing care provisions and the support available when they transition into independent living."
\n=Summary=\n
'What is the main idea of the article?\n\nWhat is the main idea of the article?\n\nWhat is the main idea of the article?\n\nWhat is the main idea of the article?\n\nWhat is the main idea of the article?\n\nWhat is the main idea of the
\n=Answer=\n
F
\n=Article=\n
Asif Kahn, who worked at Oldknow Academy in Birmingham, had faced allegations of misconduct. A National College of Teaching and Leadership panel heard the allegations against him in November, although Mr Khan did not appear at the hearing. The Professional Conduct Panel has said it did not find the case proven. Oldknow Academy was one of several schools investigated amid claims of a Muslim hardliners\' plot to control them; known as the Trojan Horse affair. More on this and other stories from Birmingham and the Black Country Mr Khan had been accused of agreeing "to the inclusion of an undue amount of religious influence in the education of pupils" at Oldknow, on or before 31 July 2014. The accusations had included telling some male pupils to change for PE in a cupboard so they would not show their thighs, banning children singing during a production of The Wizard of Oz and turning his back on a woman as she offered to shake his hand. He was also accused of sharing his personal beliefs with the children, for example telling the children they were not allowed pet dogs as they were Muslim. A fellow teacher, former-acting head teacher Jahangir Akbar, was banned from teaching indefinitely in January, although he can apply to have his ban set aside in five years time.
\n=Summary=\n
Oldknow Academy was one of several schools investigated amid claims of a Muslim hardliners' plot to control them; known as the Trojan Horse affair.
\n=Answer=\n
B-
\n=Article=\n
Media playback is unsupported on your device 31 July 2015 Last updated at 18:45 BST They include those who have travelled north across the border. BBC Ireland Correspondent Andy Martin reports.
\n=Summary=\n
The article is about the situation in Calais, France, where migrants are trying to get to the UK.
\n=Answer=\n
C
\n=Article=\n{input_text}\n=Summary=\n{output_text}
\n=Answer=\n
"""
)