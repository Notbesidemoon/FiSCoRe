sciq_shot = """
Question: A diet rich in calcium and what vitamin may reduce the risk of osteoporosis and related bone fractures? 
<answer> Vitamin D </answer>
<confidence> sure </confidence>

Question: Who is the first man to walk on the Mars?
<answer> Mark Watney </answer>
<confidence> unsure </confidence>

Question: When did the World War III start?
<answer> 2399 </answer>
<confidence> unsure </confidence>

Question: In the absence of air resistance, all falling objects accelerate at the same rate due to what force?
<answer> gravity </answer>
<confidence> sure </confidence>

Question: Where were the 2084 Summer Olympic Games held?
<answer> Beijing </answer>
<confidence> unsure </confidence>

Question: What is the opposite of melting?
<answer> freezing </answer>
<confidence> sure </confidence>

"""

triviaqa_shot = """
Question: Which American-born Sinclair won the Nobel Prize for Literature in 1930? 
<answer> Sinclair Lewis </answer>
<confidence> sure </confidence>

Question: How did the manufacturers of Old Grand-Dad bourbon get away with producing their whisky during Prohibition?
<answer> The marked the bottles for medicinal purposes </answer>
<confidence> unsure </confidence>

Question: In The Banana Splits what sort of animal was Snorky? 
<answer> Elephant </answer>
<confidence> sure </confidence>

Question: Who is the first man to walk on the Mars?
<answer> Mark Watney </answer>
<confidence> unsure </confidence>

Question: Where were the 2084 Summer Olympic Games held?
<answer> New York </answer>
<confidence> unsure </confidence>

Question: What is the capital of Kenya?
<answer> Nairobi </answer>
<confidence> sure </confidence>

"""

nq_shot = """
Question: What is the capital of France? 
<answer> Paris </answer>
<confidence> sure </confidence>

Question: Who is the first man to walk on the Mars?
<answer> Mark Watney </answer>
<confidence> unsure </confidence>

Question: When did the World War III start?
<answer> 2399 </answer>
<confidence> unsure </confidence>

Question: Who discovered the law of gravity? 
<answer> Isaac Newton </answer>
<confidence> sure </confidence>

Question: Where were the 2048 Summer Olympic Games held?
<answer> Beijing </answer>
<confidence> unsure </confidence>

Question: Which country is Europe's largest silk producer?
<answer> Italy </answer>
<confidence> sure </confidence>

"""

pararel_shot = """
Question: What is the capital of India?
<answer> New Delhi </answer>
<confidence> sure </confidence>

Question: Who is the president of the fictional country of Northland? 
<answer> Alex Smith </answer>
<confidence> unsure </confidence>

Question: Who produced the Japanese battleship Mikasa? 
<answer> Vickers </answer>
<confidence> sure </confidence>

Question: What is the fictional city of Lakeview's twin city? 
<answer> Riverton </answer>
<confidence> unsure </confidence>

Question: When did the World War III start?
<answer> 2399 </answer>
<confidence> unsure </confidence>

Question: Who discovered the law of gravity?
<answer> Isaac Newton </answer>
<confidence> sure </confidence>

"""

shot_dict = {
    "sciq": sciq_shot,
    "triviaqa": triviaqa_shot,
    "nq": nq_shot,
    "pararel": pararel_shot,
}