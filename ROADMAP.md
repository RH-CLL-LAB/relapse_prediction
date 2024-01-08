### Description of the project     
- What are we investigating? What is the project about? What do we want to know?  
	- We are investigating whether it is possible to predict bad outcomes (primarily relapse) within two years of first line treatment. We start out by doing this only with certain lymphoma patients where the impact is largest, but part of the project will be centered around investigating whether including information from patients with other types of lymphoma or maybe even other blood cancers can increase predictive performance.
### Model output 
- What type of task should the model perform? (classification, clustering, etc.) 
	- Classification of what patients are at high risk of bad outcomes
- Is the output categorical, continuous?
	- One could have different levels of bad outcomes, but the output is more or less categorical
- What makes most sense clinically?
	- From my non-clinical standpoint, what we really want to know is whether or not the patient is at a high risk of developing bad outcomes in the future. Relapse is but one of such outcomes - we are interested in whether we should be more aggressive from the start of treatment than we otherwise would be. 
- What is possible from a technical standpoint?
	- We start out by making non-sequential models, but ideally we would move on to some basic of sequential models. There might not be enough data to do this for this cohort but would be interesting to try. This might be outside the scope of this project though, but should be considered. 
### Clinical relevance 
- How far can the lookback window be while still being clinically relevant? 
	- Very open question. It is not completely clear why one would not just use all the available data from the past to predict the future outcome, but I would be open to discussing this if that is not a fair point. 
- Is the model useful if it works?
	- Yes, very! We would be able to have a better screening of patients before first treatment which would help indicate whether some patients need to be considered for more aggressive treatment. That might just save some lives. 
### Variable selection     
- What kind of variables/signal is the model supposed to capture?  
	- The model is supposed to capture some of the previous medical history from the patient while overemphasizing the last couple of weeks of data before treatment. Usually the cohort is treated within 15 days of diagnosis. The numbers given in the period from diagnosis to treatment are probably the most informative features.
- What variables would be good candidates to include for analysis? (obvious biomarkers of interest, obvious demographic markers, etc.)
	- More or less all RKKPs columns (excluding columns with future leakage) are very valuable. Other than those columns, we should look at pathology codes, diagnosis codes, lab values
- What variables need to be excluded? (Information leakage, contaminants of the dataset, confounding variables)  
- Is there a strong bias between groups of interest in the data? (different diseases, different regions, etc.) 
- How does the variable “behave”? How often is the variable recorded? In what context is the variable typically recorded? 
- What is the coverage of the data?  
- Is time important to model explicitly? 
- What defines the relevant cohort?  
- What are known risk factors in terms of demographics? (Age, region, etc.)  

### Model specification     
- Which other models will the model be compared to? What’s the current state-of-the-art?  
- What objective function to optimize for? (sensitivity, specificity, etc.) 
- What is the minimum performance that would be considered a success? (better than baseline, PR-AUC = 0.7)
