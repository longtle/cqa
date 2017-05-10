Author: Long T.Le
02/24/2016 

This code is written for research prototype. 

I put my sample code and sample data 
- Quality.py: this is the main classification problem
		InputFile: us-answer-feat-100.csv contains the 200k questions with features. You can add or remove the features.
		OutputFile: us-answer-pred-100k.csv contain the prediction and score
		
		
Most of features must be calculated before applying classification. 
Section 3 in paper.pdf explains the features 
Some features might be not straight forward to calculate, so I put here:
- GetNetSimileValues.py: reads friendship file and returns community features
		InputFiles: 	us-friendship.csv
		OutputFiles: 	us-ids: the id file
						us-netsimile.csv are the outputs of community features
- Readable features:take a look at labelARI in Quality.py. In the experiment, readability features are week predictors. So you might
drop them without worrying the drop in the accuracy.