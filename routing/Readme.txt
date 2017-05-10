The data for Us and Poland market

There are two types of files:
1. date-question-listusers.csv
The first column is the question_id, then followed by top 100 potential answerers.
The potential answerers is ranked in decreasing order. We should start with small number first (i.e, 10 potential users).
I eleminated the prefix (us, pl) as Alistair suggestions.

2. date-user-listquestions.csv
The first column is the user_id
The second column is #of questions he answered during last 90 days (represent his activeness)
Then, the 10 potential questions he might answers (in decreasing order)
