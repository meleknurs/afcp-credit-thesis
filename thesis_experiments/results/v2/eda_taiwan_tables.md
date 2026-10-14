n = 30000, default rate = 0.2212, missing values = 0

Raw code frequencies (before recoding):
  SEX: 1:11888, 2:18112
  EDUCATION: 0:14, 1:10585, 2:14030, 3:4917, 4:123, 5:280, 6:51
  MARRIAGE: 0:54, 1:13659, 2:15964, 3:323
  AGE: min 21, median 34.0, max 79
  PAY_0 codes: -2:2759, -1:5686, 0:14737, 1:3688, 2:2667, 3:322, 4:76, 5:26, 6:11, 7:9, 8:19

Default rate by group (after recoding):
  sex_female: male: n=11888, default=0.242; female: n=18112, default=0.208
  marriage_grp: married: n=13659, default=0.235; single: n=15964, default=0.209; others: n=377, default=0.236
  education_grp: graduate: n=10585, default=0.192; university: n=14030, default=0.237; high school: n=4917, default=0.252; others: n=468, default=0.071
  age_lt25: >=25: n=27315, default=0.216; <25: n=2685, default=0.272
  age_3way: <=35: n=16809, default=0.214; 36-55: n=12347, default=0.228; >55: n=844, default=0.265

Default rate by age decade:
  20-29: n=9618, default=0.228
  30-39: n=11238, default=0.203
  40-49: n=6464, default=0.230
  50-59: n=2341, default=0.249
  60+: n=339, default=0.283

Intersection sex x age_3way (n, default rate):
            count              mean              
age_3way        0     1    2      0      1      2
sex_female                                       
0            6151  5298  439  0.230  0.255  0.237
1           10658  7049  405  0.205  0.207  0.296