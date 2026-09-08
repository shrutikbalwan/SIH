# NEGATIVE DATASET EXPANSION REPORT

## 1. Global Statistics
- **Total Negative Clips Before:** 6000
- **Total Negative Clips After:** 18000
- **Speech Negative Clips Before:** 4000
- **Speech Negative Clips After:** 16000
- **Ambient Negative Clips:** 2000
- **Unique Negative Speech Speakers Before:** 28
- **Unique Negative Speech Speakers After:** 148

## 2. Split Specifics (New Data Only)
### TRAIN
- **Number of Speakers:** 90
- **Number of Clips:** 9000
- **Clips per speaker (Min/Median/Max):** 100 / 100.0 / 100
- **Speaker IDs:** 1089, 1188, 121, 1255, 1462, 1585, 1630, 1650, 1651, 1673, 1686, 1688, 1701, 174, 1919, 1995, 1998, 2086, 2300, 2414, 2506, 260, 2609, 2803, 2830, 2961, 3000, 3005, 3080, 3081, 3170, 3528, 3538, 3663, 3752, 3915, 3997, 4153, 4198, 4294, 4515, 4570, 4572, 4970, 4992, 5142, 533, 5338, 5442, 5484, 5536, 5543, 5639, 5683, 5764, 5849, 6070, 61, 6123, 6128, 6267, 6295, 6313, 6319, 6345, 6432, 6455, 6599, 672, 6841, 6930, 6938, 700, 7018, 7105, 7127, 7176, 7697, 7729, 777, 7976, 8131, 8173, 8224, 8288, 84, 8455, 8461, 8555, 8842

### VALIDATION
- **Number of Speakers:** 15
- **Number of Clips:** 1500
- **Clips per speaker (Min/Median/Max):** 100 / 100.0 / 100
- **Speaker IDs:** 116, 1284, 1580, 1988, 2277, 2902, 4323, 4350, 4446, 652, 6829, 8188, 8297, 8463, 908

### UNSEEN_TEST
- **Number of Speakers:** 15
- **Number of Clips:** 1500
- **Clips per speaker (Min/Median/Max):** 100 / 100.0 / 100
- **Speaker IDs:** 1221, 1320, 2033, 2094, 2412, 2428, 251, 3331, 3536, 3570, 3576, 3660, 367, 3764, 5895

## 3. Metadata
- **New LibriSpeech Subsets Used:** dev-other, test-clean, dev-clean, test-other
- **Rejected Silent Windows:** 4
- **Leakage Checks:** PASSED (Zero overlap between all 4 speaker pools, zero duplicate paths).
- **Ready for V2.3:** YES.
