# myocardial-infraction-diagnosis-model

Python program using neural networks to detect myocardial infraction using 59 features/symptomps.

## Features list:

### General
1. 	Age (integer number)
2. 	Sex (0 - female, 1 - male)
### Pain
3. 	Pain location:
```
Retrosternal (1)
Left precordial (2)
Right precordial (3)
Left lateral chest (4)
Right lateral chest (5)
Abdominal (6)
Back (7)
Other (8)
```
4. Chest pain radiation:
```
Neck (1)
Jaw (2)
Left shoulder (3)
Left arm (4)
Right shoulder (5)
Back (6)
Abdomen (7)
Other (8)
```
5. Pain character:
```
Constant (1)
Episodic (2)
Episodic than constant (3)
Constant than episodic (4)
Dull/pressure (5)
Sharp (6)
Burning (7)
Pleuritic (8)
```
6. 	Onset of pain:
```
With exertion (1)
At rest (2)
Asleep (3)
```
7. Number of hours since onset (integer number)
8. Duration of the last episode
```
Less than 5 minutes (1)
5 - 30 min (2)
30 - 60 min (3)
1 - 6 hours (4)
6 - 12 hours (5)
Greater than 12 hours (6)
```
### Associated symptoms
9. 	Nausea (0 -none, 1 - present)
10. Diaphoresis (0 -none, 1 - present)
11. Palpitations (0 -none, 1 - present)
12. Dyspnea (0 -none, 1 - present)
13. Dizziness/syncope (0 -none, 1 - present)
14. Burping (0 -none, 1 - present)
15. Palliative factors
```
None (1)
Relieved by nitroglycerin within 5 min (2)
Relieved by nitroglycerin after more than 5 min (3)
Antacids (4)
Analgesia except morphine (5)
Morphine (6)
```
### History of similar pain
16. Prior chest pain of this type (0 -none, 1 - present)
17. Physician consulted for prior pain (0 -none, 1 - present)
18. Prior pain related to heart (0 -none, 1 - present) 
19. Prior pain due to MI (0 -none, 1 - present)
20. Prior pain due to angina prectoris (0 -none, 1 - present)
### Past medical history
21. Prior MI (0 -none, 1 - present)
22. Prior angina prectoris (0 -none, 1 - present)
23. Prior atypical chest pain (0 -none, 1 - present)
24. Congestive heart failure (0 -none, 1 - present)
25. Peripheral vascular disease (0 -none, 1 - present)
26. Hiatal hernia (0 -none, 1 - present)
27. Hypertension (0 -none, 1 - present)
28. Diabetes (0 -none, 1 - present)
29. Smoker (0 -none, 1 - present)
### Current medication usage
30. Diuretics (0 -none, 1 - present)
31. Nitrates (0 -none, 1 - present)
32. Beta blockers (0 -none, 1 - present)
33. Digitalis (0 -none, 1 - present)
34. Nonsteroidal anti-inflammatory (0 -none, 1 - present)
35. Antacids/H2 blockers (0 -none, 1 - present)
### Physical examinations
36. Systolic blood pressure (integer)
37. Diastolic blood pressure (integer)
38. Heart rate (integer)
39. Respiration rate (integer)
40. Rales (0 -none, 1 - present)
41. Cyanosis (0 -none, 1 - present)
42. Pallor (0 -none, 1 - present)
43. Systolic murmur (0 -none, 1 - present)
44. Diastolic murmur (0 -none, 1 - present)
45. Oedema (0 -none, 1 - present)
46. S3 gallop (0 -none, 1 - present)
47. S4 gallop (0 -none, 1 - present)
48. Chest wall tenderness (0 -none, 1 - present)
49. Diaphoresis (0 -none, 1 - present)
### ECG examination
50. New Q wave (0 -none, 1 - present)
51. Any Q wave (0 -none, 1 - present)
52. New ST segment elevation (0 -none, 1 - present)
53. Any ST segment elevation (0 -none, 1 - present)
54. New ST segment depression (0 -none, 1 - present)
55. Any ST segment depression (0 -none, 1 - present)
56. New T wave inversion (0 -none, 1 - present)
57. Any T wave inversion (0 -none, 1 - present)
58. New intraventricular conduction defect (0 -none, 1 - present)
59. Any intraventricular conduction defect (0 -none, 1 - present)


