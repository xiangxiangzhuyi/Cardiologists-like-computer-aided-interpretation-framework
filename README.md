# Cardiologists-like-computer-aided-interpretation-framework


Arrhythmias can increase the risk of complications such as stroke or heart failure, leading to cardiac arrest, shock, and sudden death. The computer-aided interpretation system of ECG is an important tool for providing decision support to cardiologists in arrhythmias diagnosis. Based on the performance of computer-aided interpretation systems, two kinds of arrhythmias are defined: aggressive arrhythmias, which are easy to identify, and vulnerable arrhythmias, which are difficult to identify. In those systems, the bullying from aggressive arrhythmias against vulnerable arrhythmias makes the patients with vulnerable arrhythmias likely to be underdiagnosed. Inspired by the diagnostic thinking of cardiologists, a method for arrhythmia diagnosis that combined morphological-characteristics-based waveforms clustering and Bayesian theory was proposed in this study. Our method was validated in the GDPH ECG-Arrhythmia Dataset. Compared with alternative methods, our method not only achieved comparable performance on aggressive arrhythmias but also protected vulnerable arrhythmias from being bullied by aggressive arrhythmias. With increasing bullying from aggressive arrhythmias, our method could still make a fine diagnosis of vulnerable arrhythmias. Moreover, the characteristics of the maximum cluster were consistent with the diagnostic criteria of arrhythmias, which indicates that our method has certain interpretability.


## 1. The related code of the cardiologists-like computer aided interpretation framework of ECG is in the folder "code".


## 2. Some ECG examples are shared in the folder "ECG_data". These data have been anonymized. you can use numpy.load() to access the data.

sinoatrial block (SA block.npy)

sinus bradycardia	(SB.npy)

sinus tachycardia	(ST.npy)

sinus arrhythmia	(SA.npy)

atrioventricular block	(AV block.npy)

junction tachycardia	(JT.npy)

junction escape	(JE.npy)

junction escape rhythm	(JER.npy)

premature junctional contraction	(PJC.npy）

atrial fibrillation	(AF.npy）

atrial tachycardia	(AT.npy）

premature atrial contraction	(PAC.npy）

intraventricular block	(IV block.npy）

ventricular tachycardia	(VT.npy）

ventricular escape	(VE.npy）

premature ventricular contraction	(PVC.npy）

atrial flutter	(AFL.npy）

normal (Normal.npy)

## 3. 100 arrhythmia ECG images are shared in the folder "ECG_image".
![0](https://user-images.githubusercontent.com/15710573/204076687-52639870-8ded-495f-b777-70214b48e2a4.png)


## For more information, wait for our paper "Cardiologists-like computer-aided interpretation framework for protecting vulnerable arrhythmias from the bullying of the aggressive" to be public.
