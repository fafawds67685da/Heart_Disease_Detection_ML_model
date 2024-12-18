import matplotlib.pyplot as m
import numpy as n
import pandas as p

# important note:
# for more information on how was the model build and trained...you can look into
# the Model_training jupiter notebook
#there we trained the model...and had obtained these weights value


#the datset was taken from UCL 
w1=-0.05076649293721135
w2=0.03808140719163982
w3=0.17279908014593054

w4=0.1485099289367947
w5=0.017092663387214987
w6=-0.0385291125928489

w7=0.057469943517028346
w8=-0.11996097220498134
w9=0.05405609014843155

w10=0.28522740179953904
w11=0.06890960749923179
w12=0.28520056008490524

w13=0.17369091099876371

b=0.04223648508086458



def predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,b):
  z1= x1*w1
  z2= x2*w2
  z3= x3*w3

  z4= x4*w4
  z5= x5*w5
  z6= x6*w6

  z7= x7*w7
  z8= x8*w8
  z9= x9*w9

  z10= x10*w10
  z11= x11*w11
  z12= x12*w12

  z13= x13*w13

  return (z1+z2+z3+z4+z5+z6+z7+z8+z9+z10+z11+z12+z13+b)


name=[]
age = []
sex = []
cp = []

trestbps = []
chol = []
fbs = []

restecg = []
thalach = []
exang = []

oldpeak = []
slope = []
ca=[]

thal=[]
num_cases2 = int(input("Enter number of patients\n"))

for i in range(1,num_cases2+1,1):
  a0=input("Enter patient "+str(i)+" name\n")
  print("\n")
  a1=float(input("Enter "+a0+"'s Age\n"))
  print("\n")
  a2=float(input("Enter "+a0+"'s Gender \n Enter 0 for Female \n Enter 1 for Male \n"))
  print("\n")
  a3=float(input("Enter "+a0+"'s Magnitude of chest pain \n Enter 0: Typical angina \n Enter 1: Atypical angina \n Enter 2: Non-anginal pain \n Enter 3: Asymptomatic \n"))
  print("\n")
  a4=float(input("Enter "+a0+"'s Resting blood pressure\n"))
  print("\n")
  a5=float(input("Enter "+a0+"'s Cholestrol Level \n"))
  print("\n")
  a6=float(input("Enter "+a0+"'s Fasting Blood sugar \n Enter 0 if reading is ≤ 120 mg/dl \n Enter 1 if reading is > 120 mg/dl) \n"))
  print("\n") 

  a7=float(input("Enter "+a0+"'s Resting Electrocardiographic Results coded as: \n 0: Normal \n 1: Having ST-T wave abnormality (unspecified) \n 2: Showing probable or definite left ventricular hypertrophy\n"))
  print("\n")
  a8=float(input("Enter "+a0+"'s Maximum Heart Rate Achieved\n"))
  print("\n")
  a9=float(input("Enter "+a0+"'s Exercise Induced Angina  , (chest pain) induced by exercise. \n Enter 0 for none \n Enter 1 if yes \n"))
  print("\n")

  a10=float(input("Enter "+a0+"'s Oldpeak value (amount of ST segment depression observed on an electrocardiogram (ECG) )\n"))
  print("\n")
  a11=float(input("Enter "+a0+"'s  Slope of the Peak Exercise ST Segment coded as: \n 0 for Upsloping \n 1 for Flat \n 2 for Downsloping\n"))
  print("\n")      
  a12=float(input("Enter "+a0+"'s  Number of Major Vessels (0-3) Colored by Fluoroscopy \n"))
  print("\n") 

  a13=float(input("Enter "+a0+"'s results of a thallium stress test coded as: \n 0 for Normal \n 1 for Fixed defect \n 2 for Reversible defect \n"))
  print("\n")

  name.append(a0)

  age.append(a1)
  sex.append(a2)
  cp.append(a3)

  trestbps.append(a4)
  chol.append(a5)
  fbs.append(a6)

  restecg.append(a7)
  thalach.append(a8)
  exang.append(a9)

  oldpeak.append(a10)
  slope.append(a11)
  ca.append(a12)

  thal.append(a13)

def safe_normalize(arr):
    min_val = n.min(arr)
    max_val = n.max(arr)
    if min_val == max_val:
        return n.zeros_like(arr)  
    else:
        return (arr - min_val) / (max_val - min_val)

age2=n.array(age)
sex2=n.array(sex)
cp2=n.array(cp)

trestbps2=n.array(trestbps)
chol2=n.array(chol)
fbs2=n.array(fbs)

restecg2=n.array(restecg)
thalach2=n.array(thalach)
exang2=n.array(exang)

oldpeak2=n.array(oldpeak)
slope2=n.array(slope)
ca2=n.array(ca)

thal2=n.array(thal)



age22 = safe_normalize(age2)
sex22 = safe_normalize(sex2)
cp22 = safe_normalize(cp2)

trestbps22 = safe_normalize(trestbps2)
chol22 = safe_normalize(chol2)
fbs22 = safe_normalize(fbs2)

restecg22 = safe_normalize(restecg2)
thalach22 = safe_normalize(thalach2)
exang22 = safe_normalize(exang2)

oldpeak22 = safe_normalize(oldpeak2)
slope22 = safe_normalize(slope2)
ca22 = safe_normalize(ca2)

thal22 = safe_normalize(thal2)


y_res=predict(age22,sex22,cp22,trestbps22,chol22,fbs22,restecg22,thalach22,exang22,oldpeak22,slope22,ca22,thal22,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,b)
y_res=y_res*100



print("\n⚠️ IMPORTANT NOTE:")
print("In this assessment, a higher score indicates a higher risk of heart problems.")
print("The higher the score, the more attention should be paid to the patient's cardiovascular health.")
print("\n\n")
for i in range(1,num_cases2+1,1):
   print(f"\n🫀 Arterial Diameter Narrowing Index")
   print(" Patient Name: ",name[i-1])
   print(" Result : ",round(y_res[i-1],2)," %\n")

print("\n")
print("\nIMPORTANT DISCLAIMER:")
print("This information is for educational purposes only and does not constitute a medical diagnosis.")
print("For accurate health information and personalized advice, please consult with a qualified healthcare professional.\n")
print("I hope this information is helpful!")
