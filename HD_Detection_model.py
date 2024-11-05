import matplotlib.pyplot as m
import numpy as n
import pandas as p

print("Please wait for few moments , while the model evalutes the provided random dataset !")
print("This can take a few seconds \n")

num_cases = 100

#1 ( bada wala bp)
bp_systolic = n.random.randint(90, 180, num_cases)
#2 ( chhota wala bp)
bp_diastolic = n.random.randint(60, 120, num_cases)
#3 (pulse)
heart_rate = n.random.randint(50, 120, num_cases)
#4 (cholestrol level)
cholesterol_level = n.random.randint(150, 300, num_cases)
#5 (glucose level)
glucose_level = n.random.randint(70, 200, num_cases)
#6 (BMI)
BMI = n.round(n.random.uniform(18, 40, num_cases), 2)
#7 (physical activity)
physical_activity = n.random.randint(1, 11, num_cases)
#8 Oxygen levels
oxygen_saturation = n.random.randint(90, 100, num_cases)

# 9 Physical Symptom ( chest pain , shortness of breath , palpitations)
#       Enter 1 for chest pain
#       Enter 2 for shortness of breath
#       Enter 3 for palpitations
#       Enter 4 for chest pain and shortness of breath
#       Enter 5 for shortness of breath and palpitations
#       Enter 6 for chest pain and palpitations
#       Enter 7 for all of them

symptoms = n.random.randint(1, 8, num_cases)

# 10 result probqability
heart_complexity_probability = n.round(n.random.uniform(0, 100, num_cases), 2)

#11 age
age=n.random.randint(18,100,num_cases)

#12 gender
# enter 1 for male 2 for female and 3 for others
gender=n.random.randint(1,4,num_cases)



bp1=(bp_systolic - n.min(bp_systolic)) / (n.max(bp_systolic) - n.min(bp_systolic))
bp2=(bp_diastolic - n.min(bp_diastolic)) / (n.max(bp_diastolic) - n.min(bp_diastolic))
hr=(heart_rate - n.min(heart_rate)) / (n.max(heart_rate) - n.min(heart_rate))
col=(cholesterol_level - n.min(cholesterol_level)) / (n.max(cholesterol_level) - n.min(cholesterol_level))

gl=(glucose_level - n.min(glucose_level)) / (n.max(glucose_level) - n.min(glucose_level))
bmi=(BMI - n.min(BMI)) / (n.max(BMI) - n.min(BMI))
pa=(physical_activity - n.min(physical_activity)) / (n.max(physical_activity) - n.min(physical_activity))
os=(oxygen_saturation - n.min(oxygen_saturation)) / (n.max(oxygen_saturation) - n.min(oxygen_saturation))
s=(symptoms - n.min(symptoms)) / (n.max(symptoms) - n.min(symptoms))
age=(age - n.min(age)) / (n.max(age) - n.min(age))
gen=(gender - n.min(gender)) / (n.max(gender) - n.min(gender))
y=(heart_complexity_probability - n.min(heart_complexity_probability)) / (n.max(heart_complexity_probability) - n.min(heart_complexity_probability))


w1=0
w2=0
w3=0

w4=0
w5=0
w6=0

w7=0
w8=0
w9=0

w10=0
w11=0

b=0
lr=0.01


def predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b):
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


  return (z1+z2+z3+z4+z5+z6+z7+z8+z9+z10+z11+b)

def cost_func(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b,y):

  y2=predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)

  mse=(y2-y)**2

  return n.mean(mse)


def update(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b,lr):

  dLdW=n.mean(-2*bp1*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))
  dLdW2=n.mean(-2*bp2*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))
  dLdW3=n.mean(-2*hr*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))

  dLdW4=n.mean(-2*col*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))
  dLdW5=n.mean(-2*gl*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))
  dLdW6=n.mean(-2*bmi*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))

  dLdW7=n.mean(-2*pa*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))
  dLdW8=n.mean(-2*os*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))
  dLdW9=n.mean(-2*s*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))

  dLdW10=n.mean(-2*age*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))
  dLdW11=n.mean(-2*gen*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))

  dLdb=n.mean(-2*(y-predict(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b)))

  w1=w1-lr*dLdW
  w2=w2-lr*dLdW2
  w3=w3-lr*dLdW3

  w4=w4-lr*dLdW4
  w5=w5-lr*dLdW5
  w6=w6-lr*dLdW6

  w7=w7-lr*dLdW7
  w8=w8-lr*dLdW8
  w9=w9-lr*dLdW9

  w10=w10-lr*dLdW10
  w11=w11-lr*dLdW11
  b=b-lr*dLdb
  return w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b




def train(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,y,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b,lr,tol=1e-13,n_epochs=100, verbose=False):
  weights1=[w1]
  weights2=[w2]
  weights3=[w3]

  weights4=[w4]
  weights5=[w5]
  weights6=[w6]

  weights7=[w7]
  weights8=[w8]
  weights9=[w9]

  weights10=[w10]
  weights11=[w11]

  biases=[b]
  costs=[]

  ct=1
  while True:
    cost=cost_func(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b,y)
    costs.append(cost)
    if len(costs)>1 and abs(costs[-2]-costs[-1]) < tol:
      break



    w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b=update(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b,lr)

    weights1.append(w1)
    weights2.append(w2)
    weights3.append(w3)

    weights4.append(w4)
    weights5.append(w5)
    weights6.append(w6)

    weights7.append(w7)
    weights8.append(w8)
    weights9.append(w9)

    weights10.append(w10)
    weights11.append(w11)

    biases.append(b)



  return weights1,weights2,weights3,weights4,weights5,weights6,weights7,weights8,weights9,weights10,weights11,biases,costs





w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b,c=train(bp1,bp2,hr,col,gl,bmi,pa,os,s,age,gen,y,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b,lr, verbose=True)

print("Thank you for the patience :) \n")

name=[]
bp_systolic2 = []
bp_diastolic2 = []
heart_rate2 = []
cholesterol_level2 = []
glucose_level2 = []
BMI2 = []
physical_activity2 = []
oxygen_saturation2 = []
symptoms2 = []
age2= []
gen2=[]
num_cases2 = int(input("Enter number of patients\n"))

for i in range(1,num_cases2+1,1):
  a1=input("Enter patient "+str(i)+" name\n")
  a11=int(input("Enter "+a1+"'s Age\n"))
  a12=int(input("Enter "+a1+"'s Gender \n Enter 1 for Male \n Enter 2 for Female \n Enter 3 for others \n"))
  a2=int(input("Enter "+a1+"'s Systolic blood pressure\n"))
  a3=int(input("Enter "+a1+"'s diastolic blood pressure\n"))
  a4=int(input("Enter "+a1+"'s Pulse\n"))
  a5=int(input("Enter "+a1+"'s Cholestrol Level\n")) 
  a6=int(input("Enter "+a1+"'s Glucose Level\n"))
  a7=int(input("Enter "+a1+"'s BMI\n"))
  a8=int(input("Enter "+a1+"'s Physical activity ranging from 1 to 10\n"))
  a9=int(input("Enter "+a1+"'s Oxygen saturation level\n"))
  a10=int(input("Enter "+a1+"'s symptoms info based upon \n Enter 1 for chest pain \n Enter 2 for shortness of breath \n Enter 3 for palpitations \n Enter 4 for chest pain and shortness of breath \n Enter 5 for shortness of breath and palpitations \n Enter 6 for chest pain and palpitations \n Enter 7 for all of them\n"))      

  name.append(a1)
  bp_systolic2.append(a2)
  bp_diastolic2.append(a3)
  heart_rate2.append(a4)
  cholesterol_level2.append(a5)
  glucose_level2.append(a6)
  BMI2.append(a7)
  physical_activity2.append(a8)
  oxygen_saturation2.append(a9)
  symptoms2.append(a10)
  age2.append(a11)
  gen2.append(a12)

def safe_normalize(arr):
    min_val = n.min(arr)
    max_val = n.max(arr)
    if min_val == max_val:
        return n.zeros_like(arr)  
    else:
        return (arr - min_val) / (max_val - min_val)

bp_systolic2=n.array(bp_systolic2)
bp_diastolic2=n.array(bp_diastolic2)
heart_rate2=n.array(heart_rate2)
cholesterol_level2=n.array(cholesterol_level2)
glucose_level2=n.array(glucose_level2)
BMI2=n.array(BMI2)
physical_activity2=n.array(physical_activity2)
oxygen_saturation2=n.array(oxygen_saturation2)
symptoms2=n.array(symptoms2)
age2=n.array(age2)
gen2=n.array(gen2)



bp12 = safe_normalize(bp_systolic2)
bp22 = safe_normalize(bp_diastolic2)
hr2 = safe_normalize(heart_rate2)
col2 = safe_normalize(cholesterol_level2)
gl2 = safe_normalize(glucose_level2)
bmi2 = safe_normalize(BMI2)
pa2 = safe_normalize(physical_activity2)
os2 = safe_normalize(oxygen_saturation2)
s2 = safe_normalize(symptoms2)
age2 = safe_normalize(age2)
gen2 = safe_normalize(gen2)


y_res=predict(bp12,bp22,hr2,col2,gl2,bmi2,pa2,os2,s2,age2,gen2,w1[-1],w2[-1],w3[-1],w4[-1],w5[-1],w6[-1],w7[-1],w8[-1],w9[-1],w10[-1],w11[-1],b[-1])

y_res=y_res*100
print("\n⚠️ IMPORTANT NOTE:")
print("In this assessment, a higher Cardiovascular Index indicates a higher risk of heart problems.")
print("The higher the score, the more attention should be paid to the patient's cardiovascular health.")
print("\n\n")
for i in range(1,num_cases2+1,1):
   print(f"\n🫀 CARDIAC FITNESS REPORT")
   print(f"Patient Name: {name[i-1]}")
   print(f"Cardiovascular Health Index: {y_res[i-1]} %\n")

print("\n--- Further Assessment ---")
print("Would you like to explore potential common heart complications based on this assessment?")
print("\nIMPORTANT DISCLAIMER:")
print("This information is for educational purposes only and does not constitute a medical diagnosis.")
print("For accurate health information and personalized advice, please consult with a qualified healthcare professional.\n")

response = input("Enter 'Y' for Yes or 'N' for No: ")
print("\n\n")
if response=='Y' or 'y':
   for i in range(1,num_cases2+1,1):
      print("For patient "+name[i-1])
      if y_res[i-1]>=0 and y_res[i-1]<=10:
         print("potential risk of Coronary Artery Disease (CAD)")
         print("For futhur evaluation please contact a health care professional !\n\n")

      elif y_res[i-1]>10 and y_res[i-1]<=20:
         print("potential risk of Hypertension (High Blood Pressure)")
         print("For futhur evaluation please contact a health care professional !\n\n")

      elif y_res[i-1]>20 and y_res[i-1]<=30:
         print("potential risk of Heart Failure")
         print("For futhur evaluation please contact a health care professional !\n\n")

      elif y_res[i-1]>30 and y_res[i-1]<=40:
         print("potential risk of Arrhythmias (Irregular heartbeats)")
         print("For futhur evaluation please contact a health care professional !\n\n")

      elif y_res[i-1]>40 and y_res[i-1]<=50:
         print("potential risk of Heart Valve Disease")
         print("For futhur evaluation please contact a health care professional !\n\n")
         
      elif y_res[i-1]>50 and y_res[i-1]<=60:
         print("potential risk of Myocardial Infarction (Heart Attack)")
         print("For futhur evaluation please contact a health care professional !\n\n")

      elif y_res[i-1]>60 and y_res[i-1]<=70:
         print("potential risk of Atrial Fibrillation")
         print("For futhur evaluation please contact a health care professional !\n\n")

      else:
         print("potential risk of Angina Pectoris")
         print("For futhur evaluation please contact a health care professional !\n\n")


print("I hope this information is helpful!")
