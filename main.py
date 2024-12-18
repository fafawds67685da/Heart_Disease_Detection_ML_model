from fastapi import FastAPI, File, UploadFile
import pandas as p
import numpy as n
import matplotlib.pyplot as m
import io
from io import BytesIO
from fastapi.responses import StreamingResponse
from matplotlib.backends.backend_pdf import PdfPages

app = FastAPI()

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

@app.get('/')
def default():
    return {'App' : 'Running'}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = p.read_csv(io.BytesIO(contents))
    df.columns = ['Age','Sex','Chest_pain','trestbps','cholestrol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','targets']
    df['predictions'] = w1*df['Age'] + w2*df['Sex'] + w3*df['Chest_pain'] + w4*df['trestbps'] + w5*df['cholestrol'] + w6*df['fbs'] + w7*df['restecg'] + w8*df['thalach'] + w9*df['exang'] + w10*df['oldpeak'] + w11*df['slope'] + w12*df['ca'] + w13*df['thal'] + b
    output = df.to_csv(index=False).encode('utf-8')
    return StreamingResponse(io.BytesIO(output),
                             media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})

@app.post("/plot/")
async def plot(file: UploadFile = File(...)):
    contents = await file.read()
    df = p.read_csv(io.BytesIO(contents))
    if df.shape[1] == 15:
        df = df.iloc[:, 1:]  # Drop the first column


    df.columns = ['Age','Sex','Chest_pain','trestbps','cholestrol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','targets']
    df['predictions'] = w1*df['Age'] + w2*df['Sex'] + w3*df['Chest_pain'] + w4*df['trestbps'] + w5*df['cholestrol'] + w6*df['fbs'] + w7*df['restecg'] + w8*df['thalach'] + w9*df['exang'] + w10*df['oldpeak'] + w11*df['slope'] + w12*df['ca'] + w13*df['thal'] + b
    rmse_score = n.sqrt(n.mean(n.square(df['predictions'] - df['targets'])))

    abc = ['Age','Sex','Chest_pain','trestbps','cholestrol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    
    # Create a PDF object
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        for i in range(13):
            m.figure(figsize=(10, 6))
            m.scatter(df[abc[i]], df['targets'], color='royalblue', label='Actual Targets', marker='x')
            m.plot(df[abc[i]], df['predictions'], color='k', label='Predictions', linewidth=2)
            m.title(f'Linear Regression For {abc[i]} (RMSE : {round(rmse_score, 3)})', color='maroon', fontsize=15)
            m.xlabel(abc[i], color='m', fontsize=13)
            m.ylabel('HD predictions', color='m', fontsize=13)
            m.legend()

            # Save each plot to the PDF
            pdf.savefig()
            m.close()  # Close the figure to prevent excessive memory use

    # Seek the buffer to the beginning before returning
    pdf_buffer.seek(0)

    # Return the PDF as a streaming response
    return StreamingResponse(pdf_buffer,
                             media_type="application/pdf",
                             headers={"Content-Disposition": "attachment; filename=plots.pdf"})
