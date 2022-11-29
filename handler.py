import pickle
import pandas as pd
from flask import Flask, Response, request 
from rossmann.Rossmann import Rossmann
import os 

# loading model
model = pickle.load (open('model/rossman.pkl', 'rb'))
# quando a API iniciar ela carregara o modelo em memoria e com endpoint ativo
# ae receber uma requisao, ela prepara os dados, passa pro modelo, recebe a previsao, anexa nos dados 
# passados pelo usuario, e retorna


#initialize API
app = Flask ( __name__ )

#criando o endpoint que vai receber a request
@app.route ('/rossmann/predict', methods = ['POST'])
def rossmann_predict ():
    test_json = request.get_json()
    
    if test_json:#there is data
        if isinstance (test_json, dict): #Unique example
            test_raw = pd.DataFrame(test_json, index=[0])
            #nesse teste, verifica-se se o arquivo é um dicionario com uma cahve:valor
            
        else: #Multiple examples
            test_raw  = pd.DataFrame (test_json, columns = test_json[0].keys())
            # se o dado json vier em forma de varias chaves:valores, nao apenas um, 
            # sera coletado apenas o primeiro par, sendo que a coluna sera dada por keys()
                
        # Instatiate Rossmann class
        pipeline = Rossmann()
        # com isso tenho acesso a todos os metodos de Rossmann
            
        # data cleaning
        df1 = pipeline.data_cleaning (test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering (df1)
        
        # data preparation
        df3 = pipeline.data_preparation (df2)
        
        # prediction
        df_response = pipeline.get_prediction ( model, test_raw, df3 )
        # metodo recebe o modelo, o dado transformado e o dado real
        
        return df_response
    
    
    else:
        return Response ('{}', status=200, mimetype='application/json')
        #teste para ver se o dado existe e a execucao deu certo 

        
        
        
if __name__ == '__main__':
    port = os.environ.get ('PORT', 5000)
    #para setar a porta 5000, que é a padrao do FLASK
    app.run(host = '0.0.0.0', port = port)
    # sequencia de 0 indica que o app esta rodando em ambiente local